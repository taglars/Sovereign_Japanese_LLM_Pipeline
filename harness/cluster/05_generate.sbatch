#!/usr/bin/env bash
#SBATCH --job-name=jv-generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480gb
#SBATCH --output=%x-%j.out
#
# Experiment 3: generate every synthetic customer's script, EN and JA prompts.
#
#   sbatch $(python3 cluster/sbatch_args.py qwen3.8-27b) \
#          cluster/05_generate.sbatch qwen3.8-27b          # full run
#   sbatch $(JV_TIME=00:30:00 python3 cluster/sbatch_args.py qwen3.8-27b gpu_h100_short) \
#          cluster/05_generate.sbatch qwen3.8-27b 3        # smoke: 3 customers
#   sbatch ... cluster/05_generate.sbatch qwen3.8-27b classify
#          same server, files Exp 4b feedback points into the codebook
#
# Serving as in 02_judge.sbatch; only the config lookup, the data-parallel flag and
# the client differ. Resumable: resubmit and it continues from the store.

set -euo pipefail

MODEL="${1:-}"
LIMIT="${2:-0}"
[ -n "$MODEL" ] || { echo "usage: sbatch ... 05_generate.sbatch <generator> [limit]" >&2; exit 2; }

# Slurm copies the script to its spool dir; resolve the repo from the submit dir.
JV_HOME="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)}"
if [ ! -f "$JV_HOME/cluster/env.sh" ]; then
    echo "ERROR: cannot find cluster/env.sh from '$JV_HOME'." >&2
    echo "       Submit from the repo root: cd judge-validation && sbatch ..." >&2
    exit 1
fi
export JV_CODE="${JV_CODE:-$JV_HOME}"
source "$JV_HOME/cluster/env.sh"
set -euo pipefail
cd "$JV_CODE"

CONFIG="$JV_CODE/config/experiment.yaml"
read -r HF_ID DTYPE QUANT TP DP GPU_UTIL MAX_LEN NEEDS_IMAGE <<EOF
$(python3 - "$CONFIG" "$MODEL" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
for spec in config.get("generators", []):
    if spec["name"] == sys.argv[2]:
        print(spec["hf_id"], spec.get("dtype", "auto"),
              spec.get("quantization", "none"),
              spec.get("tensor_parallel", 1),
              spec.get("data_parallel", 1),
              spec.get("gpu_memory_utilization", 0.90),
              spec.get("max_model_len", 32768),
              spec.get("needs_image", "stable"))
        break
else:
    sys.exit(f"no generator named {sys.argv[2]!r} in the config")
PY
)
EOF

EXTRA_RAW="$(python3 - "$CONFIG" "$MODEL" <<'YAMLARGS'
import shlex, sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
for spec in config.get("generators", []):
    if spec["name"] == sys.argv[2]:
        print(" ".join(shlex.quote(str(a)) for a in spec.get("extra_args", [])))
        break
YAMLARGS
)"
EXTRA_ARGS=()
[ -n "$EXTRA_RAW" ] && eval "EXTRA_ARGS=($EXTRA_RAW)"

MODEL_DIR="$(jv_model_dir "$HF_ID")"
JV_SQSH="$(jv_image_for "$NEEDS_IMAGE")"
mkdir -p "$JV_OUT/exp3"
DB="$JV_OUT/exp3/generation.db"
LOG="$JV_OUT/exp3/${MODEL}-vllm-${SLURM_JOB_ID:-local}.log"

echo "=============================================================="
echo "generator  $MODEL"
echo "model      $HF_ID   tp $TP   dp $DP"
echo "staged at  $MODEL_DIR"
echo "image      $JV_SQSH ($NEEDS_IMAGE)"
echo "node       $(hostname)   gpus ${SLURM_GPUS_ON_NODE:-?}"
echo "store      $DB"
echo "mode       $LIMIT (0 = all customers, N = first N, classify = Exp 4b)"
echo "vllm log   $LOG"
echo "=============================================================="

[ -f "$MODEL_DIR/config.json" ] || {
    echo "ERROR: weights not staged. Run on a login node first:" >&2
    echo "  bash cluster/01_stage_weights.sh $MODEL" >&2
    exit 1
}

if [ "$LIMIT" = "classify" ]; then
    ITEMS="$JV_CODE/data/codebook/items.jsonl"
    [ -s "$ITEMS" ] || { echo "ERROR: no $ITEMS. Run: make codebook-items" >&2; exit 1; }
    mkdir -p "$JV_OUT/codebook"
    CLIENT=(python3 -m jv classify --base-url "http://127.0.0.1:$JV_PORT"
            --model "$HF_ID" --items "$ITEMS" --db "$JV_OUT/codebook/labels.db"
            --ready-timeout 1500)
    STATUS_CMD=(true)
else
    # The JA arm needs a complete translation table; checked before the server starts.
    python3 -c 'import sys; sys.path.insert(0, "."); from gen import templates as T; T.load_table(); print("translation table: complete")' || exit 1
    # A chained link with nothing left to do exits before loading the model.
    LEFT="$(python3 -m gen pending --db "$DB" --limit "$LIMIT" 2>/dev/null | tail -1 || true)"
    echo "chains left: ${LEFT:-unknown}"
    if [ "$LEFT" = "0" ]; then
        echo "nothing left to generate; exiting before loading the model"
        exit 0
    fi
    CLIENT=(python3 -m gen run --base-url "http://127.0.0.1:$JV_PORT"
            --model "$HF_ID" --db "$DB" --limit "$LIMIT" --ready-timeout 1500)
    STATUS_CMD=(python3 -m gen status --db "$DB")
fi

# ---- start the server ------------------------------------------------------

VLLM_ARGS=(
    # Positional, not --model: serve deprecates the option and warns on it.
    "$MODEL_DIR"
    --served-model-name "$HF_ID"
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization "$GPU_UTIL"
    --max-model-len "$MAX_LEN"
    --port "$JV_PORT"
    --host 127.0.0.1
)
[ "$DTYPE" != "auto" ] && VLLM_ARGS+=(--dtype "$DTYPE")
# Data parallelism only when the config asks for it.
if [ "$DP" -gt 1 ]; then VLLM_ARGS+=(--data-parallel-size "$DP"); fi
# ${arr[@]+...} so an empty array does not trip `set -u`.
VLLM_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
echo "serve args: ${VLLM_ARGS[*]}"

if command -v enroot >/dev/null 2>&1 && [ -s "$JV_SQSH" ]; then
    echo "runtime: enroot"
    ENROOT_RUN=(enroot start --root --rw --mount "$JV_WS:$JV_WS"
                --env HF_HOME="$HF_HOME" "$JV_SQSH")
    # Entrypoint from the text of --help, as in 02_judge.sbatch.
    HELP="$("${ENROOT_RUN[@]}" --help 2>&1 || true)"
    if printf '%s' "$HELP" | grep -q 'usage: vllm serve'; then
        echo "entrypoint: vllm serve (passing options only)"
        SERVE=()
    elif printf '%s' "$HELP" | grep -q '{chat,complete,serve'; then
        echo "entrypoint: vllm (adding 'serve')"
        SERVE=(serve)
    else
        echo "ERROR: could not identify the container entrypoint." >&2
        printf '%s\n' "$HELP" | head -10 >&2
        exit 1
    fi
    "${ENROOT_RUN[@]}" ${SERVE[@]+"${SERVE[@]}"} "${VLLM_ARGS[@]}" > "$LOG" 2>&1 &
elif [ -n "${JV_SIF:-}" ] && [ -f "${JV_SIF:-}" ]; then
    echo "runtime: apptainer"
    RUNNER=$(command -v apptainer || command -v singularity)
    "$RUNNER" exec --nv --bind "$JV_WS:$JV_WS" --env HF_HOME="$HF_HOME" \
        "$JV_SIF" vllm serve "${VLLM_ARGS[@]}" > "$LOG" 2>&1 &
else
    echo "runtime: bare vllm on PATH"
    command -v vllm >/dev/null 2>&1 || {
        echo "ERROR: no container and no vllm on PATH." >&2
        echo "       Run cluster/00_setup.sh on a login node." >&2
        exit 1
    }
    vllm serve "${VLLM_ARGS[@]}" > "$LOG" 2>&1 &
fi
SERVER_PID=$!

cleanup() {
    echo "stopping vLLM (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- run the client --------------------------------------------------------
#
# In the background and watched, so a dead server ends the job.

set +e
"${CLIENT[@]}" &
CLIENT_PID=$!

while kill -0 "$CLIENT_PID" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM exited. Last lines of $LOG:" >&2
        tail -40 "$LOG" >&2
        kill "$CLIENT_PID" 2>/dev/null
        wait "$CLIENT_PID" 2>/dev/null
        STATUS=1
        break
    fi
    sleep 10
done
if [ -z "${STATUS:-}" ]; then
    wait "$CLIENT_PID"
    STATUS=$?
fi
set -e

echo
"${STATUS_CMD[@]}" || true

if [ "$STATUS" -ne 0 ]; then
    echo "client exited $STATUS. vLLM log tail:" >&2
    tail -40 "$LOG" >&2
fi
exit "$STATUS"
