#!/usr/bin/env bash
#SBATCH --job-name=jv-judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480gb
#SBATCH --output=%x-%j.out
#
# --mem is required: the default host RAM per CPU is too small to load the weights.
# 480gb fits every partition's cap; cluster/sbatch_args.py passes the value per judge.
#
# One judge, one node, every script under both rubric languages.
#
#   sbatch -p gpu_h100 --gres=gpu:4 -t 08:00:00 \
#          cluster/02_judge.sbatch deepseek-v4-flash
#
# The server runs inside the vLLM container; the client runs outside it on the
# same node and talks to 127.0.0.1. Both die with the job.

set -euo pipefail

JUDGE="${1:-}"
[ -n "$JUDGE" ] || { echo "usage: sbatch ... 02_judge.sbatch <judge>" >&2; exit 2; }

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
read -r HF_ID DTYPE QUANT TP GPU_UTIL MAX_LEN NEEDS_IMAGE <<EOF
$(python3 - "$CONFIG" "$JUDGE" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
for judge in config["judges"]:
    if judge["name"] == sys.argv[2]:
        print(judge["hf_id"], judge.get("dtype", "auto"),
              judge.get("quantization", "none"),
              judge.get("tensor_parallel", 4),
              judge.get("gpu_memory_utilization", 0.90),
              judge.get("max_model_len", 32768),
              judge.get("needs_image", "stable"))
        break
else:
    sys.exit(f"no judge named {sys.argv[2]!r} in the config")
PY
)
EOF

# Serve flags specific to this checkpoint, shell-quoted.
EXTRA_RAW="$(python3 - "$CONFIG" "$JUDGE" <<'YAMLARGS'
import shlex, sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
for judge in config["judges"]:
    if judge["name"] == sys.argv[2]:
        print(" ".join(shlex.quote(str(a)) for a in judge.get("extra_args", [])))
        break
YAMLARGS
)"
EXTRA_ARGS=()
[ -n "$EXTRA_RAW" ] && eval "EXTRA_ARGS=($EXTRA_RAW)"

MODEL_DIR="$(jv_model_dir "$HF_ID")"
# Judges may need different vLLM builds; the config says which.
JV_SQSH="$(jv_image_for "$NEEDS_IMAGE")"
# Optional overrides for another corpus and store (Experiment 3).
DB="${JV_DB:-$JV_OUT/${JUDGE}.db}"
LOG="$JV_OUT/${JUDGE}-vllm-${SLURM_JOB_ID:-local}.log"

echo "=============================================================="
echo "judge      $JUDGE"
echo "model      $HF_ID"
echo "staged at  $MODEL_DIR"
echo "image      $JV_SQSH ($NEEDS_IMAGE)"
echo "node       $(hostname)   gpus ${SLURM_GPUS_ON_NODE:-?}"
echo "store      $DB"
echo "vllm log   $LOG"
echo "=============================================================="

[ -f "$MODEL_DIR/config.json" ] || {
    echo "ERROR: weights not staged. Run on a login node first:" >&2
    echo "  bash cluster/01_stage_weights.sh $JUDGE" >&2
    exit 1
}

# FP8 needs Hopper; refuse before loading.
if [ "$QUANT" = "fp8" ]; then
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "$CAP" ] && [ "$CAP" -lt 90 ]; then
        echo "ERROR: $JUDGE is FP8 and this node reports compute capability" >&2
        echo "       ${CAP:0:1}.${CAP:1}. FP8 has no kernel below 9.0 (Hopper)." >&2
        echo "       Submit to -p gpu_h100, or set dtype: bfloat16 in the config." >&2
        exit 1
    fi
fi

# A chained link with nothing left to do exits before loading the model.
LEFT="$(python3 -m jv run --judge "$JUDGE" --db "$DB" --dry-run ${JV_RUN_ARGS:-} 2>/dev/null | tail -1 || true)"
echo "cells left: ${LEFT:-unknown}"
if [ "$LEFT" = "0" ]; then
    echo "nothing left for $JUDGE; exiting before loading the model"
    exit 0
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
# ${arr[@]+...} so an empty array does not trip `set -u`.
VLLM_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
echo "serve args: ${VLLM_ARGS[*]}"

if command -v enroot >/dev/null 2>&1 && [ -s "$JV_SQSH" ]; then
    echo "runtime: enroot"
    ENROOT_RUN=(enroot start --root --rw --mount "$JV_WS:$JV_WS"
                --env HF_HOME="$HF_HOME" "$JV_SQSH")
    # The image's entrypoint is `vllm serve`; detect it from the text of --help,
    # since the exit code is 0 either way.
    HELP="$("${ENROOT_RUN[@]}" --help 2>&1 || true)"
    if printf '%s' "$HELP" | grep -q 'usage: vllm serve'; then
        echo "entrypoint: vllm serve (passing options only)"
        SERVE=()
    elif printf '%s' "$HELP" | grep -q '{chat,complete,serve'; then
        echo "entrypoint: vllm (adding 'serve')"
        SERVE=(serve)
    else
        # Unrecognised help text means the probe broke: refuse rather than guess.
        echo "ERROR: could not identify the container entrypoint." >&2
        printf '%s\n' "$HELP" | head -10 >&2
        exit 1
    fi
    "${ENROOT_RUN[@]}" ${SERVE[@]+"${SERVE[@]}"} "${VLLM_ARGS[@]}" > "$LOG" 2>&1 &
elif [ -n "${JV_SIF:-}" ] && [ -f "${JV_SIF:-}" ]; then
    echo "runtime: apptainer"
    RUNNER=$(command -v apptainer || command -v singularity)
    "$RUNNER" exec --nv --bind "$JV_WS:$JV_WS" --env HF_HOME="$HF_HOME" \
        "$JV_SIF" vllm serve "${VLLM_ARGS[@]}" > "$LOG" 2>&1 &   # apptainer exec bypasses ENTRYPOINT
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

# The server must die with the job, including on scancel and on walltime.
cleanup() {
    echo "stopping vLLM (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- run the client --------------------------------------------------------
#
# The client runs in the background; whichever of client and server dies first
# ends the job.

set +e
python3 -m jv run \
    --judge "$JUDGE" \
    --base-url "http://127.0.0.1:$JV_PORT" \
    --db "$DB" \
    --ready-timeout 1500 ${JV_RUN_ARGS:-} &
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
python3 -m jv status --db "$DB" || true

if [ "$STATUS" -ne 0 ]; then
    echo "client exited $STATUS. vLLM log tail:" >&2
    tail -40 "$LOG" >&2
fi
exit "$STATUS"
