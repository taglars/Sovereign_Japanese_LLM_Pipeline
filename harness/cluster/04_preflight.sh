#!/usr/bin/env bash
# Checks before queueing a job. Login node, no GPU, no queue.
#
#   bash cluster/04_preflight.sh deepseek-v4-flash
#
# Checks that the container starts with the expected entrypoint, the weights are
# staged and complete, and the config names a real image for this model. vLLM
# needs a GPU to build its argument parser, so the arguments themselves are only
# checked by the job.

source "$(dirname "$0")/env.sh"
set -uo pipefail

JUDGE="${1:-}"
[ -n "$JUDGE" ] || { echo "usage: $0 <judge-name>" >&2; exit 2; }

CONFIG="${JV_CODE}/config/experiment.yaml"
[ -f "$CONFIG" ] || { echo "ERROR: no config at $CONFIG" >&2; exit 1; }

read -r HF_ID DTYPE QUANT TP DP GPU_UTIL MAX_LEN NEEDS_IMAGE <<EOF
$(python3 - "$CONFIG" "$JUDGE" <<'SCALARS'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
# Generators (Experiment 3) are served the same way; check them the same way.
for judge in config["judges"] + config.get("generators", []):
    if judge["name"] == sys.argv[2]:
        print(judge["hf_id"], judge.get("dtype", "auto"),
              judge.get("quantization", "none"),
              judge.get("tensor_parallel", 4),
              judge.get("data_parallel", 1),
              judge.get("gpu_memory_utilization", 0.90),
              judge.get("max_model_len", 32768),
              judge.get("needs_image", "stable"))
        break
else:
    sys.exit(f"no judge or generator named {sys.argv[2]!r}")
SCALARS
)
EOF

EXTRA_RAW="$(python3 - "$CONFIG" "$JUDGE" <<'LISTARGS'
import shlex, sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
for judge in config["judges"] + config.get("generators", []):
    if judge["name"] == sys.argv[2]:
        print(" ".join(shlex.quote(str(a)) for a in judge.get("extra_args", [])))
        break
LISTARGS
)"
EXTRA_ARGS=()
[ -n "$EXTRA_RAW" ] && eval "EXTRA_ARGS=($EXTRA_RAW)"

MODEL_DIR="$(jv_model_dir "$HF_ID")"
# Judges may need different vLLM builds; the config says which.
JV_SQSH="$(jv_image_for "$NEEDS_IMAGE")"

# Keep identical to 02_judge.sbatch.
VLLM_ARGS=(
    "$MODEL_DIR"
    --served-model-name "$HF_ID"
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization "$GPU_UTIL"
    --max-model-len "$MAX_LEN"
    --port "$JV_PORT"
    --host 127.0.0.1
)
[ "$DTYPE" != "auto" ] && VLLM_ARGS+=(--dtype "$DTYPE")
if [ "$DP" -gt 1 ]; then VLLM_ARGS+=(--data-parallel-size "$DP"); fi
VLLM_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

echo "=============================================================="
echo "judge      $JUDGE"
echo "model dir  $MODEL_DIR"
echo "image      $JV_SQSH ($NEEDS_IMAGE)"
echo "serve args ${VLLM_ARGS[*]}"
echo "=============================================================="
echo

[ -s "$JV_SQSH" ] || {
    echo "ERROR: no container at $JV_SQSH" >&2
    echo "       this judge needs the '$NEEDS_IMAGE' image:" >&2
    echo "         bash cluster/00_setup.sh $NEEDS_IMAGE" >&2
    exit 1; }
[ -f "$MODEL_DIR/config.json" ] || {
    echo "ERROR: weights not staged at $MODEL_DIR" >&2; exit 1; }

# NVIDIA_VISIBLE_DEVICES=void skips enroot's GPU hook, which aborts without a GPU.
ENROOT_RUN=(enroot start --root --rw --mount "$JV_WS:$JV_WS"
            --env HF_HOME="$HF_HOME"
            --env NVIDIA_VISIBLE_DEVICES=void
            --env NVIDIA_DRIVER_CAPABILITIES=""
            "$JV_SQSH")

# The container must start before anything it reports counts.
PROBE="$("${ENROOT_RUN[@]}" --help 2>&1 || true)"
if printf '%s' "$PROBE" | grep -qiE 'nvidia-container-cli|hooks\.d/.*exited|libnvidia-ml'; then
    echo "=============================================================="
    echo "PREFLIGHT INCONCLUSIVE: the container did not start here."
    echo "=============================================================="
    printf '%s\n' "$PROBE" | tail -6 | sed 's/^/  /'
    echo
    echo "enroot's NVIDIA hook failed even with NVIDIA_VISIBLE_DEVICES=void."
    echo "Nothing below would be meaningful, so this refuses rather than"
    echo "reporting a pass it cannot support."
    exit 1
fi

# Without a GPU even `--help` raises; checked before the entrypoint branch.
if printf '%s' "$PROBE" | grep -q 'Failed to infer device type'; then
    echo "=============================================================="
    echo "PREFLIGHT PARTIAL"
    echo "=============================================================="
    echo "  container starts       OK"
    echo "  weights staged         OK"
    echo "  image selected         OK  ($NEEDS_IMAGE)"
    echo
    echo "  entrypoint and arguments NOT validated: vLLM needs a GPU even to"
    echo "  build its parser, so neither can be checked from a login node."
    echo "  The job identifies the entrypoint itself at runtime and aborts"
    echo "  within ~2 minutes if an argument is wrong."
    echo "=============================================================="
    echo
    echo "submit with:  make submit-one JUDGE=$JUDGE"
    exit 0
fi

# Entrypoint, from the help text rather than the exit code.
HELP="$PROBE"
if printf '%s' "$HELP" | grep -q 'usage: vllm serve'; then
    SERVE=(); WHICH="vllm serve (options only)"
elif printf '%s' "$HELP" | grep -q '{chat,complete,serve'; then
    SERVE=(serve); WHICH="vllm (adding 'serve')"
else
    # Unrecognised help text means the probe broke: refuse rather than guess.
    echo "PREFLIGHT FAILED: could not identify the entrypoint from --help." >&2
    echo "  expected 'usage: vllm serve' or a '{chat,complete,serve' line." >&2
    # An import traceback names the missing module near its end.
    printf '%s\n' "$HELP" | tail -30 | sed 's/^/  /' >&2
    exit 1
fi
echo "entrypoint : $WHICH"
echo

echo "trying the real command for up to 90s (no GPU here, it will not finish)"
OUT="$(timeout 90 "${ENROOT_RUN[@]}" ${SERVE[@]+"${SERVE[@]}"} \
       "${VLLM_ARGS[@]}" 2>&1 || true)"

echo
# Without a GPU vLLM cannot check the arguments; everything else has passed.
if printf '%s' "$OUT" | grep -q 'Failed to infer device type'; then
    echo "=============================================================="
    echo "PREFLIGHT PARTIAL"
    echo "=============================================================="
    echo "  container starts       OK"
    echo "  entrypoint identified  OK  ($WHICH)"
    echo "  weights staged         OK"
    echo "  image selected         OK  ($NEEDS_IMAGE)"
    echo
    echo "  arguments NOT validated: vLLM cannot build its parser without a"
    echo "  GPU, so this cannot be checked from a login node. The job aborts"
    echo "  within ~2 minutes if an argument is wrong, so submitting is cheap."
    echo "=============================================================="
    echo
    echo "submit with:  make submit-one JUDGE=$JUDGE"
    exit 0
fi

if printf '%s' "$OUT" | grep -qE 'error: (argument|unrecognized|the following)'; then
    echo "=============================================================="
    echo "PREFLIGHT FAILED: vLLM rejected the command line."
    echo "=============================================================="
    printf '%s\n' "$OUT" | grep -E 'error:|usage:|invalid choice' | head -12
    echo
    echo "Fix config/experiment.yaml, then run this again. Do NOT submit."
    exit 1
fi

if printf '%s' "$OUT" | grep -qiE 'no.*(gpu|cuda|device)|device_count|not available|assert' ; then
    VERDICT="arguments accepted; stopped at hardware, as expected on a login node"
else
    VERDICT="arguments accepted; no argparse error seen"
fi

echo "=============================================================="
echo "PREFLIGHT PASSED"
echo "  $VERDICT"
echo "=============================================================="
echo
echo "last lines from the attempt, for context:"
printf '%s\n' "$OUT" | tail -8 | sed 's/^/  /'
echo
echo "submit with:  make short JUDGE=$JUDGE"
