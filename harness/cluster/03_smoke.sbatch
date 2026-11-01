#!/usr/bin/env bash
#SBATCH --job-name=jv-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#
# Five minutes in the dev queue: checks the container and the GPUs. Loads no weights.
#
#   sbatch -p dev_gpu_h100 --gres=gpu:4 cluster/03_smoke.sbatch
#
# dev_gpu_h100 allows 30 minutes and one running job.

set -uo pipefail

# Slurm copies the script to its spool dir; resolve the repo from the submit dir.
JV_HOME="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)}"
if [ ! -f "$JV_HOME/cluster/env.sh" ]; then
    echo "ERROR: cannot find cluster/env.sh from '$JV_HOME'." >&2
    echo "       Submit from the repo root: cd judge-validation && sbatch ..." >&2
    exit 1
fi
export JV_CODE="${JV_CODE:-$JV_HOME}"
source "$JV_HOME/cluster/env.sh"
# No -e: every check runs, even after one fails.
set +e
cd "$JV_CODE"

FAIL=0
check() {   # check <description> <command...>
    local label="$1"; shift
    if "$@" >/tmp/jv-smoke.$$ 2>&1; then
        echo "  PASS  $label"
        sed 's/^/          /' /tmp/jv-smoke.$$ | head -6
    else
        echo "  FAIL  $label"
        sed 's/^/          /' /tmp/jv-smoke.$$ | head -12
        FAIL=$((FAIL + 1))
    fi
    rm -f /tmp/jv-smoke.$$
}

echo "=============================================================="
echo "node        $(hostname)"
echo "job         ${SLURM_JOB_ID:-none}"
echo "partition   ${SLURM_JOB_PARTITION:-none}"
echo "gpus        ${SLURM_GPUS_ON_NODE:-unset}"
echo "cpus        ${SLURM_CPUS_ON_NODE:-unset}"
echo "mem         ${SLURM_MEM_PER_NODE:-unset} MB"
echo "workspace   $JV_WS"
echo "=============================================================="
echo

echo "-- host --"
check "nvidia-smi sees GPUs" \
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
check "python3 present" python3 --version
check "PyYAML importable (the client needs it)" python3 -c "import yaml"
check "jv package imports" python3 -c "import jv, jv.rubric, jv.analyze"
check "corpus is readable" python3 -c "
from jv import corpus
s = corpus.load('data/corpus.json')
print(f'{len(s)} scripts, {len({x.arm for x in s})} arms')"

echo
echo "-- host memory actually granted --"
free -g | sed 's/^/  /'
echo "  (vLLM needs enough host RAM to stage the checkpoint. 700 GB is"
echo "   requested by 02_judge.sbatch for exactly this reason.)"

echo
echo "-- gpu compute capability --"
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
echo "  compute_cap = ${CAP:-unknown}"
case "$CAP" in
    9.*|10.*) echo "  Hopper or newer: FP8 works, deepseek-v4-flash can run here." ;;
    8.*)      echo "  Ampere: NO FP8 kernel. deepseek-v4-flash cannot run here." ;;
    *)        echo "  could not determine; check the partition." ;;
esac

echo
echo "-- container --"
# Arguments to `enroot start` go to the image's vllm entrypoint, so the GPUs are
# checked on the host and the container through its entrypoint.
ENROOT_RUN=(enroot start --root --rw --mount "$JV_WS:$JV_WS" "$JV_SQSH")
if command -v enroot >/dev/null 2>&1 && [ -s "$JV_SQSH" ]; then
    check "container mounts, vllm runs" "${ENROOT_RUN[@]}" collect-env
    check "vllm serve parses arguments" "${ENROOT_RUN[@]}" serve --help

    # Which invocation the real job must use.
    if "${ENROOT_RUN[@]}" serve --help >/dev/null 2>&1; then
        echo "  entrypoint IS vllm  -> job runs: enroot start IMG serve ..."
    else
        echo "  no vllm entrypoint  -> job runs: enroot start IMG vllm serve ..."
    fi

    echo
    echo "-- does this build know deepseek_v4? --"
    ARCH=$(python3 -c "
import json
print(json.load(open('$JV_MODELS/deepseek-ai__DeepSeek-V4-Flash/config.json'))
      .get('architectures', ['?'])[0])" 2>/dev/null)
    echo "  checkpoint architecture: ${ARCH:-unknown}"
    "${ENROOT_RUN[@]}" collect-env 2>&1 | grep -iE "^vllm|version" | head -4 \
        | sed 's/^/  /'
elif [ -n "${JV_SIF:-}" ] && [ -f "${JV_SIF:-}" ]; then
    RUNNER=$(command -v apptainer || command -v singularity)
    check "apptainer image, vllm version" "$RUNNER" exec --nv "$JV_SIF" vllm --version
else
    echo "  SKIP  no container image found at $JV_SQSH"
    echo "        run cluster/00_setup.sh on a login node"
    FAIL=$((FAIL + 1))
fi

echo
echo "-- staged weights --"
for dir in "$JV_MODELS"/*/; do
    [ -d "$dir" ] || continue
    if [ -f "$dir/config.json" ]; then
        echo "  PASS  $(basename "$dir")  $(du -sh "$dir" 2>/dev/null | cut -f1)"
    else
        echo "  FAIL  $(basename "$dir") has no config.json, download incomplete"
        FAIL=$((FAIL + 1))
    fi
done
[ -n "$(ls -A "$JV_MODELS" 2>/dev/null)" ] || {
    echo "  none staged yet: bash cluster/01_stage_weights.sh <judge>"
    FAIL=$((FAIL + 1))
}

echo
echo "=============================================================="
if [ "$FAIL" -eq 0 ]; then
    echo "SMOKE PASSED. Submit the real job:"
    echo "  sbatch -p gpu_h100 --gres=gpu:4 -t 08:00:00 \\"
    echo "         cluster/02_judge.sbatch deepseek-v4-flash"
else
    echo "SMOKE FAILED: $FAIL check(s). Fix these before the real job."
fi
echo "=============================================================="
exit "$FAIL"
