#!/usr/bin/env bash
# Shared settings, sourced by every other script here. Only the workspace tooling
# and the partition names are specific to bwUniCluster.

# No `set -e`: this file is sourced, and each caller sets its own options.

# Weights and results go in the workspace; $HOME has a small quota.
JV_WS_NAME="${JV_WS_NAME:-judge-validation}"
JV_WS_DAYS="${JV_WS_DAYS:-60}"

# Resolve the workspace, creating it the first time.
if command -v ws_find >/dev/null 2>&1; then
    if ! JV_WS="$(ws_find "$JV_WS_NAME" 2>/dev/null)" || [ -z "$JV_WS" ]; then
        echo "creating workspace $JV_WS_NAME for $JV_WS_DAYS days"
        JV_WS="$(ws_allocate "$JV_WS_NAME" "$JV_WS_DAYS")"
        JV_WS="$(ws_find "$JV_WS_NAME")"
    fi
else
    # Not on bwUniCluster. Fall back to a directory so the scripts stay testable.
    JV_WS="${JV_WS:-$HOME/judge-validation-ws}"
    mkdir -p "$JV_WS"
fi
export JV_WS

export JV_MODELS="${JV_MODELS:-$JV_WS/models}"
export JV_IMAGES="${JV_IMAGES:-$JV_WS/images}"
export JV_OUT="${JV_OUT:-$JV_WS/out}"
export JV_CODE="${JV_CODE:-$HOME/judge-validation}"
mkdir -p "$JV_MODELS" "$JV_IMAGES" "$JV_OUT"

# The Hugging Face token, if any, from ~/.hf_token (mode 600), never from the repo.
if [ -z "${HF_TOKEN:-}" ] && [ -r "$HOME/.hf_token" ]; then
    HF_TOKEN="$(tr -d '\r\n' < "$HOME/.hf_token")"
    export HF_TOKEN
fi

# Hugging Face caches into the workspace too.
export HF_HOME="${HF_HOME:-$JV_WS/hf}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
mkdir -p "$HF_HOME"

# Partition defaults. gpu_h100: 4x94 GB, 72 h.
export JV_PARTITION="${JV_PARTITION:-gpu_h100}"
export JV_GPUS="${JV_GPUS:-4}"
export JV_TIME="${JV_TIME:-08:00:00}"

# vLLM's own image, imported once on the login node (compute nodes have no
# registry access).
export JV_IMAGE_TAG="${JV_IMAGE_TAG:-vllm/vllm-openai:latest}"
export JV_SQSH="${JV_SQSH:-$JV_IMAGES/vllm.sqsh}"

# A nightly image for architectures newer than the stable release (GLM-5.3-Flash).
export JV_IMAGE_TAG_NIGHTLY="${JV_IMAGE_TAG_NIGHTLY:-vllm/vllm-openai:cu129-nightly}"
export JV_SQSH_NIGHTLY="${JV_SQSH_NIGHTLY:-$JV_IMAGES/vllm-nightly.sqsh}"

jv_image_for() {   # jv_image_for <needs_image>  ->  path to the .sqsh
    case "${1:-stable}" in
        nightly) echo "$JV_SQSH_NIGHTLY" ;;
        *)       echo "$JV_SQSH" ;;
    esac
}

# Port for the vLLM server inside the job. Node-local, never exposed.
export JV_PORT="${JV_PORT:-8000}"

jv_model_dir() {   # jv_model_dir <hf_id>  ->  the staged path
    local hf_id="$1"
    echo "$JV_MODELS/${hf_id//\//__}"
}

jv_require() {
    for tool in "$@"; do
        command -v "$tool" >/dev/null 2>&1 || {
            echo "ERROR: $tool not found on PATH." >&2
            return 1
        }
    done
}
