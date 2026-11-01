#!/usr/bin/env bash
# Pull one model's weights into the workspace. Login node only.
#
#   bash cluster/01_stage_weights.sh deepseek-v4-flash
#
# Gated repositories need a token:
#   export HF_TOKEN=hf_...

source "$(dirname "$0")/env.sh"
set -euo pipefail

JUDGE="${1:-}"
[ -n "$JUDGE" ] || { echo "usage: $0 <judge-name-from-config>" >&2; exit 2; }

CONFIG="${JV_CODE}/config/experiment.yaml"
[ -f "$CONFIG" ] || { echo "ERROR: no config at $CONFIG" >&2; exit 1; }

read -r HF_ID WEIGHT_GIB GATED <<EOF
$(python3 - "$CONFIG" "$JUDGE" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
for judge in config["judges"] + config.get("generators", []):
    if judge["name"] == sys.argv[2]:
        print(judge["hf_id"], judge.get("approx_weight_gib", 0),
              str(judge.get("gated", False)).lower())
        break
else:
    sys.exit(f"no judge named {sys.argv[2]!r} in the config")
PY
)
EOF

TARGET="$(jv_model_dir "$HF_ID")"
echo "judge   : $JUDGE"
echo "hf_id   : $HF_ID"
echo "target  : $TARGET"
echo "size    : ~${WEIGHT_GIB} GiB"
echo "gated   : $GATED"
echo

if [ "$GATED" = "true" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: $HF_ID is gated and no token was found." >&2
    echo "       1. Request access at https://huggingface.co/$HF_ID" >&2
    echo "          (Llama 4 is MANUALLY approved: this can take days)" >&2
    echo "       2. Create a READ token at huggingface.co/settings/tokens" >&2
    echo "       3. On the cluster, store it privately:" >&2
    echo "            read -rsp 'token: ' T && printf '%s' \"\$T\" > ~/.hf_token" >&2
    echo "            chmod 600 ~/.hf_token" >&2
    exit 1
fi
if [ -n "${HF_TOKEN:-}" ]; then
    # Never print the token itself, only enough to tell which one is loaded.
    echo "token   : ${HF_TOKEN:0:7}... (${#HF_TOKEN} chars)"
fi

AVAIL=$(df -BG --output=avail "$JV_MODELS" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
if [ -n "$AVAIL" ] && [ "$AVAIL" -gt 0 ] && [ "$WEIGHT_GIB" -gt 0 ] \
   && [ "$AVAIL" -lt "$WEIGHT_GIB" ]; then
    echo "ERROR: only ${AVAIL} GiB free but ~${WEIGHT_GIB} GiB needed." >&2
    exit 1
fi

if [ -f "$TARGET/config.json" ]; then
    echo "already staged. Delete $TARGET to re-download."
    du -sh "$TARGET"
    exit 0
fi

# The Python API rather than the CLI, which is not on PATH after `pip install --user`.
python3 -c "import huggingface_hub" 2>/dev/null || {
    echo "installing huggingface_hub"
    pip install --user -q huggingface_hub
}

python3 - "$HF_ID" "$TARGET" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

repo_id, target = sys.argv[1], sys.argv[2]
token = os.environ.get("HF_TOKEN") or None

try:
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        # vLLM reads safetensors; skip the other weight formats.
        ignore_patterns=["*.pth", "*.bin", "original/*", "*.msgpack", "*.h5"],
        max_workers=8,
        token=token,
    )
except Exception as exc:                                       # noqa: BLE001
    message = str(exc)
    if "401" in message or "403" in message or "gated" in message.lower():
        sys.exit(f"access denied for {repo_id}. If it is gated, approval may "
                 f"still be pending, or HF_TOKEN may lack read scope.\n{message[:300]}")
    sys.exit(f"download failed: {message[:500]}")

print(f"\ndownloaded to {path}")
PY

echo
if [ -f "$TARGET/config.json" ]; then
    echo "staged: $TARGET"
    du -sh "$TARGET"
    echo "safetensors: $(ls "$TARGET"/*.safetensors 2>/dev/null | wc -l)"
else
    echo "ERROR: download finished but $TARGET/config.json is missing." >&2
    exit 1
fi
