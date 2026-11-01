#!/usr/bin/env bash
# Run once on a login node: pulls the image, which jobs then read from the workspace.

source "$(dirname "$0")/env.sh"
set -euo pipefail

# bash cluster/00_setup.sh            -> the stable image
# bash cluster/00_setup.sh nightly    -> the nightly, for GLM-5.3-Flash
if [ "${1:-}" = "nightly" ]; then
    JV_IMAGE_TAG="$JV_IMAGE_TAG_NIGHTLY"
    JV_SQSH="$JV_SQSH_NIGHTLY"
    echo "building the NIGHTLY image (needed for Glm5Next)"
    echo
fi

echo "workspace : $JV_WS"
echo "models    : $JV_MODELS"
echo "images    : $JV_IMAGES"
echo

# Compute nodes mount only xz, zlib, lz4 and zstd images; lz4 builds fastest.
export ENROOT_SQUASH_OPTIONS="${ENROOT_SQUASH_OPTIONS:--comp lz4}"

# Build on local disk, then move: writing a squashfs straight to Lustre is slow.
BUILD_DIR="$JV_IMAGES"
NEED_GIB=25
if [ -n "${TMPDIR:-}" ] && [ -d "${TMPDIR:-}" ]; then
    FREE=$(df -BG --output=avail "$TMPDIR" 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ -n "$FREE" ] && [ "$FREE" -ge "$NEED_GIB" ]; then
        BUILD_DIR="$TMPDIR"
        echo "building on local disk: $TMPDIR (${FREE} GiB free)"
    else
        echo "TMPDIR has only ${FREE:-0} GiB free, need ${NEED_GIB}; building in place"
    fi
fi
export ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-$BUILD_DIR}"
echo "squashfs options: $ENROOT_SQUASH_OPTIONS"
echo

# An image with the wrong compression fails only once a job holds GPUs: rebuild it.
if [ -s "$JV_SQSH" ] && command -v unsquashfs >/dev/null 2>&1; then
    COMP=$(unsquashfs -s "$JV_SQSH" 2>/dev/null \
           | grep -i "compression" | head -1 | awk '{print $NF}')
    case "$COMP" in
        lzo|lzma)
            echo "existing image uses '$COMP', which compute nodes cannot mount."
            echo "removing and rebuilding with zstd."
            rm -f "$JV_SQSH"
            ;;
        "") : ;;
        *) echo "existing image compression: $COMP (mountable)" ;;
    esac
fi

if [ -s "$JV_SQSH" ]; then
    echo "container already imported: $JV_SQSH"
else
    if command -v enroot >/dev/null 2>&1; then
        echo "importing $JV_IMAGE_TAG with enroot (this takes a while)"
        BUILD_SQSH="$BUILD_DIR/vllm.sqsh.$$"
        enroot import -o "$BUILD_SQSH" "docker://$JV_IMAGE_TAG"
        if [ "$BUILD_SQSH" != "$JV_SQSH" ]; then
            echo "moving image to the workspace"
            mv -f "$BUILD_SQSH" "$JV_SQSH"
        fi
    elif command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1; then
        RUNNER=$(command -v apptainer || command -v singularity)
        JV_SIF="${JV_SQSH%.sqsh}.sif"
        echo "enroot not found, building $JV_SIF with $(basename "$RUNNER")"
        "$RUNNER" build "$JV_SIF" "docker://$JV_IMAGE_TAG"
        echo "set JV_SIF=$JV_SIF when submitting"
    else
        echo "ERROR: neither enroot nor apptainer is available." >&2
        echo "On bwUniCluster: module load devel/enroot   (or devel/apptainer)" >&2
        exit 1
    fi
fi

# The client runs outside the container and needs PyYAML (installed with --user).
if ! python3 -c "import yaml" 2>/dev/null; then
    echo "installing PyYAML for the client"
    pip install --user -q pyyaml
fi
python3 -c "import yaml; print('PyYAML ok')"

# A nightly tag moves: record the exact build.
if command -v enroot >/dev/null 2>&1; then
    STAMP="${JV_SQSH%.sqsh}.imageinfo"
    {
        echo "tag       $JV_IMAGE_TAG"
        echo "imported  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "sqsh      $JV_SQSH"
        echo "bytes     $(stat -c %s "$JV_SQSH" 2>/dev/null || echo '?')"
    } > "$STAMP"
    echo
    echo "recorded provenance in $STAMP:"
    sed 's/^/  /' "$STAMP"
fi

echo
echo "next:"
echo "  bash cluster/01_stage_weights.sh deepseek-v4-flash   # stable image"
echo "  bash cluster/01_stage_weights.sh glm-5.3-flash       # nightly image"
echo
echo "images present:"
for img in "$JV_SQSH" "$JV_SQSH_NIGHTLY"; do
    if [ -s "$img" ]; then
        printf "  %-14s %s\n" "$(basename "$img")" \
            "$(du -h "$img" 2>/dev/null | cut -f1)"
    else
        printf "  %-14s MISSING\n" "$(basename "$img")"
    fi
done
