#!/usr/bin/env bash
# Queue N copies of a resumable job, each starting when the previous one ends.
# Login node.
#
#   bash cluster/07_chain.sh <n> <sbatch args...> -- <job script> [args...]
#
# Each job resumes from its store, and one that finds the work done exits before
# loading the model. afterany: a link that times out still starts the next.

set -euo pipefail
N="${1:-}"
[[ "$N" =~ ^[0-9]+$ ]] && [ "$N" -ge 1 ] || {
    echo "usage: $0 <n> <sbatch args...> -- <job script> [args...]" >&2; exit 2; }
shift
ARGS=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do ARGS+=("$1"); shift; done
[ "${1:-}" = "--" ] || { echo "missing -- before the job script" >&2; exit 2; }
shift

PREV=""
for i in $(seq 1 "$N"); do
    DEP=()
    [ -n "$PREV" ] && DEP=(--dependency=afterany:"$PREV")
    PREV="$(sbatch --parsable ${ARGS[@]+"${ARGS[@]}"} ${DEP[@]+"${DEP[@]}"} "$@")"
    PREV="${PREV%%;*}"
    echo "link $i/$N: job $PREV"
done
