#!/usr/bin/env bash
# Queue one judge for the follow-up experiments. Login node.
#
#   bash cluster/06_judge_extra.sh legacy <judge> <sbatch args...>
#       the 12 legacy scripts: refill any failed scored cells, then the
#       unscored feedback pass (Exp 4b) into out/feedback/<judge>.db
#
#   bash cluster/06_judge_extra.sh exp3 <judge> <sbatch args...>
#       the Exp 3 generated corpus with customer facts, 3 repeats, into
#       out/exp3/<judge>.db, then feedback into out/exp3/feedback/<judge>.db
#
# Both reuse 02_judge.sbatch through JV_DB and JV_RUN_ARGS.

set -euo pipefail
MODE="${1:-}"
JUDGE="${2:-}"
[ -n "$MODE" ] && [ -n "$JUDGE" ] || {
    echo "usage: $0 <legacy|exp3> <judge> <sbatch args...>" >&2; exit 2; }
shift 2

source cluster/env.sh
set -euo pipefail

case "$MODE" in
    legacy)
        mkdir -p "$JV_OUT/feedback"
        # Refilled cells go to the original store, which the report reads.
        export JV_DB="$JV_OUT/${JUDGE}.db"
        export JV_RUN_ARGS="--feedback-db $JV_OUT/feedback/${JUDGE}.db --parallelism 8"
        echo "judge $JUDGE: legacy 12 scripts, refill + feedback"
        ;;
    exp3)
        CORPUS="$JV_OUT/exp3/corpus.json"
        FACTS="$JV_OUT/exp3/facts.json"
        [ -s "$CORPUS" ] && [ -s "$FACTS" ] || {
            echo "ERROR: no exported corpus at $JV_OUT/exp3. Run: make gen-export" >&2
            exit 1
        }
        read -r REPEATS LANGS MAXTOK <<EOF
$(python3 -c 'import yaml; j = yaml.safe_load(open("config/experiment.yaml"))["exp3"]["judging"]; print(j["repeats"], ",".join(j["rubric_langs"]), j.get("max_tokens", 0))')
EOF
        mkdir -p "$JV_OUT/exp3/feedback"
        export JV_DB="$JV_OUT/exp3/${JUDGE}.db"
        export JV_RUN_ARGS="--corpus $CORPUS --facts $FACTS --repeats $REPEATS --rubric-langs $LANGS --max-tokens $MAXTOK --feedback-db $JV_OUT/exp3/feedback/${JUDGE}.db --parallelism ${JV_PARALLELISM:-16}"
        echo "judge $JUDGE: exp3 corpus, $LANGS x $REPEATS, then feedback"
        ;;
    *)
        echo "unknown mode $MODE, expected legacy or exp3" >&2; exit 2 ;;
esac

echo "store $JV_DB"
echo "args  $JV_RUN_ARGS"
# JV_CHAIN=n queues n back-to-back short jobs instead of one long one.
if [ -n "${JV_CHAIN:-}" ]; then
    bash cluster/07_chain.sh "$JV_CHAIN" "$@" -- cluster/02_judge.sbatch "$JUDGE"
else
    sbatch "$@" cluster/02_judge.sbatch "$JUDGE"
fi
