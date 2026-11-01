# Local driver. Everything that touches the cluster goes over ssh.
#
#   make test                    run the test suites locally, no GPU
#   make corpus                  rebuild data/corpus.json + reference scores
#   make sync                    push the code to the cluster
#   make setup weights           one-time: container + model weights
#   make submit                  queue both judges
#   make queue                   what is running
#   make fetch report            pull results and analyse
#
#   Experiment 3 (instruction language at generation), see the block at the end:
#   make gen-weights gen-smoke gen-submit gen-status gen-export gen-judge gen-fetch
#   Experiment 4b (feedback, categorised):
#   make feedback JUDGE=<name> codebook-items codebook-classify codebook-report

CLUSTER   ?= bwunicluster
REMOTE    ?= judge-validation
JUDGES    ?= deepseek-v4-flash llama4-scout
PARTITION  ?=
GPUS      ?= 4
TIME       ?=
# The first Python found that can import yaml.
PY ?= $(shell for p in python3 $$HOME/miniconda3/bin/python3 \
        /usr/local/bin/python3 /opt/homebrew/bin/python3 /usr/bin/python3; do \
        command -v $$p >/dev/null 2>&1 && \
        $$p -c 'import yaml' >/dev/null 2>&1 && { echo $$p; break; }; done)

.PHONY: help check-py test corpus sync setup weights watch doctor smoke \
        preflight submit submit-one short queue fetch status report relative figures prompts clean \
        gen-weights gen-smoke gen-submit gen-status gen-export gen-judge gen-fetch gen-report checks \
        feedback codebook-items codebook-classify codebook-fetch codebook-report

check-py:
	@test -n "$(PY)" || { \
	  echo "ERROR: no python3 with PyYAML found."; \
	  echo "  fix: python3 -m pip install pyyaml"; exit 1; }

help:
	@sed -n '1,10p' Makefile

test:
	@$(PY) tests/test_analyze.py
	@$(PY) tests/test_corpus.py
	@$(PY) tests/test_endtoend.py
	@$(PY) tests/test_feedback.py
	@$(PY) tests/test_rating_workbook.py
	@$(PY) tests/test_exp3.py
	@$(PY) tests/test_relative.py

corpus:
	@$(PY) -m jv corpus

# --delete, but keep what the cluster wrote here (Slurm logs, stage.log).
sync:
	@rsync -avz --delete \
	  --exclude '.git' --exclude '__pycache__' --exclude 'out/' \
	  --exclude '*.db' --exclude '*.pyc' \
	  --exclude '*.out' --exclude '*.err' --exclude '*.log' \
	  ./ $(CLUSTER):$(REMOTE)/
	@echo "synced to $(CLUSTER):$(REMOTE)"

# Long-running cluster work runs in tmux, so it survives a dropped connection.
setup: sync
	@ssh $(CLUSTER) 'cd $(REMOTE) && tmux kill-session -t jv-setup 2>/dev/null; \
	  tmux new -d -s jv-setup "bash cluster/00_setup.sh 2>&1 | tee setup.log"'
	@echo "started in tmux session jv-setup. follow it with:  make watch W=setup"

# All judges in one detached session, one after another.
weights: sync
	@ssh $(CLUSTER) 'cd $(REMOTE) && tmux kill-session -t jv-stage 2>/dev/null; \
	  tmux new -d -s jv-stage "for j in $(JUDGES); do \
	    bash cluster/01_stage_weights.sh \$$j; done 2>&1 | tee stage.log"'
	@echo "started in tmux session jv-stage. follow it with:  make watch W=stage"

# make watch W=setup   |   make watch W=stage
watch:
	@ssh $(CLUSTER) 'cd $(REMOTE) && tail -25 $(W).log 2>/dev/null || \
	  echo "no $(W).log yet"' 
	@ssh $(CLUSTER) 'tmux ls 2>/dev/null || echo "(no tmux sessions running)"'

doctor: sync
	@ssh $(CLUSTER) 'cd $(REMOTE) && bash cluster/doctor.sh'

# 20 minutes in the dev queue: checks the container, the GPUs and the weights.
smoke: sync
	@ssh $(CLUSTER) 'cd $(REMOTE) && sbatch -p dev_gpu_h100 --gres=gpu:4 \
	  cluster/03_smoke.sbatch'
	@echo "watch with: make queue   then read jv-smoke-<id>.out on the cluster"

# Partition, memory and walltime per judge come from config/experiment.yaml.
submit: check-py sync
	@for j in $(JUDGES); do \
	  ARGS=$$(JV_TIME=$(TIME) $(PY) cluster/sbatch_args.py $$j $(PARTITION)) || exit 1; \
	  echo "--- $$j: $$ARGS"; \
	  ssh $(CLUSTER) "cd $(REMOTE) && sbatch $$ARGS cluster/02_judge.sbatch $$j"; \
	done

# One judge, optionally forced onto a partition:
#   make submit-one JUDGE=glm-4.7-flash
#   make submit-one JUDGE=deepseek-v4-flash PARTITION=gpu_h100_short
submit-one: check-py sync
	@test -n "$(JUDGE)" || { echo "usage: make submit-one JUDGE=<name>"; exit 2; }
	@ARGS=$$(JV_TIME=$(TIME) $(PY) cluster/sbatch_args.py $(JUDGE) $(PARTITION)) || exit 1; \
	 echo "--- $(JUDGE): $$ARGS"; \
	 ssh $(CLUSTER) "cd $(REMOTE) && sbatch $$ARGS cluster/02_judge.sbatch $(JUDGE)"

# Check the vLLM command line on the login node before queueing.
preflight: sync
	@test -n "$(JUDGE)" || { echo "usage: make preflight JUDGE=<name>"; exit 2; }
	@ssh $(CLUSTER) 'cd $(REMOTE) && bash cluster/04_preflight.sh $(JUDGE)'

# The 30-minute queue. A job that runs out of time resumes where it stopped.
short: check-py preflight
	@test -n "$(JUDGE)" || { echo "usage: make short JUDGE=<name>"; exit 2; }
	@ARGS=$$(JV_TIME=00:30:00 $(PY) cluster/sbatch_args.py $(JUDGE) gpu_h100_short) \
	  || exit 1; \
	 echo "--- $(JUDGE): $$ARGS"; \
	 ssh $(CLUSTER) "cd $(REMOTE) && sbatch $$ARGS cluster/02_judge.sbatch $(JUDGE)"

queue:
	@ssh $(CLUSTER) 'squeue --me --format="%.10i %.12P %.14j %.8T %.10M %.6D %R"'

fetch:
	@mkdir -p out
	@WS=$$(ssh $(CLUSTER) 'ws_find judge-validation 2>/dev/null || echo $$HOME/judge-validation-ws'); \
	  echo "pulling from $(CLUSTER):$$WS/out"; \
	  rsync -avz $(CLUSTER):$$WS/out/*.db out/; \
	  rsync -avz $(CLUSTER):$$WS/out/feedback/ out/feedback/ 2>/dev/null || true
	@$(PY) -m jv merge --db out/judgements.db \
	  $$(ls out/*.db | grep -v judgements.db)
	@$(PY) -m jv status --db out/judgements.db

# Read the merged store without pulling from the cluster again.
status:
	@$(PY) -m jv status --db out/judgements.db

# Experiment 4a: detailed ratings, drawn from the merged store.
figures:
	@$(PY) -m jv figures

# Which script is better, with every rater's level and spread removed:
# pairwise preference agreement, humans and judges, plus the figure.
relative: check-py
	@$(PY) -m jv relative

report:
	@$(PY) -m jv report --db out/judgements.db --out out/report.txt \
	  --json out/report.json --also-print

prompts:
	@mkdir -p out
	@$(PY) -m jv prompt S01 --lang en > out/prompt_en.txt
	@$(PY) -m jv prompt S01 --lang ja > out/prompt_ja.txt
	@$(PY) -m jv rubric --lang en > out/rubric_en.txt
	@$(PY) -m jv rubric --lang ja > out/rubric_ja.txt
	@echo "wrote out/prompt_{en,ja}.txt and out/rubric_{en,ja}.txt"

clean:
	@rm -rf out/*.db out/report.* __pycache__ jv/__pycache__ tests/__pycache__

# --------------------------------------------------------------------------
# Experiment 3: prompt language at generation.
#
#   make gen-weights             stage Qwen3.8-27B, ~56 GiB, in tmux
#   make gen-smoke               3 customers x EN/JA on the 30-minute queue
#   make gen-submit [CHAIN=n]    all 100 customers (resumes where it stopped)
#   make gen-status              what the cluster store holds
#   make gen-export              blinded corpus + facts for the judges
#   make gen-judge JUDGE=<name> [CHAIN=n]  judge the generated scripts
#   make gen-fetch               pull everything back, deterministic metrics

GEN ?= qwen3.8-27b

gen-weights: sync
	@ssh $(CLUSTER) 'cd $(REMOTE) && tmux kill-session -t jv-stage-gen 2>/dev/null; \
	  tmux new -d -s jv-stage-gen "bash cluster/01_stage_weights.sh $(GEN) 2>&1 | tee stage-gen.log"'
	@echo "started in tmux session jv-stage-gen. follow it with:  make watch W=stage-gen"

gen-smoke: check-py sync
	@ARGS=$$(JV_TIME=00:30:00 $(PY) cluster/sbatch_args.py $(GEN) gpu_h100_short) || exit 1; \
	 echo "--- $(GEN) smoke: $$ARGS"; \
	 ssh $(CLUSTER) "cd $(REMOTE) && sbatch $$ARGS cluster/05_generate.sbatch $(GEN) 3"

# CHAIN=n: n back-to-back 30-minute jobs on the short queue instead of one
# long job, which can wait days. Each link resumes; spare links exit at once.
#   make gen-submit CHAIN=8
CHAIN ?=
# Requests in flight per judge job (gen-judge). Empty = 16.
PARALLEL ?=
SHORT_PART = $(or $(PARTITION),gpu_h100_short)

gen-submit: check-py sync
	@if [ -n "$(CHAIN)" ]; then \
	   ARGS=$$(JV_TIME=00:30:00 $(PY) cluster/sbatch_args.py $(GEN) $(SHORT_PART)) || exit 1; \
	   echo "--- $(GEN), chain of $(CHAIN): $$ARGS"; \
	   ssh $(CLUSTER) "cd $(REMOTE) && bash cluster/07_chain.sh $(CHAIN) $$ARGS -- cluster/05_generate.sbatch $(GEN)"; \
	 else \
	   ARGS=$$(JV_TIME=$(TIME) $(PY) cluster/sbatch_args.py $(GEN) $(PARTITION)) || exit 1; \
	   echo "--- $(GEN): $$ARGS"; \
	   ssh $(CLUSTER) "cd $(REMOTE) && sbatch $$ARGS cluster/05_generate.sbatch $(GEN)"; \
	 fi

gen-status:
	@ssh $(CLUSTER) 'cd $(REMOTE) && source cluster/env.sh >/dev/null && \
	  python3 -m gen status --db $$JV_OUT/exp3/generation.db'

gen-export: sync
	@ssh $(CLUSTER) 'cd $(REMOTE) && source cluster/env.sh >/dev/null && \
	  python3 -m gen export --db $$JV_OUT/exp3/generation.db --out $$JV_OUT/exp3'

gen-judge: check-py sync
	@test -n "$(JUDGE)" || { echo "usage: make gen-judge JUDGE=<name>"; exit 2; }
	@if [ -n "$(CHAIN)" ]; then \
	   ARGS=$$(JV_TIME=00:30:00 $(PY) cluster/sbatch_args.py $(JUDGE) $(SHORT_PART)) || exit 1; \
	 else \
	   ARGS=$$(JV_TIME=$${TIME:-04:00:00} $(PY) cluster/sbatch_args.py $(JUDGE) $(PARTITION)) || exit 1; \
	 fi; \
	 echo "--- $(JUDGE) on exp3$(if $(CHAIN), chain of $(CHAIN)): $$ARGS"; \
	 ssh $(CLUSTER) "cd $(REMOTE) && JV_CHAIN=$(CHAIN) JV_PARALLELISM=$(PARALLEL) bash cluster/06_judge_extra.sh exp3 $(JUDGE) $$ARGS"

# Exp 3 results: paired EN vs JA per judge, rubric language and aspect.
gen-report: check-py
	@$(PY) -m jv exp3-report

# Exp 3 robustness checks (needs fugashi for the legacy keigo measure).
checks: check-py
	@$(PY) -m jv checks

gen-fetch: check-py
	@mkdir -p out/exp3
	@WS=$$(ssh $(CLUSTER) 'ws_find judge-validation 2>/dev/null || echo $$HOME/judge-validation-ws'); \
	  rsync -avz --include='*.db' --include='*.json' --include='*.csv' \
	  --include='feedback/' --include='feedback/*.db' --exclude='*' \
	  $(CLUSTER):$$WS/out/exp3/ out/exp3/
	@$(PY) -m gen status --db out/exp3/generation.db
	@$(PY) -m gen metrics --db out/exp3/generation.db --out out/exp3

# --------------------------------------------------------------------------
# Experiment 4b: what the judges say, unscored, then categorised.
#
#   make feedback JUDGE=<name>   the 12 legacy scripts: refill failed cells,
#                                then the feedback pass (one job, one load)
#   make codebook-items          collect every point into data/codebook/, and
#                                write the hand-coding sheet once
#   make codebook-classify       Qwen3.8 files every point into the codebook
#   make codebook-fetch          pull the labels back
#   make codebook-report         profiles per judge, chi-square, kappa

feedback: check-py sync
	@test -n "$(JUDGE)" || { echo "usage: make feedback JUDGE=<name>"; exit 2; }
	@ARGS=$$(JV_TIME=$(TIME) $(PY) cluster/sbatch_args.py $(JUDGE) $(PARTITION)) || exit 1; \
	 echo "--- $(JUDGE) legacy refill + feedback: $$ARGS"; \
	 ssh $(CLUSTER) "cd $(REMOTE) && bash cluster/06_judge_extra.sh legacy $(JUDGE) $$ARGS"

codebook-items: check-py
	@$(PY) -m jv codebook-items

codebook-classify: check-py sync
	@ARGS=$$(JV_TIME=$${TIME:-02:00:00} $(PY) cluster/sbatch_args.py $(GEN) $(PARTITION)) || exit 1; \
	 echo "--- classify with $(GEN): $$ARGS"; \
	 ssh $(CLUSTER) "cd $(REMOTE) && sbatch $$ARGS cluster/05_generate.sbatch $(GEN) classify"

codebook-fetch:
	@mkdir -p out/codebook
	@WS=$$(ssh $(CLUSTER) 'ws_find judge-validation 2>/dev/null || echo $$HOME/judge-validation-ws'); \
	  rsync -avz $(CLUSTER):$$WS/out/codebook/ out/codebook/

codebook-report: check-py
	@$(PY) -m jv codebook-report
