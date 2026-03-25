#!/bin/bash

# Canonical TG-MAD evaluation entrypoint.
# Configure experiments by exporting env vars before `sbatch` rather than
# using multiple wrapper presets.

#SBATCH -p PA100q
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -o job-tgmad-eval-%j.output
#SBATCH --time=24:00:00
#SBATCH -J tgmad-eval

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/export/home3/dazhou/debate-or-vote}"
cd "${REPO_ROOT}"

# Avoid inheriting train-time GPU free-memory thresholds (for example,
# DEBATER_MIN_FREE_MIB=12000) that can make eval auto-pick fail on shared nodes.
# Set TG_MAD_PRESERVE_DEBATER_MIN_FREE_MIB=1 to keep the inherited value.
if [[ "${TG_MAD_PRESERVE_DEBATER_MIN_FREE_MIB:-0}" != "1" ]]; then
	unset DEBATER_MIN_FREE_MIB
fi

# Default TG-MAD evaluations to saving debate text history. Dataset/model/history
# selection should be provided via env vars such as DATASET, EVAL_EXISTING_DATA,
# PROMPT_HISTORY_PATH, and DEBATER_MODEL_NAME. This remains overridable by the caller.
export DATASET="${DATASET:-hh_rlhf}"
export SAVE_TEXT_HISTORY="${SAVE_TEXT_HISTORY:-1}"

exec python -m tg_mad.job_runner eval "$@"
