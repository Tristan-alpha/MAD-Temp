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

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_PROFILE="${EXPERIMENT_PROFILE:-}"

apply_experiment_profile_defaults() {
    if [[ -z "${EXPERIMENT_PROFILE}" ]]; then
        return
    fi
    while IFS='=' read -r key value; do
        [[ -z "${key}" ]] && continue
        if [[ -z "${!key:-}" ]]; then
            export "${key}=${value}"
        fi
    done < <("${PYTHON_BIN}" - "${EXPERIMENT_PROFILE}" <<'PY'
from tg_mad.experiment_profiles import get_stage_env_defaults
import sys

for key, value in get_stage_env_defaults(sys.argv[1], "eval").items():
    print(f"{key}={value}")
PY
)
}

apply_experiment_profile_defaults

# Avoid inheriting train-time GPU free-memory thresholds (for example,
# DEBATER_MIN_FREE_MIB=12000) that can make eval auto-pick fail on shared nodes.
# Set TG_MAD_PRESERVE_DEBATER_MIN_FREE_MIB=1 to keep the inherited value.
if [[ "${TG_MAD_PRESERVE_DEBATER_MIN_FREE_MIB:-0}" != "1" ]]; then
	unset DEBATER_MIN_FREE_MIB
fi

# TG-MAD evaluations now save debate text history by default. Dataset/model/history
# selection should be provided via env vars such as DATASET, EVAL_EXISTING_DATA,
# PROMPT_HISTORY_PATH, and DEBATER_MODEL_NAME. This remains overridable by the caller.
export DATASET="${DATASET:-hh_rlhf}"

exec python -m tg_mad.job_runner eval "$@"
