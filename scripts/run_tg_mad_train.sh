#!/bin/bash

# Canonical TG-MAD training entrypoint.
# Configure experiments by exporting env vars before `sbatch` rather than
# using multiple wrapper presets.

#SBATCH -p PA100q
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o job-tgmad-train-%j.output
#SBATCH --time=24:00:00
#SBATCH -J tgmad-train

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

for key, value in get_stage_env_defaults(sys.argv[1], "train").items():
    print(f"{key}={value}")
PY
)
}

apply_experiment_profile_defaults

# TG-MAD now always uses per-agent prompt optimization and saves debate text
# history by default. Dataset/model/history selection should be provided via env
# vars such as DATASET, TRAIN_EXISTING_DATA, EVAL_EXISTING_DATA, and
# DEBATER_MODEL_NAME. These remain overridable by the caller.
export DATASET="${DATASET:-hh_rlhf}"

exec python -m tg_mad.job_runner train "$@"
