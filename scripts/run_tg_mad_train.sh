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

# Default TG-MAD experiments to per-agent prompt optimization with debate text
# history saved. These remain overridable by the caller.
export PER_AGENT_PROMPTS="${PER_AGENT_PROMPTS:-1}"
export SAVE_TEXT_HISTORY="${SAVE_TEXT_HISTORY:-1}"

exec python -m tg_mad.job_runner train "$@"
