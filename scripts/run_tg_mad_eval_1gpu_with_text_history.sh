#!/bin/bash

#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -o job-tgmad-eval-1gpu-text-history-%j.output
#SBATCH --time=24:00:00
#SBATCH -J tgmad-eval1h

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/export/home3/dazhou/debate-or-vote}"
cd "${REPO_ROOT}"

export SAVE_TEXT_HISTORY=1

bash "${REPO_ROOT}/scripts/run_tg_mad_eval_1gpu.sh"
