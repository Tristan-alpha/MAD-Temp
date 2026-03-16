#!/bin/bash

#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o job-tgmad-train-2gpu-text-history-%j.output
#SBATCH --time=24:00:00
#SBATCH -J tgmad-train2h

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/export/home3/dazhou/debate-or-vote}"
cd "${REPO_ROOT}"

export SAVE_TEXT_HISTORY=1

bash "${REPO_ROOT}/scripts/run_tg_mad_train_2gpu.sh"
