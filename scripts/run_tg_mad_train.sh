#!/bin/bash

#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH --gres=gpu:0
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -o job-tgmad-train-%j.output
#SBATCH -J tgmad-train

set -euo pipefail

echo "=== TG-MAD Training ==="
echo "Node: $(hostname)"
date

export OPENAI_API_KEY=EMPTY
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# Verify servers are running
echo "Checking vLLM servers..."
curl -s http://localhost:8000/health > /dev/null || { echo "ERROR: Debater server not running on port 8000. On SLURM, prefer scripts/run_tg_mad_train_2gpu.sh."; exit 1; }
curl -s http://localhost:8001/health > /dev/null || { echo "ERROR: Evaluator server not running on port 8001. On SLURM, prefer scripts/run_tg_mad_train_2gpu.sh."; exit 1; }
echo "Both servers are responsive."

python -u tg_mad/train.py \
    --debater_base_url http://localhost:8000/v1 \
    --evaluator_base_url http://localhost:8001/v1 \
    --batch_size 5 \
    --num_epochs 2 \
    --seed 42

echo "Training complete."
date
