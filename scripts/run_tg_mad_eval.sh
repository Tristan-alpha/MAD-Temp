#!/bin/bash

#SBATCH -p RTXA6Kq
#SBATCH -w node15
#SBATCH --gres=gpu:0
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -o job-tgmad-eval-%j.output
#SBATCH -J tgmad-eval

set -euo pipefail

echo "=== TG-MAD Evaluation ==="
echo "Node: $(hostname)"
date

export OPENAI_API_KEY=EMPTY
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

# Verify debater server is running (evaluator not needed for eval)
echo "Checking vLLM debater server..."
curl -s http://localhost:8000/health > /dev/null || { echo "ERROR: Debater server not running on port 8000. On SLURM, prefer scripts/run_tg_mad_eval_1gpu.sh."; exit 1; }
echo "Debater server is responsive."

python -u tg_mad/evaluate.py \
    --debater_base_url http://localhost:8000/v1 \
    --prompt_history out/tg_mad/prompt_history.json \
    --output_dir out/tg_mad/

echo "Evaluation complete."
date
