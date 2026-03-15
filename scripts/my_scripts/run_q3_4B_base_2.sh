#!/bin/bash

#SBATCH -p NH100q                # 指定分区名称，例如 NA100q [cite: 115]            
#SBATCH --gres=gpu:1            # 申请 1 个 GPU [cite: 116]
#SBATCH -n 1                    # 运行 1 个任务 [cite: 117]
#SBATCH -c 4                    # 为每个任务分配 4 个核心 [cite: 19, 93]
#SBATCH -o job-q3-%j.output        # 标准输出文件 [cite: 114]

# Configuration (based on recent logs, adjust as needed)
MODEL="qwen3-4b-base"
AGENTS=3
ROUNDS=3
DATA="gsm8k"
SIZE=50
AGENT_IDX=0 # First agent (0-indexed)

# Single baseline run (T=1.0 fixed in src/main.py)
TEMPS=(1.0)



echo "Starting Experiment 1: Modifying Round 2"
for t in "${TEMPS[@]}"; do
    echo "Running Round 2, Agent $AGENT_IDX"
    python -u src/main.py --model $MODEL --num_agents $AGENTS --data $DATA --data_size $SIZE --debate_rounds $ROUNDS \
        --model_dir ./models \
        --data_dir ./datasets
done

echo "All experiments completed."
