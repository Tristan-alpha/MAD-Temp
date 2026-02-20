#!/bin/bash

#SBATCH -p PA100q                # 指定分区名称，例如 NA100q [cite: 115]          
#SBATCH --gres=gpu:1            # 申请 1 个 GPU [cite: 116]
#SBATCH -n 1                    # 运行 1 个任务 [cite: 117]
#SBATCH -c 4                    # 为每个任务分配 4 个核心 [cite: 19, 93]
#SBATCH -o job-q3-8192-%j.output        # 标准输出文件 [cite: 114]

# Configuration (based on recent logs, adjust as needed)
MODEL="qwen3-4b"
AGENTS=3
ROUNDS=3
DATA="gsm8k"
SIZE=500
AGENT_IDX=0 # First agent (0-indexed)

# Temperatures to traverse
TEMPS=(0 0.5 1.0 1.5 2.0 5.0 10.0 20.0)



echo "Starting Experiment 1: Modifying Round 2"
for t in "${TEMPS[@]}"; do
    echo "Running Round 2, Agent $AGENT_IDX, Temp $t"
    python -u src/main.py --model $MODEL --num_agents $AGENTS --data $DATA --data_size $SIZE --debate_rounds $ROUNDS \
        --target_round 2 --target_agent_idx $AGENT_IDX --target_temp $t \
        --model_dir ./models \
        --data_dir ./datasets \
        --max_new_tokens 8192
done

echo "All experiments completed."
