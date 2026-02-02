#!/bin/bash
#SBATCH -p V100q                # 指定分区名称，例如 NA100q [cite: 115]
#SBATCH --gres=gpu:1            # 申请 1 个 GPU [cite: 116]
#SBATCH -n 1                    # 运行 1 个任务 [cite: 117]
#SBATCH -c 4                    # 为每个任务分配 4 个核心 [cite: 19, 93]
#SBATCH -o job-%j.output        # 标准输出文件 [cite: 114]

# Run temperature experiments with dual-generation logic
# Target Agent 0 receives variants, others receive defaults.

# Temperatures to test
TEMPS=(0.5 1.0 1.5 2.0)
# TEMPS=(2.5 1.7 1.3 0.7 0.3)
DATA_SIZE=50 # Adjusted for reasonable runtime during testing
ROUNDS=3
DATASET="gsm8k"


# Enable Offline Mode for HuggingFace Datasets to prevent locking/hanging on compute nodes
export HF_DATASETS_OFFLINE=1

for T in "${TEMPS[@]}"; do
    echo "Running experiment with Variant Temperature: $T"
    
    # Run command
    # Using gsm8k as the dataset as per previous context
    # Adjust model path/name if needed, defaulting to qwen2.5-7b as seen in other scripts
    /export/home3/dazhou/miniconda3/envs/venv/bin/python -u src/main.py \
        --model qwen2.5-7b \
        --num_agents 3 \
        --data $DATASET \
        --data_size $DATA_SIZE \
        --debate_rounds $ROUNDS \
        --variant_temperature $T \
        --target_agent_index 0 \
        --data_dir /export/home3/dazhou/debate-or-vote/datasets \
        --model_dir /export/home3/dazhou/debate-or-vote/models \
        --out_dir "out/temp_exp_T=${T}"

    echo "Completed Temp=$T"
done
