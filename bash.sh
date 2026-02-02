#!/bin/sh
#SBATCH -p NA100q                # 指定分区名称，例如 NA100q [cite: 115]
#SBATCH --gres=gpu:1            # 申请 1 个 GPU [cite: 116]
#SBATCH -n 1                    # 运行 1 个任务 [cite: 117]
#SBATCH -c 4                    # 为每个任务分配 4 个核心 [cite: 19, 93]
#SBATCH -o job-%j.output        # 标准输出文件 [cite: 114]

# 加载必要的 CUDA 模块 [cite: 76, 118]
# module load cuda11.1/toolkit

# 执行您的 Python 命令
# 注意：在 Slurm 环境下，通常不需要手动设置 CUDA_VISIBLE_DEVICES，
# 调度器会自动分配可用的 GPU [cite: 63, 116]。
python src/main.py \
    --model qwen2.5-7b \
    --num_agents 3 \
    --data gsm8k \
    --data_size 100 \
    --debate_rounds 3 \
    --data_dir ./datasets \
    --model_dir ./models

# srun -p V100q -n 1 --gres=gpu:1 python -u src/main.py --model qwen2.5-7b --num_agents 3 --data gsm8k --data_size 100 --debate_rounds 3 --data_dir ./datasets --model_dir ./models