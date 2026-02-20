#!/bin/bash

#SBATCH -p V100q                # 指定分区名称，例如 NA100q [cite: 115]
#SBATCH --gres=gpu:1            # 申请 1 个 GPU [cite: 116]
#SBATCH -n 1                    # 运行 1 个任务 [cite: 117]
#SBATCH -c 4                   # 为每个任务分配 4 个核心 [cite: 19, 93]
#SBATCH -o test.output        # 标准输出文件 [cite: 114]

# 加载 CUDA 模块（Flash Attention 2 需要）
module load cuda12.8/toolkit/12.8.1

python test.py
