"""
可视化 GSM8K 格式错误分析 (v3 - 修正指标)

核心改变：不再用 format_fixable/wrong 作为纵轴（幸存者偏差）。
改用以下指标（基于 ALL 样本，而非仅错误样本）：

  图1 (柱状图): Format Non-Compliance Rate（格式不遵从率）
      = 未成功提取到答案的 agent 回复数 / 总 agent 回复数
      这直接衡量 MAD 是否提高了指令遵循（格式遵从）

  图2 (柱状图): 在所有样本上，agent 级别的格式错误率
      = (提取答案 != 正确答案 且 文本含正确答案) 的 agent 数 / 总 agent 数
      这衡量 "模型知道答案但格式提取失败" 的比例

  Qwen2.5 和 Qwen3 分开画。
"""

import json
import glob
import os
import re
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from analyze_format_errors_v2 import (
    extract_arithmetic_answer, response_contains_correct_number,
    detect_task_type, HISTORY_DIR,
)


def parse_filename(filepath):
    basename = os.path.basename(filepath)
    m = re.match(
        r"([a-zA-Z0-9_]+)_(\d+)__(.+?)_N=(\d+)_R=(\d+)_TR=(\d+)_TT=([\d.]+)(?:_MNT=(\d+))?\.jsonl",
        basename
    )
    if m:
        return {
            "dataset": m.group(1),
            "data_size": int(m.group(2)),
            "model": m.group(3),
            "num_agents": int(m.group(4)),
            "num_rounds": int(m.group(5)),
            "target_round": int(m.group(6)),
            "temperature": float(m.group(7)),
            "max_new_tokens": int(m.group(8)) if m.group(8) else 512,
            "filepath": filepath,
        }
    return None


def collect_gsm8k_files(history_dir=HISTORY_DIR):
    all_files = glob.glob(os.path.join(history_dir, "**", "*.jsonl"), recursive=True)
    files = [f for f in all_files
             if "previous_backup" not in f and "accuracy_summary" not in f]
    files = [f for f in files if os.path.basename(f).startswith("gsm8k")]
    records = []
    for fp in files:
        info = parse_filename(fp)
        if info is not None:
            records.append(info)
    return records


def compute_all_sample_stats(filepath):
    """
    对所有样本（包括正确的）计算每个 round 的 agent 级别统计:
      - agent_total:          总 agent 回复数
      - agent_extraction_empty:  提取结果为空（""）的 agent 数
      - agent_extraction_wrong:  提取到了但答案不对的 agent 数
      - agent_extraction_correct: 提取到了且答案正确的 agent 数
      - agent_text_has_correct:  文本中包含正确答案的 agent 数（不管提取结果如何）
      - agent_format_error:    提取 != 正确答案 且 文本含正确答案
    """
    task_type = detect_task_type(filepath)
    if task_type is None:
        return None

    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    round_stats = {}
    for sample_data in samples:
        for round_key in sample_data.keys():
            rd = sample_data[round_key]
            if round_key not in round_stats:
                round_stats[round_key] = {
                    "total_samples": 0,
                    "agent_total": 0,
                    "agent_extraction_empty": 0,
                    "agent_extraction_wrong": 0,
                    "agent_extraction_correct": 0,
                    "agent_text_has_correct": 0,
                    "agent_format_error": 0,
                }

            rs = round_stats[round_key]
            rs["total_samples"] += 1

            responses = rd["responses"]
            final_answers = rd["final_answers"]
            correct_answer = rd["answer"]
            correct_val = np.round(float(correct_answer), 1)

            for i, agent_name in enumerate(responses.keys()):
                resp = responses[agent_name]
                extracted = final_answers[i]

                rs["agent_total"] += 1

                # 文本中是否包含正确答案
                text_has_correct = response_contains_correct_number(resp, correct_answer)
                if text_has_correct:
                    rs["agent_text_has_correct"] += 1

                # 提取结果分类
                if extracted == "":
                    rs["agent_extraction_empty"] += 1
                    # 提取为空 且 文本含正确答案 → 格式错误
                    if text_has_correct:
                        rs["agent_format_error"] += 1
                elif extracted == correct_val:
                    rs["agent_extraction_correct"] += 1
                else:
                    rs["agent_extraction_wrong"] += 1
                    # 提取到了错误值 且 文本含正确答案 → 格式错误
                    if text_has_correct:
                        rs["agent_format_error"] += 1

    return round_stats


# ──────────────────────────────────────────────────
# 图1: 格式不遵从率 (提取失败率) by Round
# ──────────────────────────────────────────────────

def plot_extraction_failure_rate(records, output_dir):
    """
    纵轴: agent 回复中提取为空的比例 (所有样本, 所有温度聚合)
    横轴: Debate Round
    """
    agg = {}
    for rec in records:
        key = (rec["dataset"], rec["model"])
        round_stats = compute_all_sample_stats(rec["filepath"])
        if round_stats is None:
            continue
        if key not in agg:
            agg[key] = {}
        for rk, rs in round_stats.items():
            rk_int = int(rk)
            if rk_int not in agg[key]:
                agg[key][rk_int] = {"agent_total": 0, "agent_extraction_empty": 0}
            agg[key][rk_int]["agent_total"] += rs["agent_total"]
            agg[key][rk_int]["agent_extraction_empty"] += rs["agent_extraction_empty"]

    qwen25_keys = sorted([k for k in agg if "qwen2.5" in k[1]])
    qwen3_keys = sorted([k for k in agg if "qwen3" in k[1]])

    for group_name, group_keys in [("Qwen2.5", qwen25_keys), ("Qwen3", qwen3_keys)]:
        if not group_keys:
            continue
        _draw_bar_chart(
            agg, group_keys, group_name, output_dir,
            value_key="agent_extraction_empty",
            denom_key="agent_total",
            ylabel="Extraction Failure Rate\n(empty answers / all agent responses)",
            title_suffix="Extraction Failure Rate by Round",
            filename_suffix="extraction_failure_rate",
        )


# ──────────────────────────────────────────────────
# 图2: 格式错误率 (模型知道答案但提取失败) by Round
# ──────────────────────────────────────────────────

def plot_format_error_rate(records, output_dir):
    """
    纵轴: (提取答案 != 正确 且 文本含正确答案) / 总 agent 数
    横轴: Debate Round
    """
    agg = {}
    for rec in records:
        key = (rec["dataset"], rec["model"])
        round_stats = compute_all_sample_stats(rec["filepath"])
        if round_stats is None:
            continue
        if key not in agg:
            agg[key] = {}
        for rk, rs in round_stats.items():
            rk_int = int(rk)
            if rk_int not in agg[key]:
                agg[key][rk_int] = {"agent_total": 0, "agent_format_error": 0}
            agg[key][rk_int]["agent_total"] += rs["agent_total"]
            agg[key][rk_int]["agent_format_error"] += rs["agent_format_error"]

    qwen25_keys = sorted([k for k in agg if "qwen2.5" in k[1]])
    qwen3_keys = sorted([k for k in agg if "qwen3" in k[1]])

    for group_name, group_keys in [("Qwen2.5", qwen25_keys), ("Qwen3", qwen3_keys)]:
        if not group_keys:
            continue
        _draw_bar_chart(
            agg, group_keys, group_name, output_dir,
            value_key="agent_format_error",
            denom_key="agent_total",
            ylabel="Format Error Rate\n(text has correct answer but extraction failed\n / all agent responses)",
            title_suffix="Format Error Rate by Round",
            filename_suffix="format_error_rate",
        )


# ──────────────────────────────────────────────────
# 图3: Debate 准确率 by Round
# ──────────────────────────────────────────────────

def plot_accuracy(records, output_dir):
    """
    纵轴: debate_answer_iscorr 的比例（所有温度聚合）
    横轴: Debate Round
    """
    agg = {}   # key -> {round_int -> {correct, total}}
    for rec in records:
        key = (rec["dataset"], rec["model"])
        filepath = rec["filepath"]

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for round_key, rd in sample.items():
                    try:
                        rk_int = int(round_key)
                    except ValueError:
                        continue
                    if key not in agg:
                        agg[key] = {}
                    if rk_int not in agg[key]:
                        agg[key][rk_int] = {"correct": 0, "total": 0}
                    agg[key][rk_int]["total"] += 1
                    if rd.get("debate_answer_iscorr", False):
                        agg[key][rk_int]["correct"] += 1

    qwen25_keys = sorted([k for k in agg if "qwen2.5" in k[1]])
    qwen3_keys = sorted([k for k in agg if "qwen3" in k[1]])

    for group_name, group_keys in [("Qwen2.5", qwen25_keys), ("Qwen3", qwen3_keys)]:
        if not group_keys:
            continue
        _draw_bar_chart(
            agg, group_keys, group_name, output_dir,
            value_key="correct",
            denom_key="total",
            ylabel="Debate Accuracy\n(correct samples / total samples)",
            title_suffix="Debate Accuracy by Round",
            filename_suffix="accuracy",
        )


# ──────────────────────────────────────────────────
# 通用柱状图绘制
# ──────────────────────────────────────────────────

# ─── Instruct / Base 分类 ───
# Instruct 模型: 带 -Instruct 后缀的 Qwen3 变体
# Base 模型: 纯 base 或 thinking 模式
INSTRUCT_MODELS = {"qwen3-4b", "qwen3-30b-a3b"}
BASE_MODELS = {"qwen3-8b", "qwen3-4b-base", "qwen3-4b-thinking", "qwen3-30b-a3b-base"}

BASE_HATCH = "///"          # 斜线阴影
INSTRUCT_HATCH = ""         # 无阴影
BASE_ALPHA = 0.70
INSTRUCT_ALPHA = 0.90


def _is_base_model(model_name):
    return model_name in BASE_MODELS


def _sort_qwen3_keys(keys):
    """Instruct 模型排前面, Base 模型排后面; 组内按名称排序."""
    instruct = sorted([k for k in keys if not _is_base_model(k[1])])
    base = sorted([k for k in keys if _is_base_model(k[1])])
    return instruct + base


def _draw_bar_chart(agg, group_keys, group_name, output_dir,
                    value_key, denom_key, ylabel, title_suffix, filename_suffix):
    # 对 Qwen3 组按 instruct/base 排序
    if group_name == "Qwen3":
        group_keys = _sort_qwen3_keys(group_keys)

    all_rounds = set()
    for key in group_keys:
        all_rounds.update(agg[key].keys())
    all_rounds = sorted(all_rounds)

    n_groups = len(group_keys)
    n_rounds = len(all_rounds)

    fig, ax = plt.subplots(figsize=(14, 7))
    bar_width = 0.8 / n_groups
    # 用两套配色区分 instruct / base
    instruct_colors = list(plt.cm.Set2.colors)   # 明亮色
    base_colors = list(plt.cm.Pastel2.colors)      # 淡色（搭配阴影更清晰）
    # 为每个模型分配颜色
    instruct_idx = 0
    base_idx = 0
    x_base = np.arange(n_rounds)

    for idx, key in enumerate(group_keys):
        dataset, model = key
        data = agg[key]
        is_base = _is_base_model(model)

        ratios = []
        for r in all_rounds:
            if r in data and data[r][denom_key] > 0:
                ratios.append(data[r][value_key] / data[r][denom_key])
            else:
                ratios.append(0.0)

        if is_base:
            color = base_colors[base_idx % len(base_colors)]
            hatch = BASE_HATCH
            alpha = BASE_ALPHA
            label = f"{model} [Base]"
            base_idx += 1
        else:
            color = instruct_colors[instruct_idx % len(instruct_colors)]
            hatch = INSTRUCT_HATCH
            alpha = INSTRUCT_ALPHA
            # Qwen2.5 组只有 instruct，不需要标签
            if group_name == "Qwen3":
                label = f"{model} [Instruct]"
            else:
                label = f"{dataset} / {model}"
            instruct_idx += 1

        offset = (idx - n_groups / 2 + 0.5) * bar_width
        bars = ax.bar(x_base + offset, ratios, bar_width * 0.9,
                     label=label,
                     color=color, hatch=hatch,
                     edgecolor='gray', linewidth=0.5, alpha=alpha)

        for bar_obj, ratio in zip(bars, ratios):
            if ratio > 0:
                ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                       bar_obj.get_height() + 0.005,
                       f"{ratio:.1%}", ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Debate Round", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"GSM8K {group_name}: {title_suffix}\n"
                 f"(all samples, aggregated across all temperatures)", fontsize=14)
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"Round {r}" for r in all_rounds], fontsize=12)

    # 图例: 自定义顺序 (instruct first, base second), 加分隔
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10, loc='best',
              handlelength=2.5, handleheight=1.5)
    ax.grid(True, axis='y', alpha=0.3)

    # 自动调整 ylim
    all_vals = []
    for key in group_keys:
        for r in all_rounds:
            if r in agg[key] and agg[key][r][denom_key] > 0:
                all_vals.append(agg[key][r][value_key] / agg[key][r][denom_key])
    max_val = max(all_vals) if all_vals else 0.5
    ax.set_ylim(0, min(max_val * 1.3, 1.05))

    plt.tight_layout()
    safe_name = group_name.lower().replace(".", "")
    out_path = os.path.join(output_dir, f"v3_{filename_suffix}_{safe_name}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    output_dir = HISTORY_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("Collecting GSM8K files...")
    records = collect_gsm8k_files()
    print(f"Found {len(records)} experiment files.")

    combos = sorted(set((r["dataset"], r["model"]) for r in records))
    print(f"\nDataset/Model combinations:")
    for ds, mdl in combos:
        count = sum(1 for r in records if r["dataset"] == ds and r["model"] == mdl)
        print(f"  {ds} / {mdl}: {count} files")

    print("\n--- Plot 1: Extraction Failure Rate (empty answers) by Round ---")
    plot_extraction_failure_rate(records, output_dir)

    print("\n--- Plot 2: Format Error Rate (text has answer, extraction failed) by Round ---")
    plot_format_error_rate(records, output_dir)

    print("\n--- Plot 3: Debate Accuracy by Round ---")
    plot_accuracy(records, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
