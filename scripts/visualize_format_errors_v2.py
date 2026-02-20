"""
可视化 GSM8K 格式错误分析结果 (v2 简化版):
  1. 折线图: 横轴温度, 纵轴格式可修复比例
  2. 柱状图(Qwen2.5): 横轴 debate round, 纵轴格式可修复比例 (所有温度聚合)
  3. 柱状图(Qwen3):   同上
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
    classify_debate_error, detect_task_type, HISTORY_DIR,
)


def parse_filename(filepath):
    basename = os.path.basename(filepath)
    m = re.match(
        r"([a-zA-Z0-9_]+)_(\d+)__(.+?)_N=(\d+)_R=(\d+)_TR=(\d+)_TT=([\d.]+)\.jsonl",
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
            "filepath": filepath,
        }
    return None


def collect_gsm8k_files(history_dir=HISTORY_DIR):
    all_files = glob.glob(os.path.join(history_dir, "**", "*.jsonl"), recursive=True)
    files = [f for f in all_files
             if "previous_backup" not in f and "accuracy_summary" not in f]
    # 只保留 gsm8k
    files = [f for f in files if os.path.basename(f).startswith("gsm8k")]

    records = []
    for fp in files:
        info = parse_filename(fp)
        if info is not None:
            records.append(info)
    return records


def compute_format_error_stats(filepath):
    """
    计算每个 round 的 {total, wrong, format_fixable}。
    使用 v2 的 classify_debate_error（二分类）。
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
                round_stats[round_key] = {"total": 0, "wrong": 0, "format_fixable": 0}

            rs = round_stats[round_key]
            rs["total"] += 1

            is_correct = rd.get("debate_answer_iscorr", False)
            if not is_correct:
                rs["wrong"] += 1
                result = classify_debate_error(rd)
                if result["debate_error_type"] == "format_fixable":
                    rs["format_fixable"] += 1

    return round_stats


# ──────────────────────────────────────────────────
# 图1: 折线图 - 温度 vs 格式可修复错误比例
# ──────────────────────────────────────────────────

def plot_temperature_vs_format_ratio(records, output_dir):
    # 按 (dataset, model, temperature) 聚合
    agg = {}
    for rec in records:
        key = (rec["dataset"], rec["model"])
        temp = rec["temperature"]

        round_stats = compute_format_error_stats(rec["filepath"])
        if round_stats is None:
            continue

        if key not in agg:
            agg[key] = {}
        if temp not in agg[key]:
            agg[key][temp] = {"wrong": 0, "format_fixable": 0}

        for rk, rs in round_stats.items():
            agg[key][temp]["wrong"] += rs["wrong"]
            agg[key][temp]["format_fixable"] += rs["format_fixable"]

    fig, ax = plt.subplots(figsize=(12, 7))

    color_cycle = plt.cm.tab10.colors
    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    sorted_keys = sorted(agg.keys())
    for idx, key in enumerate(sorted_keys):
        dataset, model = key
        temp_data = agg[key]

        temps = sorted(temp_data.keys())
        ratios = []
        for t in temps:
            d = temp_data[t]
            ratios.append(d["format_fixable"] / d["wrong"] if d["wrong"] > 0 else 0.0)

        color = color_cycle[idx % len(color_cycle)]
        marker = marker_cycle[idx % len(marker_cycle)]
        label = f"{dataset} / {model}"

        ax.plot(temps, ratios, marker=marker, color=color, label=label,
                linewidth=2, markersize=8, alpha=0.85)

    ax.set_xlabel("Temperature", fontsize=14)
    ax.set_ylabel("Format-Fixable Error Ratio\n(among wrong samples)", fontsize=14)
    ax.set_title("GSM8K: Format-Fixable Error Ratio vs. Temperature (v2)", fontsize=15)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "v2_format_error_ratio_vs_temperature.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────
# 图2 & 图3: 柱状图 - debate round vs 格式可修复比例
# ──────────────────────────────────────────────────

def plot_round_bar_charts(records, output_dir):
    # 按 (dataset, model, round) 聚合
    agg = {}
    for rec in records:
        key = (rec["dataset"], rec["model"])

        round_stats = compute_format_error_stats(rec["filepath"])
        if round_stats is None:
            continue

        if key not in agg:
            agg[key] = {}

        for rk, rs in round_stats.items():
            rk_int = int(rk)
            if rk_int not in agg[key]:
                agg[key][rk_int] = {"wrong": 0, "format_fixable": 0, "total": 0}
            agg[key][rk_int]["wrong"] += rs["wrong"]
            agg[key][rk_int]["format_fixable"] += rs["format_fixable"]
            agg[key][rk_int]["total"] += rs["total"]

    qwen25_keys = sorted([k for k in agg.keys() if "qwen2.5" in k[1] or "qwen2" in k[1]])
    qwen3_keys = sorted([k for k in agg.keys() if "qwen3" in k[1]])

    for group_name, group_keys in [("Qwen2.5", qwen25_keys), ("Qwen3", qwen3_keys)]:
        if not group_keys:
            print(f"No data for {group_name}, skipping.")
            continue

        all_rounds = set()
        for key in group_keys:
            all_rounds.update(agg[key].keys())
        all_rounds = sorted(all_rounds)

        n_groups = len(group_keys)
        n_rounds = len(all_rounds)

        fig, ax = plt.subplots(figsize=(12, 7))
        bar_width = 0.8 / n_groups
        color_cycle = plt.cm.Set2.colors if n_groups <= 8 else plt.cm.tab20.colors
        x_base = np.arange(n_rounds)

        for idx, key in enumerate(group_keys):
            dataset, model = key
            data = agg[key]

            ratios = []
            for r in all_rounds:
                if r in data and data[r]["wrong"] > 0:
                    ratios.append(data[r]["format_fixable"] / data[r]["wrong"])
                else:
                    ratios.append(0.0)

            offset = (idx - n_groups / 2 + 0.5) * bar_width
            bars = ax.bar(x_base + offset, ratios, bar_width * 0.9,
                         label=f"{dataset} / {model}",
                         color=color_cycle[idx % len(color_cycle)],
                         edgecolor='gray', linewidth=0.5, alpha=0.85)

            for bar_obj, ratio in zip(bars, ratios):
                if ratio > 0:
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                           bar_obj.get_height() + 0.01,
                           f"{ratio:.1%}", ha='center', va='bottom', fontsize=9)

        ax.set_xlabel("Debate Round", fontsize=14)
        ax.set_ylabel("Format-Fixable Error Ratio\n(among wrong samples)", fontsize=14)
        ax.set_title(f"GSM8K {group_name}: Format-Fixable Error Ratio by Round (v2)\n"
                     f"(aggregated across all temperatures)", fontsize=15)
        ax.set_xticks(x_base)
        ax.set_xticklabels([f"Round {r}" for r in all_rounds], fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.15)

        plt.tight_layout()
        safe_name = group_name.lower().replace(".", "")
        out_path = os.path.join(output_dir, f"v2_format_error_by_round_{safe_name}.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
        plt.close(fig)


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

    print("\n--- Plot 1: Temperature vs Format Error Ratio ---")
    plot_temperature_vs_format_ratio(records, output_dir)

    print("\n--- Plot 2 & 3: Round Bar Charts ---")
    plot_round_bar_charts(records, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
