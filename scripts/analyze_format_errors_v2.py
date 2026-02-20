"""
分析 GSM8K 错误样本：格式错误 vs 推理错误（简化版 v2）

Agent 级别分类（二分类）：
  - format_error:    提取答案 != 正确答案，但回复全文中包含正确数值
  - reasoning_error: 提取答案 != 正确答案，且回复全文中不包含正确数值

Debate 级别分类（二分类）：
  对于每个 format_error 的 agent，修正答案 = 正确答案
  对于每个 reasoning_error 的 agent，修正答案 = 原始提取值（可能为 ""）
  修正后重新投票：
  - format_fixable:   修正后投票正确 → 格式导致（修正可救）
  - reasoning_error:   修正后投票仍错 → 推理错误
"""

import json
import glob
import os
import re
import argparse
import collections
import numpy as np
from pathlib import Path


HISTORY_DIR = "/export/home3/dazhou/debate-or-vote/out/history"


# ──────────────────────────────────────────
# 答案提取（复刻 evaluator.py）
# ──────────────────────────────────────────

def extract_arithmetic_answer(response):
    """从回复中用标准 {…} 格式提取数值答案"""
    try:
        pred = re.findall(r"\{(.*?)\}", response)[-1]
        pred = float(pred.replace("final answer:", "").strip())
        return np.round(pred, 1)
    except:
        return ""


# ──────────────────────────────────────────
# 正确答案搜索
# ──────────────────────────────────────────

def response_contains_correct_number(response, correct_answer):
    """
    在整个回复文本中搜索是否出现了正确的数值。
    匹配所有数字（包括小数），与正确答案做 round(1) 比较。
    """
    correct_val = np.round(float(correct_answer), 1)
    # 匹配整数和小数
    numbers = re.findall(r'[\d]+\.?\d*', response)
    for num_str in numbers:
        try:
            val = np.round(float(num_str), 1)
            if val == correct_val:
                return True
        except:
            continue
    return False


# ──────────────────────────────────────────
# 核心分类函数
# ──────────────────────────────────────────

def classify_agent_error(response, extracted_answer, correct_answer):
    """
    Agent 级别二分类。
    前提：该 agent 的提取答案 != 正确答案。

    返回:
      "format_error"     - 回复全文中包含正确数值（格式问题导致没提取到）
      "reasoning_error"  - 回复全文中不包含正确数值（模型本身算错了）
    """
    if response_contains_correct_number(response, correct_answer):
        return "format_error"
    else:
        return "reasoning_error"


def classify_debate_error(round_data):
    """
    Debate 级别二分类。
    前提：debate_answer_iscorr == False。

    步骤：
      1. 对每个 agent 做二分类
      2. format_error 的 agent 修正答案 = 正确答案
         reasoning_error 的 agent 修正答案 = 原始提取值
      3. 修正后重新多数投票
      4. 投票正确 → format_fixable，否则 → reasoning_error
    """
    responses = round_data["responses"]
    final_answers = round_data["final_answers"]
    correct_answer = round_data["answer"]
    correct_val = np.round(float(correct_answer), 1)

    agent_names = list(responses.keys())

    agent_errors = []
    corrected_answers = []

    for i, agent_name in enumerate(agent_names):
        resp = responses[agent_name]
        extracted = final_answers[i]

        # 判断该 agent 是否已经正确
        is_correct = (extracted != "" and extracted == correct_val)

        if is_correct:
            agent_errors.append("correct")
            corrected_answers.append(extracted)
        else:
            err_type = classify_agent_error(resp, extracted, correct_answer)
            agent_errors.append(err_type)

            if err_type == "format_error":
                corrected_answers.append(correct_val)
            else:
                # 保持原始提取值（可能是 "" 或错误数值）
                corrected_answers.append(extracted)

    # 修正后重新投票
    valid_corrected = [a for a in corrected_answers if a != ""]
    if valid_corrected:
        counter = collections.Counter(valid_corrected)
        max_count = max(counter.values())
        most_common = [k for k, v in counter.items() if v == max_count]
        corrected_is_correct = any(mc == correct_val for mc in most_common)
    else:
        corrected_is_correct = False

    debate_error_type = "format_fixable" if corrected_is_correct else "reasoning_error"

    return {
        "agent_errors": agent_errors,
        "corrected_answers": corrected_answers,
        "debate_error_type": debate_error_type,
    }


def detect_task_type(filename):
    """根据文件名判断任务类型"""
    basename = os.path.basename(filename).lower()
    if any(x in basename for x in ["gsm8k", "arithmetics"]):
        return "arithmetic"
    return None  # v2 只处理 arithmetic


# ──────────────────────────────────────────
# 文件级分析
# ──────────────────────────────────────────

def analyze_file(filepath, verbose=False):
    """分析单个 JSONL 文件"""
    task_type = detect_task_type(filepath)
    if task_type is None:
        return None

    basename = os.path.basename(filepath)

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

    stats = {
        "file": basename,
        "task_type": task_type,
        "total_samples": len(samples),
        "rounds": {},
    }

    for sample_idx, sample_data in enumerate(samples):
        for round_key in sorted(sample_data.keys(), key=int):
            rd = sample_data[round_key]

            if round_key not in stats["rounds"]:
                stats["rounds"][round_key] = {
                    "total": 0, "correct": 0, "wrong": 0,
                    # Agent 级别
                    "agent_total": 0,
                    "agent_correct": 0,
                    "agent_format_error": 0,
                    "agent_reasoning_error": 0,
                    # Debate 级别
                    "debate_format_fixable": 0,
                    "debate_reasoning_error": 0,
                    # 示例
                    "examples": [],
                }

            rs = stats["rounds"][round_key]
            rs["total"] += 1

            is_correct = rd.get("debate_answer_iscorr", False)
            if is_correct:
                rs["correct"] += 1
                n_agents = len(rd.get("final_answers", []))
                rs["agent_total"] += n_agents
                rs["agent_correct"] += n_agents
            else:
                rs["wrong"] += 1
                result = classify_debate_error(rd)

                n_agents = len(rd.get("final_answers", []))
                rs["agent_total"] += n_agents
                for ae in result["agent_errors"]:
                    if ae == "correct":
                        rs["agent_correct"] += 1
                    elif ae == "format_error":
                        rs["agent_format_error"] += 1
                    elif ae == "reasoning_error":
                        rs["agent_reasoning_error"] += 1

                if result["debate_error_type"] == "format_fixable":
                    rs["debate_format_fixable"] += 1
                else:
                    rs["debate_reasoning_error"] += 1

                if len(rs["examples"]) < 3:
                    rs["examples"].append({
                        "sample_idx": sample_idx,
                        "correct_answer": rd["answer"],
                        "debate_answer": rd["debate_answer"],
                        "final_answers": rd["final_answers"],
                        "agent_errors": result["agent_errors"],
                        "corrected_answers": [
                            float(a) if a != "" else "" for a in result["corrected_answers"]
                        ],
                        "debate_error_type": result["debate_error_type"],
                    })

    return stats


# ──────────────────────────────────────────
# 打印
# ──────────────────────────────────────────

def print_stats(stats):
    print(f"\n{'='*80}")
    print(f"文件: {stats['file']}")
    print(f"总样本数: {stats['total_samples']}")
    print(f"{'='*80}")

    for round_key in sorted(stats["rounds"].keys(), key=int):
        rs = stats["rounds"][round_key]
        total = rs["total"]
        correct = rs["correct"]
        wrong = rs["wrong"]

        print(f"\n--- Round {round_key} ---")
        print(f"  准确率: {correct}/{total} = {correct/total*100:.1f}%")
        print(f"  错误数: {wrong}/{total} = {wrong/total*100:.1f}%")

        if wrong == 0:
            print(f"  (无错误样本)")
            continue

        # Agent 级别
        a_fmt = rs["agent_format_error"]
        a_rsn = rs["agent_reasoning_error"]
        a_total_wrong = a_fmt + a_rsn

        print(f"\n  [Agent 级别] {a_total_wrong} 个错误回复 / {rs['agent_total']} 个 agent 回复")
        if a_total_wrong > 0:
            print(f"    格式错误 (文本含正确答案): {a_fmt}/{a_total_wrong} = {a_fmt/a_total_wrong*100:.1f}%")
            print(f"    推理错误 (文本无正确答案): {a_rsn}/{a_total_wrong} = {a_rsn/a_total_wrong*100:.1f}%")

        # Debate 级别
        d_fmt = rs["debate_format_fixable"]
        d_rsn = rs["debate_reasoning_error"]

        print(f"\n  [Debate 级别] {wrong} 个错误样本")
        print(f"    格式可修复 (修正后投票正确): {d_fmt}/{wrong} = {d_fmt/wrong*100:.1f}%")
        print(f"    推理错误   (修正后仍然错误): {d_rsn}/{wrong} = {d_rsn/wrong*100:.1f}%")

        if rs["examples"]:
            print(f"\n  [错误示例]")
            for ex in rs["examples"][:2]:
                print(f"    Sample#{ex['sample_idx']}: "
                      f"正确={ex['correct_answer']}, "
                      f"投票={ex['debate_answer']}, "
                      f"提取={ex['final_answers']}, "
                      f"分类={ex['agent_errors']}, "
                      f"修正={ex['corrected_answers']}, "
                      f"debate={ex['debate_error_type']}")


def print_aggregate_summary(all_stats):
    print(f"\n{'#'*80}")
    print(f"{'#'*20}  汇总统计  {'#'*20}")
    print(f"{'#'*80}")

    groups = {}
    for stats in all_stats:
        if stats is None:
            continue
        fname = stats["file"]
        parts = fname.split("__")
        dataset = parts[0].rsplit("_", 1)[0] if parts else "unknown"
        model = parts[1].split("_N=")[0] if len(parts) > 1 else "unknown"
        key = (dataset, model)
        groups.setdefault(key, []).append(stats)

    for (dataset, model), stat_list in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"数据集: {dataset}, 模型: {model}")
        print(f"{'='*60}")

        agg = {"total_wrong": 0,
               "agent_format_error": 0, "agent_reasoning_error": 0,
               "debate_format_fixable": 0, "debate_reasoning_error": 0}

        for stats in stat_list:
            for rk, rs in stats["rounds"].items():
                agg["total_wrong"] += rs["wrong"]
                agg["agent_format_error"] += rs["agent_format_error"]
                agg["agent_reasoning_error"] += rs["agent_reasoning_error"]
                agg["debate_format_fixable"] += rs["debate_format_fixable"]
                agg["debate_reasoning_error"] += rs["debate_reasoning_error"]

        tw = agg["total_wrong"]
        if tw == 0:
            print("  无错误样本")
            continue

        atw = agg["agent_format_error"] + agg["agent_reasoning_error"]
        print(f"\n  跨所有温度/轮次, 共 {tw} 个错误 debate 样本")
        print(f"\n  [Agent 级别] {atw} 个错误 agent 回复:")
        print(f"    格式错误: {agg['agent_format_error']}/{atw} = {agg['agent_format_error']/atw*100:.1f}%")
        print(f"    推理错误: {agg['agent_reasoning_error']}/{atw} = {agg['agent_reasoning_error']/atw*100:.1f}%")
        print(f"\n  [Debate 级别] {tw} 个错误样本:")
        print(f"    格式可修复: {agg['debate_format_fixable']}/{tw} = {agg['debate_format_fixable']/tw*100:.1f}%")
        print(f"    推理错误:   {agg['debate_reasoning_error']}/{tw} = {agg['debate_reasoning_error']/tw*100:.1f}%")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="分析 GSM8K 错误: 格式 vs 推理 (v2 简化版)")
    parser.add_argument('--history_dir', type=str, default=HISTORY_DIR)
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--round', type=str, default=None)
    args = parser.parse_args()

    if args.file:
        if os.path.isabs(args.file):
            files = [args.file]
        else:
            files = glob.glob(os.path.join(args.history_dir, "**", args.file), recursive=True)
    else:
        files = glob.glob(os.path.join(args.history_dir, "**", "*.jsonl"), recursive=True)
        files = [f for f in files if "previous_backup" not in f and "accuracy_summary" not in f]
        # 只保留 gsm8k 文件
        files = [f for f in files if os.path.basename(f).startswith("gsm8k")]

    files.sort()
    print(f"找到 {len(files)} 个 GSM8K 文件待分析")

    all_stats = []
    for filepath in files:
        print(f"\n分析: {os.path.basename(filepath)}...")
        stats = analyze_file(filepath, verbose=args.verbose)
        if stats is not None:
            if args.round is not None:
                stats["rounds"] = {k: v for k, v in stats["rounds"].items() if k == args.round}
            print_stats(stats)
            all_stats.append(stats)

    if len(all_stats) > 1:
        print_aggregate_summary(all_stats)


if __name__ == "__main__":
    main()
