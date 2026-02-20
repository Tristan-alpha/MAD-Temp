"""
分析错误样本中，有多少是格式不符（答案提取失败）导致的，有多少是实际推理/计算错误导致的。

错误分类（针对每个 agent 的单次回复）：
  1. extraction_failure: 正则完全没有匹配到 {…}，final_answer 为空字符串 ""
  2. format_mismatch: 提取到了答案，但提取结果与模型实际意图不符
       - 对于 MCQ：模型在回答文本中最后提到了正确答案选项 (如 "(D)")，但提取到的却是其他选项
       - 对于 arithmetic/gsm8k：模型在回答文本中出现了正确数值，但提取到的是其他数值
  3. reasoning_error: 模型本身推理/计算错误，即使完美提取也会得到错误答案

同时在 debate 层面（投票/多数表决）也做分析：
  - debate_extraction_failure: 所有 agent 都没提取到答案（全空）
  - debate_format_error: 至少有一个 agent 的格式问题影响了最终投票结果
    （即如果把格式错误的 agent 修正后，投票结果会变成正确的）
  - debate_reasoning_error: 即使所有 agent 都完美提取，多数答案仍然错误
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
# 答案提取逻辑（复刻 evaluator.py 中的逻辑）
# ──────────────────────────────────────────

def extract_arithmetic_answer(response):
    """从回复中提取数值答案（复刻 evaluate_arithmetics）"""
    try:
        pred = re.findall(r"\{(.*?)\}", response)[-1]
        pred = float(pred.replace("final answer:", "").strip())
        return np.round(pred, 1)
    except:
        return ""


def extract_mcq_answer(response):
    """从回复中提取选择题答案（复刻 evaluate_mcq）"""
    try:
        pred = re.findall(r"\{(.*?)\}", response)[-1]
        pred = pred.replace("final answer:", "").strip()
        if len(pred) == 0:
            return ""
        elif len(pred) < 3:
            pred = pred[0]
            return f"({pred})"
        else:
            pred = pred[1]
            return f"({pred})"
    except:
        return ""


# ──────────────────────────────────────────
# 宽松的答案搜索（用于判断模型是否"知道"正确答案）
# ──────────────────────────────────────────

def response_contains_correct_arithmetic(response, correct_answer):
    """检查回复文本中是否包含正确的数值答案"""
    correct_val = np.round(float(correct_answer), 1)
    # 搜索所有数字
    numbers = re.findall(r'[\d]+\.?\d*', response)
    for num_str in numbers:
        try:
            val = np.round(float(num_str), 1)
            if val == correct_val:
                return True
        except:
            continue
    return False


def response_contains_correct_mcq(response, correct_answer):
    """检查回复文本中是否在 {} 外面提到了正确选项"""
    # correct_answer 格式如 "(D)"
    letter = correct_answer.strip("()")
    
    # 去掉 {…} 中的内容，只看正文
    text_without_braces = re.sub(r"\{.*?\}", "", response)
    
    # 检查是否在正文中出现了该选项 (如 "(D)" 或 "answer is D" 等)
    patterns = [
        rf'\({letter}\)',                          # (D)
        rf'answer\s*(?:is|:)\s*\(?{letter}\)?',   # answer is D / answer: (D)
        rf'option\s+\(?{letter}\)?',               # option D / option (D)
        rf'choose\s+\(?{letter}\)?',               # choose D
        rf'correct\s+(?:answer|option)\s+(?:is\s+)?\(?{letter}\)?',
    ]
    for pat in patterns:
        if re.search(pat, text_without_braces, re.IGNORECASE):
            return True
    return False


# ──────────────────────────────────────────
# 判断模型回答文本中的"实际意图答案"
# ──────────────────────────────────────────

def get_intended_arithmetic_answer(response):
    """
    尝试用多种方式从回复中推断模型的实际意图答案（不限于 {…} 格式）。
    优先用 {} ，然后尝试各种常见模式，最后用回复中最后出现的数字。
    """
    # 1) 先尝试标准格式
    std = extract_arithmetic_answer(response)
    if std != "":
        return std
    
    # 2) 尝试常见非标准模式
    patterns = [
        r'(?:final\s+answer|the\s+answer)\s*(?:is|:)\s*([\d]+\.?\d*)',
        r'(?:total|result)\s*(?:is|=|:)\s*([\d]+\.?\d*)',
        r'\*\*\s*([\d]+\.?\d*)\s*\*\*',  # **123**
        r'\\boxed\{([\d]+\.?\d*)\}',      # \boxed{123}
        r'=\s*([\d]+\.?\d*)\s*$',          # = 123 at end of line
    ]
    for pat in patterns:
        matches = re.findall(pat, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            try:
                return np.round(float(matches[-1]), 1)
            except:
                continue
    
    # 3) 最后用回复中最后出现的独立数字
    numbers = re.findall(r'(?<![.\w])([\d]+\.?\d*)(?![.\w])', response)
    if numbers:
        try:
            return np.round(float(numbers[-1]), 1)
        except:
            pass
    
    return ""


def get_intended_mcq_answer(response):
    """
    尝试用多种方式从回复中推断模型的实际意图答案。
    """
    # 1) 先尝试标准格式
    std = extract_mcq_answer(response)
    if std != "":
        return std
    
    # 2) 尝试常见非标准模式
    # 搜索 "answer is (X)" 或 "answer: (X)" 等
    patterns = [
        r'(?:final\s+answer|the\s+answer|my\s+answer)\s*(?:is|:)\s*\(?([A-E])\)?',
        r'(?:correct\s+(?:answer|option))\s*(?:is|:)\s*\(?([A-E])\)?',
        r'(?:choose|select)\s+\(?([A-E])\)?',
    ]
    for pat in patterns:
        matches = re.findall(pat, response, re.IGNORECASE)
        if matches:
            return f"({matches[-1].upper()})"
    
    # 3) 找回复末尾最后出现的选项 (X)
    matches = re.findall(r'\(([A-E])\)', response)
    if matches:
        return f"({matches[-1].upper()})"
    
    return ""


# ──────────────────────────────────────────
# 核心分析函数
# ──────────────────────────────────────────

def classify_agent_error(response, extracted_answer, correct_answer, task_type):
    """
    对一个 agent 的一次错误回复进行分类。
    前提：该 agent 的提取答案 != 正确答案。
    
    返回:
      "extraction_failure" - 完全没有提取到答案
      "format_mismatch"   - 提取到了但模型实际意图是正确答案（提取出错）
      "reasoning_error"   - 模型确实给出了错误答案
    """
    if extracted_answer == "":
        # 没有提取到任何答案
        if task_type == "arithmetic":
            intended = get_intended_arithmetic_answer(response)
            if intended != "" and intended == np.round(float(correct_answer), 1):
                return "extraction_failure_but_correct"  # 模型答对了但没提取到
            else:
                return "extraction_failure_and_wrong"    # 模型也没答对
        else:  # mcq
            intended = get_intended_mcq_answer(response)
            if intended != "" and intended == correct_answer:
                return "extraction_failure_but_correct"
            else:
                return "extraction_failure_and_wrong"
    else:
        # 提取到了答案，但答案错误
        if task_type == "arithmetic":
            # 检查模型是否在文本中实际给出了正确答案（但被错误提取）
            intended = get_intended_arithmetic_answer(response)
            if response_contains_correct_arithmetic(response, correct_answer):
                # 文本中有正确答案，但提取到了错误的
                return "format_mismatch"
            else:
                return "reasoning_error"
        else:  # mcq
            if response_contains_correct_mcq(response, correct_answer):
                return "format_mismatch"
            else:
                return "reasoning_error"


def classify_debate_error(round_data, task_type):
    """
    对一个 debate round 的错误进行分类。
    前提：debate_answer_iscorr == False。
    
    返回 dict 包含:
      - agent_errors: 每个 agent 的错误分类
      - debate_error_type: debate 层面的错误分类
      - details: 调试信息
    """
    responses = round_data["responses"]
    final_answers = round_data["final_answers"]
    correct_answer = round_data["answer"]
    debate_answer = round_data["debate_answer"]
    
    agent_names = list(responses.keys())
    
    # 分析每个 agent
    agent_errors = []
    corrected_answers = []
    
    for i, agent_name in enumerate(agent_names):
        resp = responses[agent_name]
        extracted = final_answers[i]
        
        if task_type == "arithmetic":
            is_correct = (extracted != "" and extracted == np.round(float(correct_answer), 1))
        else:
            is_correct = (extracted == correct_answer)
        
        if is_correct:
            agent_errors.append("correct")
            corrected_answers.append(extracted)
        else:
            err_type = classify_agent_error(resp, extracted, correct_answer, task_type)
            agent_errors.append(err_type)
            
            # 推断模型的实际意图答案
            if task_type == "arithmetic":
                intended = get_intended_arithmetic_answer(resp)
            else:
                intended = get_intended_mcq_answer(resp)
            
            # 如果是格式问题，修正为实际意图答案
            if err_type in ("extraction_failure_but_correct", "format_mismatch"):
                corrected_answers.append(correct_answer)
            else:
                corrected_answers.append(intended if intended != "" else extracted)
    
    # 判断 debate 层面的错误类型
    # 1) 所有 agent 都没提取到答案
    all_empty = all(a == "" for a in final_answers)
    
    # 2) 修正格式错误后重新投票
    valid_corrected = [a for a in corrected_answers if a != ""]
    if valid_corrected:
        counter = collections.Counter(valid_corrected)
        max_count = max(counter.values())
        most_common = [k for k, v in counter.items() if v == max_count]
        
        if task_type == "arithmetic":
            corrected_debate_would_be_correct = any(
                mc == np.round(float(correct_answer), 1) for mc in most_common
            )
        else:
            corrected_debate_would_be_correct = any(
                mc == correct_answer for mc in most_common
            )
    else:
        corrected_debate_would_be_correct = False
    
    # 分类
    has_format_error = any(e in (
        "extraction_failure_but_correct", "extraction_failure_and_wrong", "format_mismatch"
    ) for e in agent_errors)
    
    has_extraction_failure_correct = any(e == "extraction_failure_but_correct" for e in agent_errors)
    has_format_mismatch = any(e == "format_mismatch" for e in agent_errors)
    
    if all_empty:
        debate_error_type = "all_extraction_failure"
    elif corrected_debate_would_be_correct and has_format_error:
        debate_error_type = "format_caused_wrong_vote"
    elif has_format_error:
        debate_error_type = "format_error_but_still_wrong"
    else:
        debate_error_type = "reasoning_error"
    
    return {
        "agent_errors": agent_errors,
        "debate_error_type": debate_error_type,
        "corrected_answers": corrected_answers,
        "corrected_debate_would_be_correct": corrected_debate_would_be_correct,
    }


def detect_task_type(filename):
    """根据文件名判断任务类型"""
    basename = os.path.basename(filename).lower()
    if any(x in basename for x in ["gsm8k", "arithmetics"]):
        return "arithmetic"
    elif any(x in basename for x in ["formal_logic", "pro_medicine", "csqa", "hellaswag", "hh_rlhf"]):
        return "mcq"
    else:
        return None


def analyze_file(filepath, verbose=False):
    """分析单个 JSONL 文件"""
    task_type = detect_task_type(filepath)
    if task_type is None:
        print(f"  跳过未知类型的文件: {filepath}")
        return None
    
    basename = os.path.basename(filepath)
    
    # 统计
    stats = {
        "file": basename,
        "task_type": task_type,
        "total_samples": 0,
        "rounds": {},
    }
    
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
    
    stats["total_samples"] = len(samples)
    
    for sample_idx, sample_data in enumerate(samples):
        for round_key in sorted(sample_data.keys(), key=int):
            rd = sample_data[round_key]
            
            if round_key not in stats["rounds"]:
                stats["rounds"][round_key] = {
                    "total": 0,
                    "correct": 0,
                    "wrong": 0,
                    # Agent 级别统计
                    "agent_correct": 0,
                    "agent_extraction_failure_but_correct": 0,
                    "agent_extraction_failure_and_wrong": 0,
                    "agent_format_mismatch": 0,
                    "agent_reasoning_error": 0,
                    "agent_total": 0,
                    # Debate 级别统计
                    "debate_all_extraction_failure": 0,
                    "debate_format_caused_wrong_vote": 0,
                    "debate_format_error_but_still_wrong": 0,
                    "debate_reasoning_error": 0,
                    # 示例
                    "examples": [],
                }
            
            rs = stats["rounds"][round_key]
            rs["total"] += 1
            
            is_correct = rd.get("debate_answer_iscorr", False)
            if is_correct:
                rs["correct"] += 1
                # 统计 agent 级别正确数
                n_agents = len(rd.get("final_answers", []))
                rs["agent_total"] += n_agents
                rs["agent_correct"] += n_agents
            else:
                rs["wrong"] += 1
                result = classify_debate_error(rd, task_type)
                
                # Agent 级别统计
                n_agents = len(rd.get("final_answers", []))
                rs["agent_total"] += n_agents
                for ae in result["agent_errors"]:
                    if ae == "correct":
                        rs["agent_correct"] += 1
                    elif ae == "extraction_failure_but_correct":
                        rs["agent_extraction_failure_but_correct"] += 1
                    elif ae == "extraction_failure_and_wrong":
                        rs["agent_extraction_failure_and_wrong"] += 1
                    elif ae == "format_mismatch":
                        rs["agent_format_mismatch"] += 1
                    elif ae == "reasoning_error":
                        rs["agent_reasoning_error"] += 1
                
                # Debate 级别统计
                dt = result["debate_error_type"]
                rs[f"debate_{dt}"] = rs.get(f"debate_{dt}", 0) + 1
                
                # 保存一些错误示例
                if len(rs["examples"]) < 3:
                    example = {
                        "sample_idx": sample_idx,
                        "correct_answer": rd["answer"],
                        "debate_answer": rd["debate_answer"],
                        "final_answers": rd["final_answers"],
                        "agent_errors": result["agent_errors"],
                        "debate_error_type": result["debate_error_type"],
                        "corrected_debate_would_be_correct": result["corrected_debate_would_be_correct"],
                    }
                    rs["examples"].append(example)
    
    return stats


def print_stats(stats):
    """打印分析结果"""
    print(f"\n{'='*80}")
    print(f"文件: {stats['file']}")
    print(f"任务类型: {stats['task_type']}")
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
        
        # Agent 级别分析
        agent_total_wrong = (
            rs["agent_extraction_failure_but_correct"] +
            rs["agent_extraction_failure_and_wrong"] +
            rs["agent_format_mismatch"] +
            rs["agent_reasoning_error"]
        )
        
        print(f"\n  [Agent 级别错误分析] (共 {agent_total_wrong} 个错误回复 / {rs['agent_total']} 个 agent 回复)")
        if agent_total_wrong > 0:
            ef_correct = rs["agent_extraction_failure_but_correct"]
            ef_wrong = rs["agent_extraction_failure_and_wrong"]
            fm = rs["agent_format_mismatch"]
            re_err = rs["agent_reasoning_error"]
            
            total_format = ef_correct + ef_wrong + fm
            
            print(f"    格式错误 (共): {total_format}/{agent_total_wrong} = {total_format/agent_total_wrong*100:.1f}%")
            print(f"      - 提取失败但模型答对: {ef_correct} ({ef_correct/agent_total_wrong*100:.1f}%)")
            print(f"      - 提取失败且模型答错: {ef_wrong} ({ef_wrong/agent_total_wrong*100:.1f}%)")
            print(f"      - 提取到错误答案(文本中有正确答案): {fm} ({fm/agent_total_wrong*100:.1f}%)")
            print(f"    推理/计算错误: {re_err}/{agent_total_wrong} = {re_err/agent_total_wrong*100:.1f}%")
        
        # Debate 级别分析
        print(f"\n  [Debate 级别错误分析] (共 {wrong} 个错误样本)")
        d_all_ef = rs["debate_all_extraction_failure"]
        d_format_vote = rs["debate_format_caused_wrong_vote"]
        d_format_still = rs["debate_format_error_but_still_wrong"]
        d_reason = rs["debate_reasoning_error"]
        
        total_format_debate = d_all_ef + d_format_vote
        
        print(f"    格式导致的错误 (修正后能答对): {total_format_debate}/{wrong} = {total_format_debate/wrong*100:.1f}%")
        print(f"      - 所有agent提取失败: {d_all_ef} ({d_all_ef/wrong*100:.1f}%)")
        print(f"      - 格式错误导致投票结果错误: {d_format_vote} ({d_format_vote/wrong*100:.1f}%)")
        print(f"    有格式错误但修正后仍然答错: {d_format_still}/{wrong} = {d_format_still/wrong*100:.1f}%")
        print(f"    纯推理/计算错误 (无格式问题): {d_reason}/{wrong} = {d_reason/wrong*100:.1f}%")
        
        # 示例
        if rs["examples"]:
            print(f"\n  [错误示例]")
            for ex in rs["examples"][:2]:
                print(f"    Sample#{ex['sample_idx']}: "
                      f"正确={ex['correct_answer']}, "
                      f"投票={ex['debate_answer']}, "
                      f"各agent={ex['final_answers']}, "
                      f"分类={ex['agent_errors']}, "
                      f"debate分类={ex['debate_error_type']}")


def print_aggregate_summary(all_stats):
    """打印跨文件的汇总统计"""
    print(f"\n{'#'*80}")
    print(f"{'#'*20}  汇总统计  {'#'*20}")
    print(f"{'#'*80}")
    
    # 按 (dataset, model) 分组
    groups = {}
    for stats in all_stats:
        if stats is None:
            continue
        fname = stats["file"]
        # 解析文件名 e.g. gsm8k_50__qwen3-4b_N=3_R=3_TR=2_TT=20.0.jsonl
        parts = fname.split("__")
        dataset = parts[0].rsplit("_", 1)[0] if parts else "unknown"
        model = parts[1].split("_N=")[0] if len(parts) > 1 else "unknown"
        key = (dataset, model)
        
        if key not in groups:
            groups[key] = []
        groups[key].append(stats)
    
    for (dataset, model), stat_list in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"数据集: {dataset}, 模型: {model}")
        print(f"{'='*60}")
        
        # 汇总所有 round 的数据
        agg = {
            "total_wrong": 0,
            "agent_total_wrong": 0,
            "agent_extraction_failure_but_correct": 0,
            "agent_extraction_failure_and_wrong": 0,
            "agent_format_mismatch": 0,
            "agent_reasoning_error": 0,
            "debate_all_extraction_failure": 0,
            "debate_format_caused_wrong_vote": 0,
            "debate_format_error_but_still_wrong": 0,
            "debate_reasoning_error": 0,
        }
        
        for stats in stat_list:
            for rk, rs in stats["rounds"].items():
                agg["total_wrong"] += rs["wrong"]
                agg["agent_extraction_failure_but_correct"] += rs["agent_extraction_failure_but_correct"]
                agg["agent_extraction_failure_and_wrong"] += rs["agent_extraction_failure_and_wrong"]
                agg["agent_format_mismatch"] += rs["agent_format_mismatch"]
                agg["agent_reasoning_error"] += rs["agent_reasoning_error"]
                agg["debate_all_extraction_failure"] += rs["debate_all_extraction_failure"]
                agg["debate_format_caused_wrong_vote"] += rs["debate_format_caused_wrong_vote"]
                agg["debate_format_error_but_still_wrong"] += rs["debate_format_error_but_still_wrong"]
                agg["debate_reasoning_error"] += rs["debate_reasoning_error"]
        
        agg["agent_total_wrong"] = (
            agg["agent_extraction_failure_but_correct"] +
            agg["agent_extraction_failure_and_wrong"] +
            agg["agent_format_mismatch"] +
            agg["agent_reasoning_error"]
        )
        
        tw = agg["total_wrong"]
        atw = agg["agent_total_wrong"]
        
        if tw == 0:
            print("  无错误样本")
            continue
        
        total_format_agent = (
            agg["agent_extraction_failure_but_correct"] +
            agg["agent_extraction_failure_and_wrong"] +
            agg["agent_format_mismatch"]
        )
        
        print(f"\n  跨所有温度/轮次, 共 {tw} 个错误 debate 样本")
        print(f"\n  [Agent 级别] {atw} 个错误 agent 回复:")
        if atw > 0:
            print(f"    格式错误: {total_format_agent}/{atw} = {total_format_agent/atw*100:.1f}%")
            print(f"      - 提取失败但答对: {agg['agent_extraction_failure_but_correct']}")
            print(f"      - 提取失败且答错: {agg['agent_extraction_failure_and_wrong']}")
            print(f"      - 提取到错误答案: {agg['agent_format_mismatch']}")
            print(f"    推理错误: {agg['agent_reasoning_error']}/{atw} = {agg['agent_reasoning_error']/atw*100:.1f}%")
        
        total_format_debate = agg["debate_all_extraction_failure"] + agg["debate_format_caused_wrong_vote"]
        
        print(f"\n  [Debate 级别] {tw} 个错误样本:")
        print(f"    格式导致 (修正可救): {total_format_debate}/{tw} = {total_format_debate/tw*100:.1f}%")
        print(f"    有格式错误但仍答错: {agg['debate_format_error_but_still_wrong']}/{tw} = {agg['debate_format_error_but_still_wrong']/tw*100:.1f}%")
        print(f"    纯推理错误: {agg['debate_reasoning_error']}/{tw} = {agg['debate_reasoning_error']/tw*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="分析错误样本: 格式错误 vs 推理错误")
    parser.add_argument('--history_dir', type=str, default=HISTORY_DIR)
    parser.add_argument('--dataset', type=str, default=None,
                        help="只分析指定数据集 (gsm8k, formal_logic, pro_medicine, ...)")
    parser.add_argument('--file', type=str, default=None,
                        help="只分析指定文件 (文件名或完整路径)")
    parser.add_argument('--verbose', action='store_true',
                        help="输出更多细节")
    parser.add_argument('--round', type=str, default=None,
                        help="只分析指定轮次 (如 0, 1, 2, 3)")
    args = parser.parse_args()
    
    # 收集所有 JSONL 文件
    if args.file:
        if os.path.isabs(args.file):
            files = [args.file]
        else:
            files = glob.glob(os.path.join(args.history_dir, "**", args.file), recursive=True)
    else:
        files = glob.glob(os.path.join(args.history_dir, "**", "*.jsonl"), recursive=True)
        files = [f for f in files if "previous_backup" not in f and "accuracy_summary" not in f]
        
        if args.dataset:
            files = [f for f in files if os.path.basename(f).startswith(args.dataset)]
    
    files.sort()
    print(f"找到 {len(files)} 个文件待分析")
    
    all_stats = []
    for filepath in files:
        print(f"\n分析: {os.path.basename(filepath)}...")
        stats = analyze_file(filepath, verbose=args.verbose)
        if stats is not None:
            # 如果指定了 round，只保留该 round
            if args.round is not None:
                stats["rounds"] = {
                    k: v for k, v in stats["rounds"].items() if k == args.round
                }
            print_stats(stats)
            all_stats.append(stats)
    
    # 打印汇总
    if len(all_stats) > 1:
        print_aggregate_summary(all_stats)


if __name__ == "__main__":
    main()
