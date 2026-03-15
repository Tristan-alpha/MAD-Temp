import argparse
import glob
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


FILENAME_PATTERN = re.compile(
    r"([a-zA-Z0-9_]+)_(\d+)__(.+?)_N=.*_TR=(\d+)_TT=([\d\.]+)(?:_MNT=(\d+))?\.jsonl$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate temperature hypotheses using answer-change signals from debate history files."
        )
    )
    parser.add_argument("--history_dir", type=str, default="out/history")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--target_agent_idx",
        type=int,
        default=0,
        help="Agent index whose temperature is considered manipulated (default follows main.py).",
    )
    parser.add_argument("--min_samples", type=int, default=1)
    parser.add_argument(
        "--detail_output",
        type=str,
        default="out/history/temp_hypothesis_answer_change_detail_v1.jsonl",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default="out/history/temp_hypothesis_answer_change_grouped_v1.jsonl",
    )
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return ""
        value = float(value)
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.10g}"
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(value).strip()


def parse_final_answers(round_data: Dict[str, Any]) -> List[str]:
    final_answers = round_data.get("final_answers", [])
    if isinstance(final_answers, list):
        return [normalize_answer(v) for v in final_answers]
    if isinstance(final_answers, dict):
        pairs: List[Tuple[int, str]] = []
        for k, v in final_answers.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            pairs.append((idx, normalize_answer(v)))
        pairs.sort(key=lambda x: x[0])
        return [v for _, v in pairs]
    return []


def extract_rounds(sample: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(sample, dict):
        return {}
    if isinstance(sample.get("solutions"), dict):
        rounds = sample["solutions"]
        return {k: v for k, v in rounds.items() if isinstance(v, dict)}

    rounds: Dict[str, Dict[str, Any]] = {}
    for k, v in sample.items():
        if isinstance(k, str) and k.isdigit() and isinstance(v, dict):
            rounds[k] = v
    return rounds


def is_consensus(answers: List[str]) -> bool:
    if not answers:
        return False
    if any(ans == "" for ans in answers):
        return False
    return len(set(answers)) == 1


def parse_experiment_meta(path: str) -> Optional[Dict[str, Any]]:
    basename = os.path.basename(path)
    match = FILENAME_PATTERN.match(basename)
    if not match:
        return None
    return {
        "dataset": match.group(1),
        "data_size": int(match.group(2)),
        "model": match.group(3),
        "target_round": int(match.group(4)),
        "temperature": float(match.group(5)),
        "max_new_tokens": int(match.group(6)) if match.group(6) else 512,
        "source_file": basename,
        "source_path": path,
    }


def compute_file_metrics(
    path: str,
    meta: Dict[str, Any],
    target_agent_idx: int,
    min_samples: int,
) -> Optional[Dict[str, Any]]:
    tr = meta["target_round"]
    if tr == 0:
        # Cannot measure change without previous round.
        return None

    target_total = 0
    target_changed = 0
    target_peer_adopted = 0
    target_lock_total = 0
    target_locked = 0

    all_agent_total = 0
    all_agent_changed = 0
    all_agent_peer_adopted = 0

    consensus_total = 0
    consensus_prev = 0
    consensus_curr = 0

    usable_samples = 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            rounds = extract_rounds(sample)
            prev_key = str(tr - 1)
            curr_key = str(tr)
            if prev_key not in rounds or curr_key not in rounds:
                continue

            prev_answers = parse_final_answers(rounds[prev_key])
            curr_answers = parse_final_answers(rounds[curr_key])
            n = min(len(prev_answers), len(curr_answers))
            if n == 0:
                continue

            prev_answers = prev_answers[:n]
            curr_answers = curr_answers[:n]

            usable_samples += 1

            if target_agent_idx < n:
                target_total += 1
                prev_t = prev_answers[target_agent_idx]
                curr_t = curr_answers[target_agent_idx]
                changed = curr_t != prev_t
                if changed:
                    target_changed += 1
                    peer_prev = {
                        prev_answers[j]
                        for j in range(n)
                        if j != target_agent_idx and prev_answers[j] != ""
                    }
                    if curr_t != "" and curr_t in peer_prev:
                        target_peer_adopted += 1

                future_round_ids: List[int] = []
                for key in rounds.keys():
                    if key.isdigit():
                        idx = int(key)
                        if idx > tr:
                            future_round_ids.append(idx)
                future_round_ids.sort()

                locked = True
                for round_idx in future_round_ids:
                    future_answers = parse_final_answers(rounds.get(str(round_idx), {}))
                    if len(future_answers) <= target_agent_idx:
                        continue
                    if future_answers[target_agent_idx] != curr_t:
                        locked = False
                        break
                target_lock_total += 1
                if locked:
                    target_locked += 1

            for i in range(n):
                all_agent_total += 1
                changed_i = curr_answers[i] != prev_answers[i]
                if changed_i:
                    all_agent_changed += 1
                    peer_prev_i = {
                        prev_answers[j] for j in range(n) if j != i and prev_answers[j] != ""
                    }
                    if curr_answers[i] != "" and curr_answers[i] in peer_prev_i:
                        all_agent_peer_adopted += 1

            consensus_total += 1
            if is_consensus(prev_answers):
                consensus_prev += 1
            if is_consensus(curr_answers):
                consensus_curr += 1

    if usable_samples < min_samples:
        return None

    target_change_rate = safe_div(target_changed, target_total)
    target_stay_rate = None if target_change_rate is None else (1.0 - target_change_rate)
    target_peer_adoption_given_change = safe_div(target_peer_adopted, target_changed)
    target_peer_adoption_rate = safe_div(target_peer_adopted, target_total)
    target_lock_rate = safe_div(target_locked, target_lock_total)

    all_agent_change_rate = safe_div(all_agent_changed, all_agent_total)
    all_agent_peer_adoption_given_change = safe_div(all_agent_peer_adopted, all_agent_changed)
    all_agent_peer_adoption_rate = safe_div(all_agent_peer_adopted, all_agent_total)

    consensus_prev_rate = safe_div(consensus_prev, consensus_total)
    consensus_curr_rate = safe_div(consensus_curr, consensus_total)
    consensus_gain = None
    if consensus_prev_rate is not None and consensus_curr_rate is not None:
        consensus_gain = consensus_curr_rate - consensus_prev_rate

    return {
        **meta,
        "target_agent_idx": target_agent_idx,
        "usable_samples": usable_samples,
        "target_total": target_total,
        "target_changed": target_changed,
        "target_peer_adopted": target_peer_adopted,
        "target_lock_total": target_lock_total,
        "target_locked": target_locked,
        "all_agent_total": all_agent_total,
        "all_agent_changed": all_agent_changed,
        "all_agent_peer_adopted": all_agent_peer_adopted,
        "consensus_total": consensus_total,
        "consensus_prev": consensus_prev,
        "consensus_curr": consensus_curr,
        "target_agent_change_rate": target_change_rate,
        "target_agent_stay_rate": target_stay_rate,
        "target_agent_peer_adoption_rate": target_peer_adoption_rate,
        "target_agent_peer_adoption_given_change": target_peer_adoption_given_change,
        "target_agent_lock_rate": target_lock_rate,
        "all_agent_change_rate": all_agent_change_rate,
        "all_agent_peer_adoption_rate": all_agent_peer_adoption_rate,
        "all_agent_peer_adoption_given_change": all_agent_peer_adoption_given_change,
        "consensus_prev_rate": consensus_prev_rate,
        "consensus_curr_rate": consensus_curr_rate,
        "consensus_gain": consensus_gain,
    }


def linear_slope(records: List[Dict[str, Any]], metric: str) -> Optional[float]:
    points: List[Tuple[float, float]] = []
    for record in records:
        value = record.get(metric)
        if value is None:
            continue
        points.append((record["temperature"], value))

    if len(points) < 2:
        return None

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    var_x = np.var(xs)
    if var_x == 0:
        return None
    cov_xy = np.cov(xs, ys, ddof=0)[0, 1]
    return float(cov_xy / var_x)


def mean_metric(records: Iterable[Dict[str, Any]], metric: str) -> Optional[float]:
    values = [record.get(metric) for record in records]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def hypothesis_status(truths: List[Optional[bool]]) -> str:
    valid = [x for x in truths if x is not None]
    if not valid:
        return "insufficient_data"
    true_count = sum(1 for x in valid if x)
    if true_count == len(valid):
        return "supported"
    if true_count > 0:
        return "partially_supported"
    return "not_supported"


def summarize_group(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    key_record = records[0]
    sorted_records = sorted(records, key=lambda x: x["temperature"])

    low_records = [r for r in sorted_records if r["temperature"] < 1.0]
    high_records = [r for r in sorted_records if r["temperature"] > 1.0]
    baseline_records = [r for r in sorted_records if abs(r["temperature"] - 1.0) < 1e-9]

    low_change = mean_metric(low_records, "target_agent_change_rate")
    high_change = mean_metric(high_records, "target_agent_change_rate")
    low_peer = mean_metric(low_records, "target_agent_peer_adoption_rate")
    high_peer = mean_metric(high_records, "target_agent_peer_adoption_rate")

    low_stay = mean_metric(low_records, "target_agent_stay_rate")
    high_stay = mean_metric(high_records, "target_agent_stay_rate")
    low_lock = mean_metric(low_records, "target_agent_lock_rate")
    high_lock = mean_metric(high_records, "target_agent_lock_rate")
    low_consensus_gain = mean_metric(low_records, "consensus_gain")
    high_consensus_gain = mean_metric(high_records, "consensus_gain")

    h1_change = None if (low_change is None or high_change is None) else (high_change > low_change)
    h1_peer = None if (low_peer is None or high_peer is None) else (high_peer > low_peer)

    h2_stay = None if (low_stay is None or high_stay is None) else (low_stay > high_stay)
    h2_lock = None if (low_lock is None or high_lock is None) else (low_lock > high_lock)
    h2_consensus = None
    if low_consensus_gain is not None and high_consensus_gain is not None:
        h2_consensus = low_consensus_gain > high_consensus_gain

    temperature_grid = [r["temperature"] for r in sorted_records]

    return {
        "dataset": key_record["dataset"],
        "model": key_record["model"],
        "data_size": key_record["data_size"],
        "target_round": key_record["target_round"],
        "max_new_tokens": key_record["max_new_tokens"],
        "target_agent_idx": key_record["target_agent_idx"],
        "num_temperature_points": len(sorted_records),
        "temperature_grid": temperature_grid,
        "num_low_temps": len(low_records),
        "num_high_temps": len(high_records),
        "num_baseline_temps": len(baseline_records),
        "mean_target_change_low": low_change,
        "mean_target_change_high": high_change,
        "mean_target_peer_adoption_low": low_peer,
        "mean_target_peer_adoption_high": high_peer,
        "mean_target_stay_low": low_stay,
        "mean_target_stay_high": high_stay,
        "mean_target_lock_low": low_lock,
        "mean_target_lock_high": high_lock,
        "mean_consensus_gain_low": low_consensus_gain,
        "mean_consensus_gain_high": high_consensus_gain,
        "slope_temp_vs_target_change": linear_slope(sorted_records, "target_agent_change_rate"),
        "slope_temp_vs_target_peer_adoption": linear_slope(sorted_records, "target_agent_peer_adoption_rate"),
        "slope_temp_vs_target_stay": linear_slope(sorted_records, "target_agent_stay_rate"),
        "slope_temp_vs_target_lock": linear_slope(sorted_records, "target_agent_lock_rate"),
        "slope_temp_vs_consensus_gain": linear_slope(sorted_records, "consensus_gain"),
        "h1_high_temp_more_change": h1_change,
        "h1_high_temp_more_peer_adoption": h1_peer,
        "h1_status": hypothesis_status([h1_change, h1_peer]),
        "h2_low_temp_more_stay": h2_stay,
        "h2_low_temp_more_lock": h2_lock,
        "h2_low_temp_more_consensus_gain": h2_consensus,
        "h2_status": hypothesis_status([h2_stay, h2_lock, h2_consensus]),
        "source_files": [r["source_file"] for r in sorted_records],
    }


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    all_files = glob.glob(os.path.join(args.history_dir, "**", "*.jsonl"), recursive=True)

    detail_rows: List[Dict[str, Any]] = []
    for path in sorted(all_files):
        if "previous_backup" in path:
            continue
        meta = parse_experiment_meta(path)
        if meta is None:
            continue
        if args.dataset and meta["dataset"] != args.dataset:
            continue

        row = compute_file_metrics(
            path=path,
            meta=meta,
            target_agent_idx=args.target_agent_idx,
            min_samples=args.min_samples,
        )
        if row is not None:
            detail_rows.append(row)

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in detail_rows:
        group_key = (
            row["dataset"],
            row["model"],
            row["data_size"],
            row["target_round"],
            row["max_new_tokens"],
            row["target_agent_idx"],
        )
        grouped.setdefault(group_key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for _, records in sorted(grouped.items()):
        summary_rows.append(summarize_group(records))

    write_jsonl(args.detail_output, detail_rows)
    write_jsonl(args.summary_output, summary_rows)

    print(f"Detail rows: {len(detail_rows)} -> {args.detail_output}")
    print(f"Group summaries: {len(summary_rows)} -> {args.summary_output}")

    if not summary_rows:
        print("No valid experiment groups found.")
        return

    for row in summary_rows:
        print(
            " | ".join(
                [
                    f"dataset={row['dataset']}",
                    f"model={row['model']}",
                    f"N={row['data_size']}",
                    f"TR={row['target_round']}",
                    f"MNT={row['max_new_tokens']}",
                    f"H1={row['h1_status']}",
                    f"H2={row['h2_status']}",
                ]
            )
        )


if __name__ == "__main__":
    main()
