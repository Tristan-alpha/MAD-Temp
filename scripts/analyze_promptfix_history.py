#!/usr/bin/env python3
"""Analyze prompt-fix MAD history and save summary metrics plus plots."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute round accuracy, correction, and subversion metrics from a legacy MAD history JSONL.",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        required=True,
        help="Path to a legacy MAD history JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="Directory where the summary JSON and plots will be written.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset label for output metadata. Inferred from out/history/<dataset>/... when omitted.",
    )
    parser.add_argument(
        "--method-slug",
        type=str,
        default="promptfix_mad",
        help="Slug used in output filenames.",
    )
    parser.add_argument(
        "--method-label",
        type=str,
        default="Promptfix MAD",
        help="Display label used in plot titles.",
    )
    parser.add_argument(
        "--requested-rounds",
        type=int,
        default=None,
        help="Final debate round to evaluate. Defaults to the latest available round in the file.",
    )
    return parser.parse_args()


def infer_dataset_name(history_path: Path) -> str:
    parts = history_path.resolve().parts
    for idx in range(len(parts) - 2):
        if parts[idx] == "out" and parts[idx + 1] == "history":
            return parts[idx + 2]
    return "unknown_dataset"


def load_records(history_path: Path) -> list[dict]:
    records = []
    with history_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No JSON records found in {history_path}")
    return records


def resolve_available_final_round(record: dict, requested_rounds: int | None) -> str:
    available_rounds = sorted(int(key) for key in record if key.isdigit())
    if not available_rounds:
        raise ValueError("History record does not contain any numeric round keys.")
    if requested_rounds is None:
        return str(available_rounds[-1])

    eligible_rounds = [round_idx for round_idx in available_rounds if round_idx <= requested_rounds]
    if eligible_rounds:
        return str(eligible_rounds[-1])
    return str(available_rounds[-1])


def compute_metrics(records: list[dict], requested_rounds: int | None) -> dict:
    available_rounds = sorted(
        {
            int(key)
            for record in records
            for key in record.keys()
            if key.isdigit()
        }
    )
    if not available_rounds:
        raise ValueError("No numeric round keys found in history records.")

    round_agent_correct = {round_idx: 0 for round_idx in available_rounds}
    round_agent_total = {round_idx: 0 for round_idx in available_rounds}
    round_consensus_correct = {round_idx: 0 for round_idx in available_rounds}
    round_consensus_total = {round_idx: 0 for round_idx in available_rounds}
    final_round_distribution: dict[str, int] = {}

    correction = 0
    subversion = 0
    maintained_correct = 0
    maintained_wrong = 0

    for record in records:
        final_round = resolve_available_final_round(record, requested_rounds)
        final_round_distribution[final_round] = final_round_distribution.get(final_round, 0) + 1

        mv_is_correct = bool(record["0"].get("debate_answer_iscorr", False))
        final_is_correct = bool(record[final_round].get("debate_answer_iscorr", False))

        if mv_is_correct and not final_is_correct:
            subversion += 1
        elif (not mv_is_correct) and final_is_correct:
            correction += 1
        elif mv_is_correct and final_is_correct:
            maintained_correct += 1
        else:
            maintained_wrong += 1

        for round_idx in available_rounds:
            round_key = str(round_idx)
            if round_key not in record:
                continue
            round_data = record[round_key]
            agent_correctness = round_data.get("final_answer_iscorr", [])
            round_agent_total[round_idx] += len(agent_correctness)
            round_agent_correct[round_idx] += sum(1 for flag in agent_correctness if flag)
            round_consensus_total[round_idx] += 1
            if round_data.get("debate_answer_iscorr", False):
                round_consensus_correct[round_idx] += 1

    total = len(records)
    inferred_final_round = str(max(int(round_key) for round_key in final_round_distribution))

    return {
        "num_samples": total,
        "round_agent_accuracy": {
            f"t{round_idx}": (
                round_agent_correct[round_idx] / round_agent_total[round_idx]
                if round_agent_total[round_idx]
                else 0.0
            )
            for round_idx in available_rounds
        },
        "round_consensus_accuracy": {
            f"t{round_idx}": (
                round_consensus_correct[round_idx] / round_consensus_total[round_idx]
                if round_consensus_total[round_idx]
                else 0.0
            )
            for round_idx in available_rounds
        },
        "final_round_distribution": final_round_distribution,
        "inferred_final_round": inferred_final_round,
        "correction_count": correction,
        "correction_rate": correction / total if total else 0.0,
        "subversion_count": subversion,
        "subversion_rate": subversion / total if total else 0.0,
        "maintained_correct_count": maintained_correct,
        "maintained_correct_rate": maintained_correct / total if total else 0.0,
        "maintained_wrong_count": maintained_wrong,
        "maintained_wrong_rate": maintained_wrong / total if total else 0.0,
    }


def save_summary(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")


def plot_round_accuracy(summary: dict, method_label: str, output_path: Path) -> None:
    round_keys = sorted(summary["round_agent_accuracy"].keys(), key=lambda item: int(item[1:]))
    rounds = [int(key[1:]) for key in round_keys]
    agent_acc = [summary["round_agent_accuracy"][key] for key in round_keys]
    consensus_acc = [summary["round_consensus_accuracy"][key] for key in round_keys]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(rounds, agent_acc, marker="o", linewidth=2, color="#2A9D8F", label="Mean Agent Accuracy")
    ax.plot(
        rounds,
        consensus_acc,
        marker="s",
        linewidth=2,
        linestyle="--",
        color="#E76F51",
        label="Consensus Accuracy",
    )

    for round_idx, value in zip(rounds, agent_acc):
        ax.text(round_idx, value + 0.01, f"{value:.1%}", color="#2A9D8F", ha="center", fontsize=9)
    for round_idx, value in zip(rounds, consensus_acc):
        ax.text(round_idx, value - 0.04, f"{value:.1%}", color="#E76F51", ha="center", fontsize=9)

    ax.set_title(f"{method_label} Round Accuracy")
    ax.set_xlabel("Debate Round")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(rounds)
    ymin = max(0.0, min(agent_acc + consensus_acc) - 0.08)
    ymax = min(1.0, max(agent_acc + consensus_acc) + 0.08)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.subplots_adjust(top=0.90, bottom=0.16)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_transition_rates(summary: dict, method_label: str, output_path: Path) -> None:
    categories = [
        "Correction",
        "Subversion",
        "Maintained Correct",
        "Maintained Wrong",
    ]
    rates = [
        summary["correction_rate"],
        summary["subversion_rate"],
        summary["maintained_correct_rate"],
        summary["maintained_wrong_rate"],
    ]
    counts = [
        summary["correction_count"],
        summary["subversion_count"],
        summary["maintained_correct_count"],
        summary["maintained_wrong_count"],
    ]
    colors = ["#2A9D8F", "#C1121F", "#457B9D", "#9AA0A6"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(categories, rates, color=colors, width=0.65)
    for bar, rate, count in zip(bars, rates, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate:.1%}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(f"{method_label} Outcome Transitions")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.subplots_adjust(bottom=0.23, top=0.90)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    history_path = Path(args.history_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset = args.dataset or infer_dataset_name(history_path)

    records = load_records(history_path)
    metrics = compute_metrics(records, requested_rounds=args.requested_rounds)

    prefix = f"{dataset}_{args.method_slug}"
    summary = {
        "schema_version": 1,
        "dataset": dataset,
        "method_slug": args.method_slug,
        "method_label": args.method_label,
        "history_file": str(history_path),
        "output_dir": str(output_dir),
        "requested_rounds": args.requested_rounds,
        **metrics,
        "summary_files": {
            "json": str(output_dir / f"{prefix}_metrics.json"),
            "round_accuracy_plot": str(output_dir / f"{prefix}_round_accuracy.png"),
            "transition_plot": str(output_dir / f"{prefix}_transition_rates.png"),
        },
    }

    save_summary(summary, output_dir / f"{prefix}_metrics.json")
    plot_round_accuracy(summary, args.method_label, output_dir / f"{prefix}_round_accuracy.png")
    plot_transition_rates(summary, args.method_label, output_dir / f"{prefix}_transition_rates.png")

    print(f"Saved summary JSON to {output_dir / f'{prefix}_metrics.json'}")
    print(f"Saved round accuracy plot to {output_dir / f'{prefix}_round_accuracy.png'}")
    print(f"Saved transition plot to {output_dir / f'{prefix}_transition_rates.png'}")
    print()
    print("Promptfix MAD summary:")
    print(f"  Samples: {summary['num_samples']}")
    for round_key, accuracy in summary["round_agent_accuracy"].items():
        consensus = summary["round_consensus_accuracy"][round_key]
        print(f"  {round_key}: mean_agent_acc={accuracy:.2%}, consensus_acc={consensus:.2%}")
    print(f"  Correction: {summary['correction_count']}/{summary['num_samples']} ({summary['correction_rate']:.2%})")
    print(f"  Subversion: {summary['subversion_count']}/{summary['num_samples']} ({summary['subversion_rate']:.2%})")


if __name__ == "__main__":
    main()
