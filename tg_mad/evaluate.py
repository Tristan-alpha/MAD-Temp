"""TG-MAD evaluation: compute all metrics and generate plots."""

import argparse
import os
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import textgrad as tg

from tg_mad.config import (
    ARTIFACT_SCHEMA_VERSION,
    DEBATER_BASE_URL,
    EVALUATOR_TEMPERATURE,
    EXISTING_DATA_PATH,
    MAX_NEW_TOKENS,
    N_AGENTS,
    N_ROUNDS,
    OUTPUT_DIR,
    TEMPERATURE,
)
from tg_mad.engine import create_debater_engine
from tg_mad.experiment_profiles import (
    apply_argparse_profile_defaults,
    build_profile_metadata,
    build_runtime_tgmad_deviations,
)
from tg_mad.data_loader import (
    build_samples_from_history,
    load_existing_data,
    load_gsm8k_questions,
    load_split_info,
    load_task_questions,
    select_train_test_split,
)
from tg_mad.forward_pass import mad_forward_pass
from tg_mad.task_spec import normalize_stored_answer
from tg_mad.utils import (
    append_jsonl_record,
    majority_vote,
    build_text_history_manifest,
    init_text_history_file,
    render_transcript_text,
    resolve_artifact_paths,
    resolve_text_history_paths,
    save_json,
    serialize_rounds_for_history,
    set_seeds,
    setup_logging,
    answer_is_correct,
)


def _display_model_name(model_name):
    if model_name is None:
        return None
    return model_name.removeprefix("hosted_vllm/")


def parse_args():
    parser = argparse.ArgumentParser(description="TG-MAD Evaluation")
    parser.add_argument("--debater_base_url", type=str, default=None)
    parser.add_argument("--debater_model", type=str, default=None)
    parser.add_argument("--prompt_history", type=str, default=None)
    parser.add_argument("--experiment_profile", type=str, default=None)
    parser.add_argument(
        "--prompt_index",
        type=int,
        default=None,
        help=(
            "Prompt-history index to evaluate. Defaults to the latest checkpoint "
            "when omitted. Negative values follow normal Python indexing."
        ),
    )
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--results_file", type=str, default=None)
    parser.add_argument("--split_info_file", type=str, default=None)
    parser.add_argument("--run_config_file", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--dataset", type=str, default="hh_rlhf")
    parser.add_argument("--existing_data", type=str, default=EXISTING_DATA_PATH)
    parser.add_argument("--train_existing_data", type=str, default=None)
    parser.add_argument("--eval_existing_data", type=str, default=None)
    parser.add_argument(
        "--save_text_history",
        dest="save_text_history",
        action="store_true",
        default=True,
        help="Save per-sample debate text to JSONL files under out/history.",
    )
    parser.add_argument(
        "--no_save_text_history",
        dest="save_text_history",
        action="store_false",
        help="Disable per-sample debate text history saving.",
    )
    parser.add_argument(
        "--text_history_dir",
        type=str,
        default=None,
        help="Optional directory for train/eval text history JSONL files.",
    )
    parser.add_argument("--n_agents", type=int, default=N_AGENTS)
    parser.add_argument("--n_rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N held-out test samples from the saved split.",
    )
    parser.add_argument(
        "--allow_failed_generations",
        action="store_true",
        help="Continue with placeholder text if the debater backend fails.",
    )
    parser.add_argument(
        "--icl_existing_data",
        type=str,
        default=None,
        help="Path to ICL-MAD eval-pool JSONL for side-by-side comparison.",
    )
    args = parser.parse_args()
    apply_argparse_profile_defaults(args, parser, stage="eval")
    return args


def build_eval_config(
    args,
    artifact_paths,
    prompt_history_path,
    text_history_paths,
    resolved_existing_data,
):
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_profile": args.experiment_profile,
        "output_dir": artifact_paths["output_dir"],
        "prompt_history_file": prompt_history_path,
        "prompt_index": args.prompt_index,
        "results_file": artifact_paths["eval_results"],
        "split_info_file": artifact_paths["split_info"],
        "run_config_file": artifact_paths["run_config"],
        "save_text_history": args.save_text_history,
        "text_history_dir": text_history_paths["text_history_dir"],
        "text_history_file": text_history_paths["text_history_file"],
        "debater_base_url": args.debater_base_url or DEBATER_BASE_URL,
        "debater_model": args.debater_model,
        "debater_temperature": TEMPERATURE,
        "evaluator_temperature": EVALUATOR_TEMPERATURE,
        "n_agents": args.n_agents,
        "n_rounds": args.n_rounds,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "max_test_samples": args.max_test_samples,
        "data_dir": args.data_dir,
        "dataset": args.dataset,
        "existing_data": resolved_existing_data,
        "train_existing_data": args.train_existing_data,
        "eval_existing_data": args.eval_existing_data,
        "allow_failed_generations": args.allow_failed_generations,
        "icl_existing_data": getattr(args, "icl_existing_data", None),
    }


def _resolve_eval_history_path(args, split_info):
    eval_history_path = args.eval_existing_data
    if eval_history_path is None and split_info is not None:
        eval_history_path = split_info.get("eval_existing_data")
    return eval_history_path or args.existing_data


def resolve_prompt_checkpoint(prompt_history, prompt_index):
    """Select one prompt-history entry, defaulting to the latest checkpoint."""
    if not prompt_history:
        raise ValueError("Prompt history is empty; cannot evaluate.")

    resolved_index = len(prompt_history) - 1 if prompt_index is None else prompt_index
    if resolved_index < 0:
        resolved_index += len(prompt_history)
    if resolved_index < 0 or resolved_index >= len(prompt_history):
        raise IndexError(
            f"Prompt index {prompt_index} is out of range for history "
            f"with {len(prompt_history)} entries."
        )
    return resolved_index, prompt_history[resolved_index]


def require_per_agent_prompt_entry(prompt_entry, *, prompt_history_file: str, prompt_index: int):
    """Return per-agent prompts from a checkpoint entry or raise a clear error."""
    if "prompts" not in prompt_entry:
        raise ValueError(
            "Shared prompt histories are no longer supported. "
            f"Checkpoint {prompt_index} in {prompt_history_file} does not contain per-agent prompts."
        )
    return prompt_entry["prompts"]


def build_eval_text_history_record(
    *,
    sample,
    result,
    n_rounds: int,
    dataset: str,
    prompt_reference,
):
    existing = sample["existing_data"]
    gt = sample["ground_truth"]
    t0_answers = existing["0"].get("final_answers", [])
    t0_parsed = [normalize_stored_answer(answer, dataset) for answer in t0_answers]
    t0_majority_vote = normalize_stored_answer(existing["0"].get("debate_answer"), dataset)
    final_round_key = _resolve_available_final_round(existing, requested_rounds=n_rounds)
    standard_mad_final = existing.get(final_round_key, {})

    return {
        "record_type": "sample",
        "stage": "eval",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_index": sample["index"],
        "question": sample["question"],
        "ground_truth": gt,
        "optimized_prompt_reference": {
            "prompt_history_file": prompt_reference["prompt_history_file"],
            "optimized_prompt_index": prompt_reference["optimized_prompt_index"],
        },
        "rounds": serialize_rounds_for_history(result["rounds"]),
        "final_majority_vote": result["final_majority_vote"],
        "final_correct": result["final_correct"],
        "transcript_text": render_transcript_text(sample["question"], result["rounds"]),
        "baseline_comparison": {
            "t0_answers": t0_answers,
            "t0_parsed": t0_parsed,
            "t0_majority_vote": t0_majority_vote,
            "t0_majority_correct": answer_is_correct(t0_majority_vote, gt, dataset=dataset),
            "standard_mad_round": int(final_round_key),
            "standard_mad_final_answer": standard_mad_final.get("debate_answer"),
            "standard_mad_final_correct": standard_mad_final.get("debate_answer_iscorr", False),
        },
    }


def _load_eval_samples(args, split_info):
    if split_info is not None and split_info.get("dataset") not in (None, args.dataset):
        raise ValueError(
            f"Dataset mismatch: split info was created for {split_info.get('dataset')} "
            f"but evaluation requested {args.dataset}"
        )

    if args.dataset == "gsm8k" and args.eval_existing_data is None:
        existing_data = load_existing_data(args.existing_data)
        questions, answers = load_gsm8k_questions(args.data_dir, data_size=len(existing_data))

        if split_info is not None and "train_indices" in split_info:
            train_indices_set = set(split_info["train_indices"])
            test_samples = []
            for i in range(len(existing_data)):
                if i not in train_indices_set:
                    test_samples.append(
                        {
                            "question": questions[i],
                            "ground_truth": answers[i],
                            "existing_data": existing_data[i],
                            "index": i,
                        }
                    )
        else:
            _, test_samples, _ = select_train_test_split(existing_data, questions, answers)
        return test_samples

    eval_history_path = _resolve_eval_history_path(args, split_info)
    if eval_history_path is None:
        raise ValueError(
            f"{args.dataset} evaluation requires --eval_existing_data "
            "or split_info.json with eval_existing_data."
        )

    return build_samples_from_history(
        dataset=args.dataset,
        history_path=eval_history_path,
        data_dir=args.data_dir,
        pool="eval",
        seed=args.seed,
    )


# ─── Baseline metrics from existing JSONL ───────────────────────────────────


def _resolve_available_final_round(record: dict, *, requested_rounds: int) -> str:
    """Return the latest available numeric round key up to the requested round.

    Older baseline histories may contain fewer debate rounds than the current
    evaluation configuration. In that case we compare against the latest round
    that actually exists in the stored history instead of crashing.
    """
    available_rounds = sorted(int(key) for key in record if key.isdigit())
    if not available_rounds:
        raise ValueError("History record does not contain any numeric round keys.")

    eligible_rounds = [round_idx for round_idx in available_rounds if round_idx <= requested_rounds]
    if eligible_rounds:
        return str(eligible_rounds[-1])
    return str(available_rounds[-1])


def compute_baselines(test_samples, *, dataset: str, n_rounds=N_ROUNDS):
    """Compute baseline metrics from existing JSONL data.

    Default single-agent accuracy is the mean round-0 agent correctness across all
    agents (sum of correct t=0 answers divided by total t=0 answers). The legacy
    Agent-1-at-t0 metric is still returned as an additive field.
    """
    single_agent_agent1_correct = 0
    single_agent_t0_all_correct = 0
    single_agent_t0_all_total = 0
    mv_correct = 0
    mad_correct = 0
    total = len(test_samples)

    # Round-by-round accuracy for standard MAD
    round_agent_correct = {t: 0 for t in range(n_rounds + 1)}
    round_agent_total = {t: 0 for t in range(n_rounds + 1)}

    # Correction / subversion tracking for standard MAD
    correction = 0
    subversion = 0
    maintained_correct = 0
    maintained_wrong = 0

    for sample in test_samples:
        existing = sample["existing_data"]
        gt = sample["ground_truth"]

        # === Single Agent default: all agents at t=0 ===
        t0_answers = existing["0"].get("final_answers", [])
        for raw_answer in t0_answers:
            parsed_answer = normalize_stored_answer(raw_answer, dataset)
            single_agent_t0_all_total += 1
            if answer_is_correct(parsed_answer, gt, dataset=dataset):
                single_agent_t0_all_correct += 1

        # === Legacy Single Agent (Agent 1 at t=0) ===
        agent1_answer = normalize_stored_answer(t0_answers[0] if t0_answers else None, dataset)
        if answer_is_correct(agent1_answer, gt, dataset=dataset):
            single_agent_agent1_correct += 1

        # === MV at t=0 ===
        mv = normalize_stored_answer(existing["0"].get("debate_answer"), dataset)
        mv_is_correct = answer_is_correct(mv, gt, dataset=dataset)
        if mv_is_correct:
            mv_correct += 1

        # === Standard MAD (final round debate_answer) ===
        final_round = _resolve_available_final_round(existing, requested_rounds=n_rounds)
        mad_is_correct = existing[final_round].get("debate_answer_iscorr", False)
        if mad_is_correct:
            mad_correct += 1

        # === Round-by-round individual agent accuracy ===
        for t in range(n_rounds + 1):
            round_key = str(t)
            if round_key in existing:
                iscorr = existing[round_key].get("final_answer_iscorr", [])
                for is_c in iscorr:
                    round_agent_total[t] += 1
                    if is_c:
                        round_agent_correct[t] += 1

        # === Correction / Subversion ===
        if mv_is_correct and not mad_is_correct:
            subversion += 1
        elif not mv_is_correct and mad_is_correct:
            correction += 1
        elif mv_is_correct and mad_is_correct:
            maintained_correct += 1
        else:
            maintained_wrong += 1

    results = {
        "single_agent_accuracy": (
            single_agent_t0_all_correct / single_agent_t0_all_total
            if single_agent_t0_all_total
            else 0
        ),
        "single_agent_accuracy_agent1": (
            single_agent_agent1_correct / total if total else 0
        ),
        "mv_accuracy": mv_correct / total if total else 0,
        "standard_mad_accuracy": mad_correct / total if total else 0,
        "standard_mad_round_by_round": {
            f"t{t}": round_agent_correct[t] / round_agent_total[t]
            if round_agent_total[t]
            else 0
            for t in range(n_rounds + 1)
        },
        "standard_mad_correction_rate": correction / total if total else 0,
        "standard_mad_subversion_rate": subversion / total if total else 0,
        "standard_mad_maintained_correct_rate": maintained_correct / total if total else 0,
        "standard_mad_maintained_wrong_rate": maintained_wrong / total if total else 0,
    }
    return results


def compute_icl_baselines(
    icl_existing_data: List[dict],
    ground_truths: List,
    *,
    dataset: str,
    n_rounds=N_ROUNDS,
):
    """Compute ICL-MAD accuracy from a separate ICL JSONL history.

    Unlike compute_baselines (which extracts metrics from the same samples),
    this takes raw JSONL records + ground truths and returns ICL-MAD metrics.
    """
    total = len(icl_existing_data)
    if total == 0:
        return None

    mad_correct = 0
    round_agent_correct = {t: 0 for t in range(n_rounds + 1)}
    round_agent_total = {t: 0 for t in range(n_rounds + 1)}
    # Correction/subversion relative to ICL t=0 MV
    correction = 0
    subversion = 0
    maintained_correct = 0
    maintained_wrong = 0

    for record, gt in zip(icl_existing_data, ground_truths):
        # ICL-MAD final accuracy
        final_round = _resolve_available_final_round(record, requested_rounds=n_rounds)
        mad_is_correct = record[final_round].get("debate_answer_iscorr", False)
        if mad_is_correct:
            mad_correct += 1

        # ICL MV at t=0
        mv = normalize_stored_answer(record["0"].get("debate_answer"), dataset)
        mv_is_correct = answer_is_correct(mv, gt, dataset=dataset)

        # Correction/subversion
        if mv_is_correct and not mad_is_correct:
            subversion += 1
        elif not mv_is_correct and mad_is_correct:
            correction += 1
        elif mv_is_correct and mad_is_correct:
            maintained_correct += 1
        else:
            maintained_wrong += 1

        # Round-by-round
        for t in range(n_rounds + 1):
            round_key = str(t)
            if round_key in record:
                iscorr = record[round_key].get("final_answer_iscorr", [])
                for is_c in iscorr:
                    round_agent_total[t] += 1
                    if is_c:
                        round_agent_correct[t] += 1

    return {
        "icl_mad_accuracy": mad_correct / total,
        "icl_mad_round_by_round": {
            f"t{t}": round_agent_correct[t] / round_agent_total[t]
            if round_agent_total[t]
            else 0
            for t in range(n_rounds + 1)
        },
        "icl_mad_correction_rate": correction / total,
        "icl_mad_subversion_rate": subversion / total,
        "icl_mad_maintained_correct_rate": maintained_correct / total,
        "icl_mad_maintained_wrong_rate": maintained_wrong / total,
    }


# ─── TG-MAD evaluation ─────────────────────────────────────────────────────


def evaluate_tgmad(
    test_samples,
    optimized_prompt,
    debater_engine,
    n_agents=N_AGENTS,
    n_rounds=N_ROUNDS,
    dataset: str = "gsm8k",
    logger=None,
    allow_failed_generations: bool = False,
    text_history_file: str = None,
    prompt_reference=None,
):
    """Run debates on test set with optimized prompt and compute metrics.

    ``optimized_prompt`` must be a list of per-agent system prompts.
    """
    if logger is None:
        logger = logging.getLogger("tg_mad_eval")

    prompt_var = [
        tg.Variable(
            value=p,
            requires_grad=False,
            role_description=f"optimized system prompt for agent {i+1} (evaluation)",
        )
        for i, p in enumerate(optimized_prompt)
    ]

    total = len(test_samples)
    tgmad_correct = 0
    correction = 0
    subversion = 0
    maintained_correct = 0
    maintained_wrong = 0

    round_agent_correct = {t: 0 for t in range(n_rounds + 1)}
    round_agent_total = {t: 0 for t in range(n_rounds + 1)}

    for si, sample in enumerate(test_samples):
        gt = sample["ground_truth"]
        logger.info(f"  Evaluating {si+1}/{total}: Q='{sample['question'][:50]}...'")

        result = mad_forward_pass(
            question=sample["question"],
            ground_truth=gt,
            debater_prompt=prompt_var,
            debater_engine=debater_engine,
            n_agents=n_agents,
            n_rounds=n_rounds,
            dataset=dataset,
            allow_failed_generations=allow_failed_generations,
        )

        # TG-MAD accuracy
        sample_correct = bool(result["final_correct"])
        if sample_correct:
            tgmad_correct += 1

        # MV baseline from existing data for correction/subversion
        existing = sample["existing_data"]
        mv = normalize_stored_answer(existing["0"].get("debate_answer"), dataset)
        mv_is_correct = answer_is_correct(mv, gt, dataset=dataset)

        # Correction/subversion relative to frozen MV baseline
        if mv_is_correct and not result["final_correct"]:
            subversion += 1
        elif not mv_is_correct and result["final_correct"]:
            correction += 1
        elif mv_is_correct and result["final_correct"]:
            maintained_correct += 1
        else:
            maintained_wrong += 1

        # Round-by-round
        for t in range(n_rounds + 1):
            rd = result["rounds"].get(t, {})
            for is_c in rd.get("individual_correct", []):
                round_agent_total[t] += 1
                if is_c:
                    round_agent_correct[t] += 1

        if text_history_file is not None:
            append_jsonl_record(
                text_history_file,
                build_eval_text_history_record(
                    sample=sample,
                    result=result,
                    n_rounds=n_rounds,
                    dataset=dataset,
                    prompt_reference=prompt_reference,
                ),
            )

        logger.info(
            "    Completed %s/%s | sample_correct=%s | running_average_accuracy=%.2f%%",
            si + 1,
            total,
            sample_correct,
            100 * tgmad_correct / (si + 1),
        )

    results = {
        "tgmad_accuracy": tgmad_correct / total if total else 0,
        "tgmad_round_by_round": {
            f"t{t}": round_agent_correct[t] / round_agent_total[t]
            if round_agent_total[t]
            else 0
            for t in range(n_rounds + 1)
        },
        "tgmad_correction_rate": correction / total if total else 0,
        "tgmad_subversion_rate": subversion / total if total else 0,
        "tgmad_maintained_correct_rate": maintained_correct / total if total else 0,
        "tgmad_maintained_wrong_rate": maintained_wrong / total if total else 0,
    }
    return results


# ─── Plotting ───────────────────────────────────────────────────────────────


def generate_plots(eval_results, output_dir=OUTPUT_DIR):
    """Generate all 3 evaluation plots. Includes ICL-MAD when present."""
    os.makedirs(output_dir, exist_ok=True)
    n_rounds = len(eval_results["round_by_round"]["standard_mad"])
    has_icl = "icl_mad" in eval_results.get("round_by_round", {})

    # 1. Round-by-round accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    rounds = list(range(n_rounds))
    std_accs = [
        eval_results["round_by_round"]["standard_mad"][f"t{t}"] for t in rounds
    ]
    tg_accs = [eval_results["round_by_round"]["tgmad"][f"t{t}"] for t in rounds]

    ax.plot(rounds, std_accs, "o-", label="Standard MAD", color="tab:blue", linewidth=2)
    if has_icl:
        icl_accs = [eval_results["round_by_round"]["icl_mad"][f"t{t}"] for t in rounds]
        ax.plot(rounds, icl_accs, "D-", label="ICL-MAD", color="tab:purple", linewidth=2)
    ax.plot(rounds, tg_accs, "s-", label="TG-MAD", color="tab:orange", linewidth=2)
    ax.set_xlabel("Debate Round", fontsize=12)
    ax.set_ylabel("Mean Agent Accuracy", fontsize=12)
    title_suffix = " vs ICL-MAD" if has_icl else ""
    ax.set_title(f"Round-by-Round Accuracy: Standard MAD{title_suffix} vs TG-MAD", fontsize=13)
    ax.set_xticks(rounds)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "round_accuracy_comparison.png"), dpi=150)
    plt.close(fig)

    # 2. Subversion / correction bar chart
    fig, ax = plt.subplots(figsize=(10 if has_icl else 8, 5))
    categories = ["Correction", "Subversion", "Maintained\nCorrect", "Maintained\nWrong"]
    std_vals = [
        eval_results["correction_rate"]["standard_mad"],
        eval_results["subversion_rate"]["standard_mad"],
        eval_results["maintained_correct_rate"]["standard_mad"],
        eval_results["maintained_wrong_rate"]["standard_mad"],
    ]
    tg_vals = [
        eval_results["correction_rate"]["tgmad"],
        eval_results["subversion_rate"]["tgmad"],
        eval_results["maintained_correct_rate"]["tgmad"],
        eval_results["maintained_wrong_rate"]["tgmad"],
    ]

    x = np.arange(len(categories))
    if has_icl:
        icl_vals = [
            eval_results["correction_rate"]["icl_mad"],
            eval_results["subversion_rate"]["icl_mad"],
            eval_results["maintained_correct_rate"]["icl_mad"],
            eval_results["maintained_wrong_rate"]["icl_mad"],
        ]
        width = 0.25
        ax.bar(x - width, std_vals, width, label="Standard MAD", color="tab:blue")
        ax.bar(x, icl_vals, width, label="ICL-MAD", color="tab:purple")
        ax.bar(x + width, tg_vals, width, label="TG-MAD", color="tab:orange")
    else:
        width = 0.35
        ax.bar(x - width / 2, std_vals, width, label="Standard MAD", color="tab:blue")
        ax.bar(x + width / 2, tg_vals, width, label="TG-MAD", color="tab:orange")
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title("Correction & Subversion Rates", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "subversion_correction_bar.png"), dpi=150
    )
    plt.close(fig)

    # 3. Overall accuracy bar chart
    fig, ax = plt.subplots(figsize=(8 if has_icl else 7, 5))
    methods = ["Single\nAgent\n(t0 mean)", "Majority\nVoting", "Standard\nMAD"]
    accs = [
        eval_results["single_agent_accuracy"],
        eval_results["mv_accuracy"],
        eval_results["standard_mad_accuracy"],
    ]
    colors = ["tab:gray", "tab:green", "tab:blue"]
    if has_icl:
        methods.append("ICL-MAD")
        accs.append(eval_results["icl_mad_accuracy"])
        colors.append("tab:purple")
    methods.append("TG-MAD")
    accs.append(eval_results["tgmad_accuracy"])
    colors.append("tab:orange")

    bars = ax.bar(methods, accs, color=colors, width=0.6)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Overall Accuracy Comparison", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "overall_accuracy_bar.png"), dpi=150)
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────


def evaluate(args):
    if args.n_agents < 1:
        raise ValueError("--n_agents must be at least 1")
    if args.n_rounds < 0:
        raise ValueError("--n_rounds must be non-negative")
    if args.max_test_samples is not None and args.max_test_samples < 1:
        raise ValueError("--max_test_samples must be at least 1")

    artifact_paths = resolve_artifact_paths(
        output_dir=args.output_dir,
        prompt_history_file=args.prompt_history,
        eval_results_file=args.results_file,
        split_info_file=args.split_info_file,
        run_config_file=(
            args.run_config_file
            if args.run_config_file is not None
            else os.path.join(args.output_dir, "evaluation_run_config.json")
        ),
    )
    split_info_path = artifact_paths["split_info"]
    split_info = load_split_info(split_info_path)
    resolved_eval_history_path = _resolve_eval_history_path(args, split_info)
    text_history_paths = resolve_text_history_paths(
        output_dir=args.output_dir,
        existing_data_path=resolved_eval_history_path,
        stage="eval",
        save_text_history=args.save_text_history,
        text_history_dir=args.text_history_dir,
        dataset=args.dataset,
    )
    prompt_history_path = artifact_paths["prompt_history"]
    eval_config = build_eval_config(
        args,
        artifact_paths,
        prompt_history_path,
        text_history_paths,
        resolved_eval_history_path,
    )

    logger = setup_logging(artifact_paths["output_dir"], name="tg_mad_eval")
    set_seeds(args.seed)

    # Load selected per-agent prompt checkpoint
    logger.info(f"Loading prompt history from {prompt_history_path}")
    with open(prompt_history_path, "r") as f:
        prompt_history = json.load(f)
    selected_index, selected_entry = resolve_prompt_checkpoint(
        prompt_history,
        args.prompt_index,
    )
    first_entry = prompt_history[0]
    optimized_prompt = require_per_agent_prompt_entry(
        selected_entry,
        prompt_history_file=prompt_history_path,
        prompt_index=selected_index,
    )
    initial_prompt = require_per_agent_prompt_entry(
        first_entry,
        prompt_history_file=prompt_history_path,
        prompt_index=0,
    )
    logger.info(
        "Per-agent prompt mode detected (%d prompts) at history index %d",
        len(optimized_prompt),
        selected_index,
    )
    for i, p in enumerate(optimized_prompt):
        logger.info(f"  Agent {i+1} prompt (first 120 chars): {p[:120]}...")
    prompt_reference = {
        "prompt_history_file": prompt_history_path,
        "optimized_prompt_index": selected_index,
        "optimized_prompt_epoch": selected_entry.get("epoch"),
        "optimized_prompt_batch": selected_entry.get("batch"),
    }
    optimizer_model = _display_model_name(
        (first_entry.get("run_config") or {}).get("evaluator_model")
    )
    eval_config.update(
        build_profile_metadata(
            args.experiment_profile,
            include_tgmad_deviations=True,
        )
    )
    eval_config["tgmad_prompt_mode"] = "per_agent_system_prompt"
    eval_config["shared_prompt_mode"] = False
    eval_config["tgmad_optimizer_model"] = optimizer_model
    if "tgmad_deviations_from_paper" in eval_config:
        eval_config["tgmad_deviations_from_paper"] = build_runtime_tgmad_deviations(
            args.experiment_profile,
            prompt_mode=eval_config["tgmad_prompt_mode"],
            optimizer_model=optimizer_model,
            max_new_tokens=args.max_new_tokens,
        )
    if args.save_text_history:
        init_text_history_file(
            text_history_paths["text_history_file"],
            build_text_history_manifest(
                schema_version=ARTIFACT_SCHEMA_VERSION,
                stage="eval",
                text_history_paths=text_history_paths,
                output_dir=artifact_paths["output_dir"],
                prompt_history_file=prompt_history_path,
                run_config_file=artifact_paths["run_config"],
                split_info_file=artifact_paths["split_info"],
                results_file=artifact_paths["eval_results"],
                run_config=eval_config,
            ),
        )
        logger.info(
            "Saving eval text history to %s",
            text_history_paths["text_history_file"],
        )
    save_json(eval_config, artifact_paths["run_config"])

    # Load data
    logger.info("Loading data...")
    test_samples = _load_eval_samples(args, split_info)

    if args.max_test_samples is not None:
        original_test_count = len(test_samples)
        test_samples = test_samples[: args.max_test_samples]
        logger.info(
            "Limiting evaluation to %s/%s held-out samples",
            len(test_samples),
            original_test_count,
        )

    logger.info(f"Evaluating on {len(test_samples)} test samples")

    # === Compute baselines from existing JSONL ===
    logger.info("Computing baselines from existing JSONL...")
    baseline_results = compute_baselines(
        test_samples,
        dataset=args.dataset,
        n_rounds=args.n_rounds,
    )
    logger.info(
        "  Single Agent accuracy (t=0 mean over all agents): %.2f%%",
        100 * baseline_results["single_agent_accuracy"],
    )
    logger.info(
        "  Single Agent accuracy (Agent 1 at t=0, legacy): %.2f%%",
        100 * baseline_results["single_agent_accuracy_agent1"],
    )
    logger.info(f"  MV accuracy: {baseline_results['mv_accuracy']:.2%}")
    logger.info(f"  Standard MAD accuracy: {baseline_results['standard_mad_accuracy']:.2%}")

    # === Compute ICL-MAD baselines (optional) ===
    icl_results = None
    if getattr(args, "icl_existing_data", None) is not None:
        logger.info("Computing ICL-MAD baselines from %s...", args.icl_existing_data)
        icl_data = load_existing_data(args.icl_existing_data)
        _, icl_labels = load_task_questions(
            dataset=args.dataset,
            data_dir=args.data_dir,
            pool="eval",
            data_size=len(icl_data),
            seed=args.seed,
        )
        icl_results = compute_icl_baselines(
            icl_data,
            icl_labels,
            dataset=args.dataset,
            n_rounds=args.n_rounds,
        )
        if icl_results:
            logger.info(f"  ICL-MAD accuracy: {icl_results['icl_mad_accuracy']:.2%}")

    # === Evaluate TG-MAD ===
    logger.info("Evaluating TG-MAD with optimized prompt...")
    debater_engine = create_debater_engine(
        model=args.debater_model,
        base_url=args.debater_base_url,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    tgmad_results = evaluate_tgmad(
        test_samples,
        optimized_prompt,
        debater_engine,
        n_agents=args.n_agents,
        n_rounds=args.n_rounds,
        dataset=args.dataset,
        logger=logger,
        allow_failed_generations=args.allow_failed_generations,
        text_history_file=text_history_paths["text_history_file"],
        prompt_reference=prompt_reference,
    )
    logger.info(f"  TG-MAD accuracy: {tgmad_results['tgmad_accuracy']:.2%}")

    # === Assemble final results ===
    eval_results = {
        "single_agent_accuracy": baseline_results["single_agent_accuracy"],
        "single_agent_accuracy_agent1": baseline_results["single_agent_accuracy_agent1"],
        "mv_accuracy": baseline_results["mv_accuracy"],
        "standard_mad_accuracy": baseline_results["standard_mad_accuracy"],
        "tgmad_accuracy": tgmad_results["tgmad_accuracy"],
        "round_by_round": {
            "standard_mad": baseline_results["standard_mad_round_by_round"],
            "tgmad": tgmad_results["tgmad_round_by_round"],
        },
        "correction_rate": {
            "standard_mad": baseline_results["standard_mad_correction_rate"],
            "tgmad": tgmad_results["tgmad_correction_rate"],
        },
        "subversion_rate": {
            "standard_mad": baseline_results["standard_mad_subversion_rate"],
            "tgmad": tgmad_results["tgmad_subversion_rate"],
        },
        "maintained_correct_rate": {
            "standard_mad": baseline_results["standard_mad_maintained_correct_rate"],
            "tgmad": tgmad_results["tgmad_maintained_correct_rate"],
        },
        "maintained_wrong_rate": {
            "standard_mad": baseline_results["standard_mad_maintained_wrong_rate"],
            "tgmad": tgmad_results["tgmad_maintained_wrong_rate"],
        },
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "evaluation_config": eval_config,
        "num_test_samples": len(test_samples),
        "optimized_prompt": optimized_prompt,
        "initial_prompt": initial_prompt,
        "optimized_prompt_index": selected_index,
        "optimized_prompt_epoch": selected_entry.get("epoch"),
        "optimized_prompt_batch": selected_entry.get("batch"),
        "tgmad_prompt_mode": eval_config["tgmad_prompt_mode"],
        "shared_prompt_mode": eval_config["shared_prompt_mode"],
        "tgmad_optimizer_model": optimizer_model,
    }
    eval_results.update(
        build_profile_metadata(
            args.experiment_profile,
            include_tgmad_deviations=True,
        )
    )
    if "tgmad_deviations_from_paper" in eval_results:
        eval_results["tgmad_deviations_from_paper"] = build_runtime_tgmad_deviations(
            args.experiment_profile,
            prompt_mode=eval_config["tgmad_prompt_mode"],
            optimizer_model=optimizer_model,
            max_new_tokens=args.max_new_tokens,
        )

    # Merge ICL-MAD results if available
    if icl_results is not None:
        eval_results["icl_mad_accuracy"] = icl_results["icl_mad_accuracy"]
        eval_results["round_by_round"]["icl_mad"] = icl_results["icl_mad_round_by_round"]
        eval_results["correction_rate"]["icl_mad"] = icl_results["icl_mad_correction_rate"]
        eval_results["subversion_rate"]["icl_mad"] = icl_results["icl_mad_subversion_rate"]
        eval_results["maintained_correct_rate"]["icl_mad"] = icl_results["icl_mad_maintained_correct_rate"]
        eval_results["maintained_wrong_rate"]["icl_mad"] = icl_results["icl_mad_maintained_wrong_rate"]
        eval_results["icl_existing_data"] = args.icl_existing_data

    # Save results
    canonical_comparison_path = (
        Path(__file__).resolve().parents[1]
        / "out"
        / f"{args.dataset}_all_methods_comparison.json"
    )
    eval_results["canonical_comparison_file"] = str(canonical_comparison_path)
    save_json(eval_results, artifact_paths["eval_results"])
    save_json(eval_results, str(canonical_comparison_path))
    logger.info(f"Results saved to {artifact_paths['eval_results']}")
    logger.info("Canonical comparison saved to %s", canonical_comparison_path)

    # Generate plots
    logger.info("Generating plots...")
    generate_plots(eval_results, output_dir=args.output_dir)
    logger.info("Plots saved.")

    # Print summary
    has_icl = "icl_mad_accuracy" in eval_results
    print("\n" + "=" * 60)
    print("TG-MAD EVALUATION SUMMARY")
    print("=" * 60)
    print(
        "  Single Agent (t0 mean over all agents): "
        f"{eval_results['single_agent_accuracy']:.2%}"
    )
    if "single_agent_accuracy_agent1" in eval_results:
        print(
            "  Single Agent (Agent 1 legacy):      "
            f"{eval_results['single_agent_accuracy_agent1']:.2%}"
        )
    print(f"  Majority Vote: {eval_results['mv_accuracy']:.2%}")
    print(f"  Standard MAD:  {eval_results['standard_mad_accuracy']:.2%}")
    if has_icl:
        print(f"  ICL-MAD:       {eval_results['icl_mad_accuracy']:.2%}")
    print(f"  TG-MAD:        {eval_results['tgmad_accuracy']:.2%}")
    print(
        f"  Total TG-MAD Accuracy on {len(test_samples)} held-out problems: "
        f"{eval_results['tgmad_accuracy']:.2%}"
    )
    print()
    print("Round-by-round (mean agent accuracy):")
    for t in range(args.n_rounds + 1):
        std = eval_results["round_by_round"]["standard_mad"][f"t{t}"]
        tgm = eval_results["round_by_round"]["tgmad"][f"t{t}"]
        line = f"  t={t}: Standard MAD={std:.2%}"
        if has_icl:
            icl = eval_results["round_by_round"]["icl_mad"][f"t{t}"]
            line += f", ICL-MAD={icl:.2%}"
        line += f", TG-MAD={tgm:.2%}"
        print(line)
    print()
    print("Correction/Subversion rates:")
    corr_line = f"  Correction:  Standard MAD={eval_results['correction_rate']['standard_mad']:.2%}"
    subv_line = f"  Subversion:  Standard MAD={eval_results['subversion_rate']['standard_mad']:.2%}"
    if has_icl:
        corr_line += f", ICL-MAD={eval_results['correction_rate']['icl_mad']:.2%}"
        subv_line += f", ICL-MAD={eval_results['subversion_rate']['icl_mad']:.2%}"
    corr_line += f", TG-MAD={eval_results['correction_rate']['tgmad']:.2%}"
    subv_line += f", TG-MAD={eval_results['subversion_rate']['tgmad']:.2%}"
    print(corr_line)
    print(subv_line)
    print("=" * 60)

    return eval_results


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args)
