"""TG-MAD evaluation: compute all metrics and generate plots."""

import argparse
import os
import json
import logging
import time

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
from tg_mad.data_loader import (
    build_samples_from_history,
    load_existing_data,
    load_gsm8k_questions,
    load_split_info,
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


def parse_args():
    parser = argparse.ArgumentParser(description="TG-MAD Evaluation")
    parser.add_argument("--debater_base_url", type=str, default=None)
    parser.add_argument("--debater_model", type=str, default=None)
    parser.add_argument("--prompt_history", type=str, default=None)
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
        action="store_true",
        help="Save per-sample debate text to JSONL files under out/history.",
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
    return parser.parse_args()


def build_eval_config(args, artifact_paths, prompt_history_path, text_history_paths):
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
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
        "existing_data": args.existing_data,
        "train_existing_data": args.train_existing_data,
        "eval_existing_data": args.eval_existing_data,
        "allow_failed_generations": args.allow_failed_generations,
    }


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
    final_round_key = str(n_rounds)
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
            "standard_mad_round": n_rounds,
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

    eval_history_path = args.eval_existing_data
    if eval_history_path is None and split_info is not None:
        eval_history_path = split_info.get("eval_existing_data")
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


def compute_baselines(test_samples, *, dataset: str, n_rounds=N_ROUNDS):
    """Compute single-agent, MV, and standard MAD accuracy from existing JSONL data."""
    single_agent_correct = 0
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

        # === Single Agent (Agent 1 at t=0) ===
        t0_answers = existing["0"]["final_answers"]
        agent1_answer = normalize_stored_answer(t0_answers[0] if t0_answers else None, dataset)
        if answer_is_correct(agent1_answer, gt, dataset=dataset):
            single_agent_correct += 1

        # === MV at t=0 ===
        mv = normalize_stored_answer(existing["0"].get("debate_answer"), dataset)
        mv_is_correct = answer_is_correct(mv, gt, dataset=dataset)
        if mv_is_correct:
            mv_correct += 1

        # === Standard MAD (final round debate_answer) ===
        final_round = str(n_rounds)
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
        "single_agent_accuracy": single_agent_correct / total if total else 0,
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

    ``optimized_prompt`` can be a single string (shared mode) or a list of
    strings (per-agent mode).
    """
    if logger is None:
        logger = logging.getLogger("tg_mad_eval")

    # Non-gradient Variable(s) for evaluation
    if isinstance(optimized_prompt, list):
        prompt_var = [
            tg.Variable(
                value=p,
                requires_grad=False,
                role_description=f"optimized system prompt for agent {i+1} (evaluation)",
            )
            for i, p in enumerate(optimized_prompt)
        ]
    else:
        prompt_var = tg.Variable(
            value=optimized_prompt,
            requires_grad=False,
            role_description="optimized system prompt for debater agents (evaluation)",
        )

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
        if result["final_correct"]:
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

        if (si + 1) % 50 == 0:
            logger.info(
                f"    Progress: {tgmad_correct}/{si+1} correct "
                f"({tgmad_correct/(si+1):.2%})"
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
    """Generate all 3 evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)
    n_rounds = len(eval_results["round_by_round"]["standard_mad"])

    # 1. Round-by-round accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    rounds = list(range(n_rounds))
    std_accs = [
        eval_results["round_by_round"]["standard_mad"][f"t{t}"] for t in rounds
    ]
    tg_accs = [eval_results["round_by_round"]["tgmad"][f"t{t}"] for t in rounds]

    ax.plot(rounds, std_accs, "o-", label="Standard MAD", color="tab:blue", linewidth=2)
    ax.plot(rounds, tg_accs, "s-", label="TG-MAD", color="tab:orange", linewidth=2)
    ax.set_xlabel("Debate Round", fontsize=12)
    ax.set_ylabel("Mean Agent Accuracy", fontsize=12)
    ax.set_title("Round-by-Round Accuracy: Standard MAD vs TG-MAD", fontsize=13)
    ax.set_xticks(rounds)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "round_accuracy_comparison.png"), dpi=150)
    plt.close(fig)

    # 2. Subversion / correction bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
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
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = ["Single\nAgent", "Majority\nVoting", "Standard\nMAD", "TG-MAD"]
    accs = [
        eval_results["single_agent_accuracy"],
        eval_results["mv_accuracy"],
        eval_results["standard_mad_accuracy"],
        eval_results["tgmad_accuracy"],
    ]
    colors = ["tab:gray", "tab:green", "tab:blue", "tab:orange"]
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
    text_history_paths = resolve_text_history_paths(
        output_dir=args.output_dir,
        existing_data_path=args.eval_existing_data or args.existing_data,
        stage="eval",
        save_text_history=args.save_text_history,
        text_history_dir=args.text_history_dir,
        dataset=args.dataset,
    )
    prompt_history_path = artifact_paths["prompt_history"]
    eval_config = build_eval_config(args, artifact_paths, prompt_history_path, text_history_paths)
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
    save_json(eval_config, artifact_paths["run_config"])

    logger = setup_logging(artifact_paths["output_dir"], name="tg_mad_eval")
    set_seeds(args.seed)
    if args.save_text_history:
        logger.info(
            "Saving eval text history to %s",
            text_history_paths["text_history_file"],
        )

    # Load selected prompt(s) — supports both shared and per-agent formats
    logger.info(f"Loading prompt history from {prompt_history_path}")
    with open(prompt_history_path, "r") as f:
        prompt_history = json.load(f)
    selected_index, selected_entry = resolve_prompt_checkpoint(
        prompt_history,
        args.prompt_index,
    )
    first_entry = prompt_history[0]
    per_agent_mode = "prompts" in selected_entry
    if per_agent_mode:
        optimized_prompt = selected_entry["prompts"]   # list[str]
        initial_prompt = first_entry["prompts"]
        logger.info(
            "Per-agent prompt mode detected (%d prompts) at history index %d",
            len(optimized_prompt),
            selected_index,
        )
        for i, p in enumerate(optimized_prompt):
            logger.info(f"  Agent {i+1} prompt (first 120 chars): {p[:120]}...")
    else:
        optimized_prompt = selected_entry["prompt"]    # str
        initial_prompt = first_entry["prompt"]
        logger.info("Evaluating prompt history index %d", selected_index)
        logger.info(f"Optimized prompt (first 200 chars): {optimized_prompt[:200]}...")
    prompt_reference = {
        "prompt_history_file": prompt_history_path,
        "optimized_prompt_index": selected_index,
        "optimized_prompt_epoch": selected_entry.get("epoch"),
        "optimized_prompt_batch": selected_entry.get("batch"),
    }

    # Load data
    logger.info("Loading data...")
    split_info_path = artifact_paths["split_info"]
    split_info = load_split_info(split_info_path)
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
    logger.info(f"  Single Agent accuracy: {baseline_results['single_agent_accuracy']:.2%}")
    logger.info(f"  MV accuracy: {baseline_results['mv_accuracy']:.2%}")
    logger.info(f"  Standard MAD accuracy: {baseline_results['standard_mad_accuracy']:.2%}")

    # === Evaluate TG-MAD ===
    logger.info("Evaluating TG-MAD with optimized prompt...")
    debater_engine = create_debater_engine(
        model=args.debater_model,
        base_url=args.debater_base_url,
        max_tokens=args.max_new_tokens,
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
    }

    # Save results
    save_json(eval_results, artifact_paths["eval_results"])
    logger.info(f"Results saved to {artifact_paths['eval_results']}")

    # Generate plots
    logger.info("Generating plots...")
    generate_plots(eval_results, output_dir=args.output_dir)
    logger.info("Plots saved.")

    # Print summary
    print("\n" + "=" * 60)
    print("TG-MAD EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Single Agent:  {eval_results['single_agent_accuracy']:.2%}")
    print(f"  Majority Vote: {eval_results['mv_accuracy']:.2%}")
    print(f"  Standard MAD:  {eval_results['standard_mad_accuracy']:.2%}")
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
        print(f"  t={t}: Standard MAD={std:.2%}, TG-MAD={tgm:.2%}")
    print()
    print("Correction/Subversion rates:")
    print(
        f"  Correction:  Standard MAD={eval_results['correction_rate']['standard_mad']:.2%}, "
        f"TG-MAD={eval_results['correction_rate']['tgmad']:.2%}"
    )
    print(
        f"  Subversion:  Standard MAD={eval_results['subversion_rate']['standard_mad']:.2%}, "
        f"TG-MAD={eval_results['subversion_rate']['tgmad']:.2%}"
    )
    print("=" * 60)

    return eval_results


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args)
