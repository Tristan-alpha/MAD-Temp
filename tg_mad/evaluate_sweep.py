"""Evaluate multiple TG-MAD prompt checkpoints and plot accuracy over updates."""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tg_mad.config import DEBATER_BASE_URL, EXISTING_DATA_PATH, MAX_NEW_TOKENS, N_AGENTS, N_ROUNDS, OUTPUT_DIR
from tg_mad.engine import create_debater_engine
from tg_mad.experiment_profiles import (
    apply_argparse_profile_defaults,
    build_profile_metadata,
)
from tg_mad.evaluate import (
    _load_eval_samples,
    compute_baselines,
    evaluate_tgmad,
    require_per_agent_prompt_entry,
    resolve_prompt_checkpoint,
)
from tg_mad.data_loader import load_split_info
from tg_mad.utils import save_json, set_seeds, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="TG-MAD checkpoint-sweep evaluation")
    parser.add_argument("--debater_base_url", type=str, default=None)
    parser.add_argument("--debater_model", type=str, default=None)
    parser.add_argument("--prompt_history", type=str, default=None)
    parser.add_argument("--experiment_profile", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--results_file", type=str, default=None)
    parser.add_argument("--run_config_file", type=str, default=None)
    parser.add_argument("--plot_file", type=str, default=None)
    parser.add_argument("--split_info_file", type=str, default=None)
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
        help="Accepted for CLI compatibility; checkpoint sweeps currently do not save per-sample text history.",
    )
    parser.add_argument(
        "--no_save_text_history",
        dest="save_text_history",
        action="store_false",
        help="Accepted for CLI compatibility; checkpoint sweeps currently do not save per-sample text history.",
    )
    parser.add_argument(
        "--text_history_dir",
        type=str,
        default=None,
        help="Accepted for CLI compatibility; unused by checkpoint sweeps.",
    )
    parser.add_argument("--n_agents", type=int, default=N_AGENTS)
    parser.add_argument("--n_rounds", type=int, default=N_ROUNDS)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument(
        "--checkpoint_stride",
        type=int,
        default=20,
        help="Evaluate prompt-history entries every N updates, plus the initial and final prompt.",
    )
    parser.add_argument(
        "--checkpoint_indices",
        type=str,
        default=None,
        help="Optional comma-separated prompt-history indices to evaluate instead of checkpoint_stride.",
    )
    parser.add_argument(
        "--allow_failed_generations",
        action="store_true",
        help="Continue with placeholder text if the debater backend fails.",
    )
    args = parser.parse_args()
    apply_argparse_profile_defaults(args, parser, stage="eval_sweep")
    return args


def _resolve_sweep_indices(prompt_history, stride, explicit_indices):
    if explicit_indices:
        resolved = []
        for token in explicit_indices.split(","):
            token = token.strip()
            if not token:
                continue
            idx, _ = resolve_prompt_checkpoint(prompt_history, int(token))
            resolved.append(idx)
        return sorted(set(resolved))

    if stride < 1:
        raise ValueError("--checkpoint_stride must be at least 1")

    last_index = len(prompt_history) - 1
    indices = [0]
    indices.extend(range(stride, last_index + 1, stride))
    if indices[-1] != last_index:
        indices.append(last_index)
    return sorted(set(indices))


def _resolve_paths(args):
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return {
        "output_dir": output_dir,
        "prompt_history": os.path.abspath(
            args.prompt_history or os.path.join(output_dir, "prompt_history.json")
        ),
        "results_file": os.path.abspath(
            args.results_file or os.path.join(output_dir, "checkpoint_sweep_results.json")
        ),
        "run_config_file": os.path.abspath(
            args.run_config_file
            or os.path.join(output_dir, "checkpoint_sweep_run_config.json")
        ),
        "split_info_file": os.path.abspath(
            args.split_info_file or os.path.join(output_dir, "split_info.json")
        ),
        "plot_file": os.path.abspath(
            args.plot_file or os.path.join(output_dir, "checkpoint_sweep_accuracy.png")
        ),
    }


def _load_test_samples(args, split_info_path):
    split_info = load_split_info(split_info_path)
    test_samples = _load_eval_samples(args, split_info)

    if args.max_test_samples is not None:
        if args.max_test_samples < 1:
            raise ValueError("--max_test_samples must be at least 1")
        test_samples = test_samples[: args.max_test_samples]
    return test_samples


def _plot_sweep(summary, plot_path):
    checkpoints = summary["checkpoints"]
    x = [item["prompt_index"] for item in checkpoints]
    y = [item["tgmad_accuracy"] for item in checkpoints]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, marker="o", linewidth=2, label="TG-MAD")
    ax.axhline(
        summary["single_agent_accuracy"],
        linestyle="--",
        linewidth=1.5,
        label="Single Agent (t0 mean)",
    )
    ax.axhline(summary["mv_accuracy"], linestyle="--", linewidth=1.5, label="Majority Vote")
    ax.axhline(summary["standard_mad_accuracy"], linestyle="--", linewidth=1.5, label="Standard MAD")
    ax.set_xlabel("Prompt history index")
    ax.set_ylabel("Accuracy")
    ax.set_title("TG-MAD Accuracy vs Prompt Update Step")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def evaluate_sweep(args):
    if args.n_agents < 1:
        raise ValueError("--n_agents must be at least 1")
    if args.n_rounds < 0:
        raise ValueError("--n_rounds must be non-negative")

    paths = _resolve_paths(args)
    run_config = {
        "experiment_profile": args.experiment_profile,
        "output_dir": paths["output_dir"],
        "prompt_history_file": paths["prompt_history"],
        "results_file": paths["results_file"],
        "plot_file": paths["plot_file"],
        "split_info_file": paths["split_info_file"],
        "debater_base_url": args.debater_base_url or DEBATER_BASE_URL,
        "debater_model": args.debater_model,
        "n_agents": args.n_agents,
        "n_rounds": args.n_rounds,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "max_test_samples": args.max_test_samples,
        "checkpoint_stride": args.checkpoint_stride,
        "checkpoint_indices": args.checkpoint_indices,
        "data_dir": args.data_dir,
        "dataset": args.dataset,
        "existing_data": args.eval_existing_data or args.existing_data,
        "train_existing_data": args.train_existing_data,
        "eval_existing_data": args.eval_existing_data,
        "allow_failed_generations": args.allow_failed_generations,
    }
    run_config.update(build_profile_metadata(args.experiment_profile))
    save_json(run_config, paths["run_config_file"])

    logger = setup_logging(paths["output_dir"], name="tg_mad_eval_sweep")
    set_seeds(args.seed)

    logger.info("Loading prompt history from %s", paths["prompt_history"])
    with open(paths["prompt_history"], "r") as f:
        prompt_history = json.load(f)
    sweep_indices = _resolve_sweep_indices(
        prompt_history,
        args.checkpoint_stride,
        args.checkpoint_indices,
    )
    logger.info("Evaluating prompt checkpoints: %s", sweep_indices)

    test_samples = _load_test_samples(args, paths["split_info_file"])
    logger.info("Evaluating %d held-out samples per checkpoint", len(test_samples))

    baseline_results = compute_baselines(
        test_samples,
        dataset=args.dataset,
        n_rounds=args.n_rounds,
    )
    logger.info(
        "Baselines: single_agent_t0_mean=%.2f%% single_agent_agent1_legacy=%.2f%% mv=%.2f%% standard_mad=%.2f%%",
        100 * baseline_results["single_agent_accuracy"],
        100 * baseline_results["single_agent_accuracy_agent1"],
        100 * baseline_results["mv_accuracy"],
        100 * baseline_results["standard_mad_accuracy"],
    )

    debater_engine = create_debater_engine(
        model=args.debater_model,
        base_url=args.debater_base_url,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    checkpoint_results = []
    for prompt_index in sweep_indices:
        _, entry = resolve_prompt_checkpoint(prompt_history, prompt_index)
        optimized_prompt = require_per_agent_prompt_entry(
            entry,
            prompt_history_file=paths["prompt_history"],
            prompt_index=prompt_index,
        )
        logger.info(
            "Evaluating prompt index %d (epoch=%s, batch=%s)",
            prompt_index,
            entry.get("epoch"),
            entry.get("batch"),
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
            text_history_file=None,
            prompt_reference={
                "prompt_history_file": paths["prompt_history"],
                "optimized_prompt_index": prompt_index,
                "optimized_prompt_epoch": entry.get("epoch"),
                "optimized_prompt_batch": entry.get("batch"),
            },
        )
        checkpoint_results.append(
            {
                "prompt_index": prompt_index,
                "epoch": entry.get("epoch"),
                "batch": entry.get("batch"),
                "train_batch_accuracy": entry.get("train_batch_accuracy"),
                "tgmad_accuracy": tgmad_results["tgmad_accuracy"],
                "tgmad_round_by_round": tgmad_results["tgmad_round_by_round"],
                "tgmad_correction_rate": tgmad_results["tgmad_correction_rate"],
                "tgmad_subversion_rate": tgmad_results["tgmad_subversion_rate"],
                "tgmad_maintained_correct_rate": tgmad_results["tgmad_maintained_correct_rate"],
                "tgmad_maintained_wrong_rate": tgmad_results["tgmad_maintained_wrong_rate"],
            }
        )

    summary = {
        "schema_version": prompt_history[0].get("schema_version"),
        "experiment_profile": args.experiment_profile,
        "prompt_history_file": paths["prompt_history"],
        "split_info_file": paths["split_info_file"],
        "num_test_samples": len(test_samples),
        "single_agent_accuracy": baseline_results["single_agent_accuracy"],
        "single_agent_accuracy_agent1": baseline_results["single_agent_accuracy_agent1"],
        "mv_accuracy": baseline_results["mv_accuracy"],
        "standard_mad_accuracy": baseline_results["standard_mad_accuracy"],
        "checkpoint_stride": args.checkpoint_stride,
        "evaluated_prompt_indices": sweep_indices,
        "checkpoints": checkpoint_results,
    }
    summary.update(build_profile_metadata(args.experiment_profile))
    save_json(summary, paths["results_file"])
    _plot_sweep(summary, paths["plot_file"])

    best = max(checkpoint_results, key=lambda item: item["tgmad_accuracy"])
    print("Checkpoint sweep complete")
    print(f"  Baseline Single Agent (t0 mean): {summary['single_agent_accuracy']:.2%}")
    print(
        "  Baseline Single Agent (Agent 1 legacy): "
        f"{summary['single_agent_accuracy_agent1']:.2%}"
    )
    print(f"  Baseline MV accuracy: {summary['mv_accuracy']:.2%}")
    print(f"  Baseline Standard MAD accuracy: {summary['standard_mad_accuracy']:.2%}")
    print(
        f"  Best TG-MAD checkpoint: index={best['prompt_index']} "
        f"(epoch={best['epoch']}, batch={best['batch']}), accuracy={best['tgmad_accuracy']:.2%}"
    )
    print(f"  Results: {paths['results_file']}")
    print(f"  Plot:    {paths['plot_file']}")


if __name__ == "__main__":
    evaluate_sweep(parse_args())
