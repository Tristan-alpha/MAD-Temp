"""Shared utilities: answer parsing, majority vote, logging, seeding."""

import re
import os
import json
import random
import logging
import collections
from time import strftime
from typing import List, Optional, Any, Dict

import numpy as np
import torch

from tg_mad.task_spec import answer_is_correct as task_answer_is_correct
from tg_mad.task_spec import parse_prediction


def parse_answer(response: str, dataset: str = "gsm8k") -> Optional[Any]:
    """Extract a final answer using the legacy task parser for ``dataset``."""
    return parse_prediction(response, dataset)


def majority_vote(answers: List[Optional[Any]]) -> Optional[Any]:
    """Compute majority vote from parsed answers.

    Handles ties by random choice. Returns None if all answers are empty.
    """
    valid = [a for a in answers if a not in (None, "")]
    if not valid:
        return None
    counter = collections.Counter(valid)
    max_count = max(counter.values())
    most_common = [key for key, value in counter.items() if value == max_count]
    return random.choice(most_common)


def answer_is_correct(predicted: Optional[Any], ground_truth, dataset: str = "gsm8k") -> bool:
    """Compare predicted answer to ground truth for the selected dataset."""
    return task_answer_is_correct(predicted, ground_truth, dataset)


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: Any, path: str):
    """Save data to JSON file with numpy serialization support."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_convert_numpy)


def resolve_artifact_paths(
    output_dir: str,
    prompt_history_file: Optional[str] = None,
    eval_results_file: Optional[str] = None,
    split_info_file: Optional[str] = None,
    run_config_file: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve artifact paths relative to the selected output directory."""
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    def _resolve(path: Optional[str], filename: str) -> str:
        target = path if path is not None else os.path.join(abs_output_dir, filename)
        return os.path.abspath(target)

    return {
        "output_dir": abs_output_dir,
        "prompt_history": _resolve(prompt_history_file, "prompt_history.json"),
        "eval_results": _resolve(eval_results_file, "eval_results.json"),
        "split_info": _resolve(split_info_file, "split_info.json"),
        "run_config": _resolve(run_config_file, "run_config.json"),
    }


def infer_dataset_name(existing_data_path: str) -> str:
    """Infer the dataset name from an out/history/<dataset>/... path."""
    path_parts = os.path.abspath(existing_data_path).split(os.sep)
    for idx in range(len(path_parts) - 2):
        if path_parts[idx] == "out" and path_parts[idx + 1] == "history":
            return path_parts[idx + 2]
    return "unknown_dataset"


def resolve_text_history_paths(
    output_dir: str,
    existing_data_path: Optional[str],
    stage: str,
    save_text_history: bool = False,
    text_history_dir: Optional[str] = None,
    dataset: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Resolve optional text-history paths for TG-MAD train/eval runs."""
    if not save_text_history:
        return {
            "enabled": False,
            "dataset": None,
            "run_name": None,
            "text_history_dir": None,
            "text_history_file": None,
        }
    if stage not in {"train", "eval"}:
        raise ValueError(f"Unsupported text-history stage: {stage}")

    abs_output_dir = os.path.abspath(output_dir)
    run_name = os.path.basename(os.path.normpath(abs_output_dir))
    if dataset is None:
        dataset = infer_dataset_name(existing_data_path or "")

    if text_history_dir is None:
        resolved_dir = os.path.join("out", "history", dataset, "tg_mad_text", run_name)
    else:
        resolved_dir = text_history_dir
    abs_history_dir = os.path.abspath(resolved_dir)

    return {
        "enabled": True,
        "dataset": dataset,
        "run_name": run_name,
        "text_history_dir": abs_history_dir,
        "text_history_file": os.path.join(abs_history_dir, f"{stage}_text_history.jsonl"),
    }


def append_jsonl_record(path: str, record: Dict[str, Any]):
    """Append one JSON record to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=_convert_numpy) + "\n")


def init_text_history_file(path: str, manifest: Dict[str, Any]):
    """Create a new JSONL history file and seed it with a manifest record."""
    if os.path.exists(path):
        raise FileExistsError(
            f"Text history file already exists: {path}. "
            "Choose a different output_dir or text_history_dir."
        )
    append_jsonl_record(path, manifest)


def serialize_rounds_for_history(rounds: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Normalize round payloads into JSON-friendly dictionaries."""
    serialized = {}
    for round_idx in sorted(rounds):
        round_data = rounds[round_idx]
        serialized[str(round_idx)] = {
            "answers": list(round_data.get("answers", [])),
            "parsed": list(round_data.get("parsed", [])),
            "majority_vote": round_data.get("majority_vote"),
            "individual_correct": list(round_data.get("individual_correct", [])),
        }
    return serialized


def render_transcript_text(question: str, rounds: Dict[int, Dict[str, Any]]) -> str:
    """Render a readable transcript snapshot for later inspection."""
    lines = ["Question:", question]
    for round_idx in sorted(rounds):
        round_title = "Round 0 (independent answers)" if round_idx == 0 else f"Round {round_idx} (debate)"
        round_data = rounds[round_idx]
        lines.extend(["", round_title])
        for agent_idx, answer in enumerate(round_data.get("answers", []), start=1):
            lines.append(f"Agent {agent_idx}: {answer}")
        lines.append(f"Parsed answers: {round_data.get('parsed', [])}")
        lines.append(f"Majority vote: {round_data.get('majority_vote')}")
    return "\n".join(lines)


def build_text_history_manifest(
    *,
    schema_version: int,
    stage: str,
    text_history_paths: Dict[str, Optional[str]],
    output_dir: str,
    prompt_history_file: str,
    run_config_file: str,
    run_config: Dict[str, Any],
    split_info_file: Optional[str] = None,
    results_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the manifest record written to the top of each text-history file."""
    return {
        "record_type": "manifest",
        "artifact_type": "tg_mad_text_history",
        "schema_version": schema_version,
        "stage": stage,
        "dataset": text_history_paths["dataset"],
        "run_name": text_history_paths["run_name"],
        "created_at": strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": output_dir,
        "prompt_history_file": prompt_history_file,
        "run_config_file": run_config_file,
        "split_info_file": split_info_file,
        "results_file": results_file,
        "text_history_dir": text_history_paths["text_history_dir"],
        "text_history_file": text_history_paths["text_history_file"],
        "run_config": run_config,
    }


def _convert_numpy(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def setup_logging(output_dir: str, name: str = "tg_mad") -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # TextGrad configures its own logger at INFO and emits one line per model
    # forward pass, which overwhelms train/eval logs without adding much signal.
    textgrad_logger = logging.getLogger("textgrad")
    textgrad_logger.setLevel(logging.WARNING)
    textgrad_logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(output_dir, f"{name}.log"))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
