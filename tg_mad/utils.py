"""Shared utilities: answer parsing, majority vote, logging, seeding."""

import re
import os
import json
import random
import logging
import collections
from typing import List, Optional, Any, Dict

import numpy as np
import torch

from tg_mad.config import ANSWER_REGEX


def parse_answer(response: str) -> Optional[float]:
    """Extract numeric answer from response using {final answer: X} format.

    Mirrors src/evaluator.py evaluate_arithmetics (lines 49-53).
    """
    try:
        pred = re.findall(ANSWER_REGEX, response)[-1]
        pred = float(pred.replace("final answer:", "").strip())
        return np.round(pred, 1)
    except (IndexError, ValueError):
        return None


def majority_vote(answers: List[Optional[float]]) -> Optional[float]:
    """Compute majority vote from parsed answers.

    Handles ties by random choice. Returns None if all answers are None.
    Mirrors src/evaluator.py evaluate_arithmetics (lines 61-64).
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counter = collections.Counter(valid)
    max_count = max(counter.values())
    most_common = [key for key, value in counter.items() if value == max_count]
    return random.choice(most_common)


def answer_is_correct(predicted: Optional[float], ground_truth) -> bool:
    """Compare predicted answer to ground truth with tolerance."""
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(np.round(ground_truth, 1))) < 0.01
    except (ValueError, TypeError):
        return False


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
