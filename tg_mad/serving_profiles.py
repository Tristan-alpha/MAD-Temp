"""Serving-profile defaults for local vLLM bring-up.

These profiles intentionally capture cluster/workstation stability knobs rather
than experiment logic. The goal is to keep train/eval launch scripts short while
still allowing the user to opt into known-good local serving layouts.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, Optional


SERVING_PROFILES: Dict[str, Dict[str, object]] = {
    "local_30b_dual_gpu": {
        "description": (
            "Run a 30B local evaluator with tensor parallel size 2 on two GPUs, "
            "plus a separate debater GPU. Best fit for shared 40-48 GB cards."
        ),
        "env_defaults": {
            "train": {
                "DEBATER_AUTO_PICK_GPU": "1",
                "EVALUATOR_AUTO_PICK_GPU": "1",
                "EVALUATOR_AUTO_PICK_BEFORE_DEBATER": "1",
                "EVALUATOR_AUTO_PICK_PREFER_TOPOLOGY": "1",
                "DEBATER_AUTO_PICK_PREFERENCE": "highest",
                "DEBATER_GPU_MEMORY": "0.35",
                "EVALUATOR_GPU_MEMORY": "0.80",
                "DEBATER_MAX_MODEL_LEN": "16384",
                "EVALUATOR_MAX_MODEL_LEN": "24576",
                "DEBATER_MAX_NUM_SEQS": "1",
                "EVALUATOR_MAX_NUM_SEQS": "1",
                "EVALUATOR_TENSOR_PARALLEL_SIZE": "2",
                "DEBATER_MIN_FREE_MIB": "12000",
                "EVALUATOR_MIN_FREE_MIB": "30000",
                "MAX_WAIT_SECONDS": "1200",
            },
            "eval": {
                "DEBATER_AUTO_PICK_GPU": "1",
                "DEBATER_AUTO_PICK_PREFERENCE": "highest",
                "DEBATER_MAX_MODEL_LEN": "16384",
                "DEBATER_MAX_NUM_SEQS": "1",
                "DEBATER_MIN_FREE_MIB": "12000",
                "MAX_WAIT_SECONDS": "1200",
            },
        },
    },
    "local_30b_single_80g": {
        "description": (
            "Run a 30B local evaluator on one 80 GB GPU and keep the debater on "
            "a separate GPU. Good fit for A100 80GB-style nodes."
        ),
        "env_defaults": {
            "train": {
                "DEBATER_AUTO_PICK_GPU": "1",
                "EVALUATOR_AUTO_PICK_GPU": "1",
                "EVALUATOR_AUTO_PICK_BEFORE_DEBATER": "1",
                "DEBATER_AUTO_PICK_PREFERENCE": "highest",
                "DEBATER_GPU_MEMORY": "0.35",
                "EVALUATOR_GPU_MEMORY": "0.80",
                "DEBATER_MAX_MODEL_LEN": "16384",
                "EVALUATOR_MAX_MODEL_LEN": "24576",
                "DEBATER_MAX_NUM_SEQS": "1",
                "EVALUATOR_MAX_NUM_SEQS": "1",
                "EVALUATOR_TENSOR_PARALLEL_SIZE": "1",
                "DEBATER_MIN_FREE_MIB": "12000",
                "EVALUATOR_MIN_FREE_MIB": "30000",
                "MAX_WAIT_SECONDS": "1200",
            },
            "eval": {
                "DEBATER_AUTO_PICK_GPU": "1",
                "DEBATER_AUTO_PICK_PREFERENCE": "highest",
                "DEBATER_MAX_MODEL_LEN": "16384",
                "DEBATER_MAX_NUM_SEQS": "1",
                "DEBATER_MIN_FREE_MIB": "12000",
                "MAX_WAIT_SECONDS": "1200",
            },
        },
    },
}


def get_serving_profile(profile_name: Optional[str]) -> Optional[Dict[str, object]]:
    """Return a deep copy of the selected serving profile or ``None``."""
    if not profile_name:
        return None
    if profile_name not in SERVING_PROFILES:
        available = ", ".join(sorted(SERVING_PROFILES))
        raise ValueError(
            f"Unknown serving profile '{profile_name}'. Available profiles: {available}"
        )
    return deepcopy(SERVING_PROFILES[profile_name])


def get_stage_env_defaults(profile_name: Optional[str], stage: str) -> Dict[str, str]:
    """Return environment defaults for the requested stage."""
    profile = get_serving_profile(profile_name)
    if profile is None:
        return {}
    env_defaults = profile.get("env_defaults", {})
    return deepcopy(env_defaults.get(stage, {}))


def apply_process_env_serving_profile_defaults(*, stage: str) -> Optional[Dict[str, object]]:
    """Apply serving-profile defaults without overriding explicit user values."""
    profile_name = os.environ.get("SERVING_PROFILE")
    profile = get_serving_profile(profile_name)
    if profile is None:
        return None

    for key, value in profile.get("env_defaults", {}).get(stage, {}).items():
        os.environ.setdefault(key, str(value))
    return profile
