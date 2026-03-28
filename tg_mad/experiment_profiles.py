"""Reusable experiment-profile defaults and metadata."""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import Any, Dict, Optional


EXPERIMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "paper_anchor_formal_logic_qwen25_t2": {
        "description": (
            "Paper-anchor-compatible Formal Logic comparison using "
            "Qwen2.5-7B-Instruct, N=5, T=2, seed=42, and max_new_tokens=2048."
        ),
        "paper_anchor_reference": {
            "dataset": "formal_logic",
            "split": "mmlu_formal_logic_test_126",
            "topology": "decentralized_mad",
            "debater_model": "Qwen/Qwen2.5-7B-Instruct",
            "n_agents": 5,
            "n_rounds": 2,
            "seed": 42,
            "temperature": 1.0,
            "top_p": 0.9,
            "max_new_tokens": 2048,
        },
        "baseline_definition": "t0_mean_over_all_agents_from_single_n5_t2_seed42_run",
        "tgmad_deviations_from_paper": [
            "single_agent_accuracy uses the round-0 mean over all agents from one N=5, T=2, seed=42 run",
            "max_new_tokens is 2048 for all methods instead of the paper's 512",
            "TG-MAD uses per-agent system-prompt optimization instead of a shared prompt",
            "TG-MAD uses Qwen/Qwen3-30B-A3B-Instruct-2507 as evaluator/optimizer",
        ],
        "stages": {
            "generate_history": {
                "dataset": "formal_logic",
                "debater_model": "hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
                "n_agents": 5,
                "n_rounds": 2,
                "max_new_tokens": 2048,
                "seed": 42,
            },
            "train": {
                "dataset": "formal_logic",
                "debater_model": "hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
                "evaluator_model": "hosted_vllm/Qwen/Qwen3-30B-A3B-Instruct-2507",
                "n_agents": 5,
                "n_rounds": 2,
                "max_new_tokens": 2048,
                "evaluator_max_new_tokens": 2048,
                "seed": 42,
            },
            "eval": {
                "dataset": "formal_logic",
                "debater_model": "hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
                "n_agents": 5,
                "n_rounds": 2,
                "max_new_tokens": 2048,
                "seed": 42,
            },
            "eval_sweep": {
                "dataset": "formal_logic",
                "debater_model": "hosted_vllm/Qwen/Qwen2.5-7B-Instruct",
                "n_agents": 5,
                "n_rounds": 2,
                "max_new_tokens": 2048,
                "seed": 42,
            },
        },
        "env_defaults": {
            "baseline": {
                "DATASET": "formal_logic",
                "DEBATER_MODEL_NAME": "Qwen/Qwen2.5-7B-Instruct",
                "N_AGENTS": "5",
                "N_ROUNDS": "2",
                "MAX_NEW_TOKENS": "2048",
                "SEED": "42",
            },
            "train": {
                "DATASET": "formal_logic",
                "DEBATER_MODEL_NAME": "Qwen/Qwen2.5-7B-Instruct",
                "EVALUATOR_MODEL_NAME": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "TRAIN_N_AGENTS": "5",
                "TRAIN_N_ROUNDS": "2",
                "MAX_NEW_TOKENS": "2048",
                "EVALUATOR_MAX_NEW_TOKENS": "2048",
                "TRAIN_SEED": "42",
            },
            "eval": {
                "DATASET": "formal_logic",
                "DEBATER_MODEL_NAME": "Qwen/Qwen2.5-7B-Instruct",
                "EVAL_N_AGENTS": "5",
                "EVAL_N_ROUNDS": "2",
                "MAX_NEW_TOKENS": "2048",
                "EVAL_SEED": "42",
            },
        },
    }
}


def get_experiment_profile(profile_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return a deep copy of the selected profile or ``None``."""
    if not profile_name:
        return None
    if profile_name not in EXPERIMENT_PROFILES:
        available = ", ".join(sorted(EXPERIMENT_PROFILES))
        raise ValueError(
            f"Unknown experiment profile '{profile_name}'. Available profiles: {available}"
        )
    return deepcopy(EXPERIMENT_PROFILES[profile_name])


def apply_argparse_profile_defaults(
    args: Namespace,
    parser: ArgumentParser,
    *,
    stage: str,
) -> Optional[Dict[str, Any]]:
    """Fill unset argparse options from the selected experiment profile."""
    profile = get_experiment_profile(getattr(args, "experiment_profile", None))
    if profile is None:
        return None

    for field, value in profile.get("stages", {}).get(stage, {}).items():
        if not hasattr(args, field):
            continue
        if getattr(args, field) == parser.get_default(field):
            setattr(args, field, deepcopy(value))
    return profile


def get_stage_env_defaults(
    profile_name: Optional[str],
    stage: str,
) -> Dict[str, str]:
    """Return environment-variable defaults for a profile stage."""
    profile = get_experiment_profile(profile_name)
    if profile is None:
        return {}
    return deepcopy(profile.get("env_defaults", {}).get(stage, {}))


def apply_process_env_profile_defaults(*, stage: str) -> Optional[Dict[str, Any]]:
    """Apply profile env defaults to ``os.environ`` without overriding explicit values."""
    profile_name = os.environ.get("EXPERIMENT_PROFILE")
    profile = get_experiment_profile(profile_name)
    if profile is None:
        return None

    for key, value in profile.get("env_defaults", {}).get(stage, {}).items():
        os.environ.setdefault(key, str(value))
    return profile


def build_profile_metadata(
    profile_name: Optional[str],
    *,
    include_tgmad_deviations: bool = False,
) -> Dict[str, Any]:
    """Return additive metadata fields for artifacts produced under a profile."""
    profile = get_experiment_profile(profile_name)
    if profile is None:
        return {}

    metadata: Dict[str, Any] = {
        "experiment_profile": profile_name,
        "paper_anchor_reference": deepcopy(profile.get("paper_anchor_reference")),
        "baseline_definition": profile.get("baseline_definition"),
    }
    if include_tgmad_deviations:
        metadata["tgmad_deviations_from_paper"] = deepcopy(
            profile.get("tgmad_deviations_from_paper", [])
        )
    return metadata


def build_runtime_tgmad_deviations(
    profile_name: Optional[str],
    *,
    prompt_mode: Optional[str] = None,
    optimizer_model: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> list[str]:
    """Return TG-MAD deviation notes with runtime-specific fields filled in."""
    profile = get_experiment_profile(profile_name)
    if profile is None:
        return []

    deviations = deepcopy(profile.get("tgmad_deviations_from_paper", []))
    runtime_prompt_note = None
    if prompt_mode == "per_agent_system_prompt":
        runtime_prompt_note = (
            "TG-MAD uses per-agent system-prompt optimization instead of a shared prompt"
        )
    elif prompt_mode == "shared_system_prompt":
        runtime_prompt_note = (
            "TG-MAD uses shared system-prompt optimization instead of the paper baseline"
        )

    runtime_optimizer_note = None
    if optimizer_model:
        runtime_optimizer_note = (
            f"TG-MAD uses {optimizer_model} as evaluator/optimizer"
        )

    runtime_token_note = None
    if max_new_tokens is not None:
        runtime_token_note = (
            f"max_new_tokens is {max_new_tokens} for all methods instead of the paper's 512"
        )

    updated: list[str] = []
    for note in deviations:
        if (
            runtime_prompt_note is not None
            and "system-prompt optimization" in note
        ):
            updated.append(runtime_prompt_note)
        elif runtime_optimizer_note is not None and "evaluator/optimizer" in note:
            updated.append(runtime_optimizer_note)
        elif runtime_token_note is not None and note.startswith("max_new_tokens is "):
            updated.append(runtime_token_note)
        else:
            updated.append(note)
    return updated
