"""Thin wrapper around vLLM's OpenAI API server with TG-MAD-specific hooks."""

from __future__ import annotations

import os

import uvloop
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configure_pre_cli_env() -> None:
    if _env_flag_enabled("TG_MAD_SKIP_VLLM_KERNEL_WARMUP"):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")


def _maybe_patch_kernel_warmup() -> None:
    if not _env_flag_enabled("TG_MAD_SKIP_VLLM_KERNEL_WARMUP"):
        return

    from vllm.v1.worker import gpu_worker
    import vllm.v1.worker.gpu.warmup as gpu_warmup

    def _skip_warmup(model_runner) -> None:
        print(
            "[tg_mad.vllm_api_server] Skipping vLLM warmup_kernels because "
            "TG_MAD_SKIP_VLLM_KERNEL_WARMUP=1.",
            flush=True,
        )
        return None

    gpu_warmup.warmup_kernels = _skip_warmup
    gpu_worker.warmup_kernels = _skip_warmup


def main() -> None:
    _configure_pre_cli_env()
    cli_env_setup()
    _maybe_patch_kernel_warmup()

    parser = FlexibleArgumentParser(
        description="TG-MAD wrapper for the vLLM OpenAI-compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
