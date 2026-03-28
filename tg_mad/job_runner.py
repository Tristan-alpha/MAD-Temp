"""Minimal Python orchestration for TG-MAD train/eval SLURM jobs."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from tg_mad.experiment_profiles import apply_process_env_profile_defaults


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_optional(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def env_int(name: str, default: int) -> int:
    return int(env_str(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(env_str(name, str(default)))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_optional_int(name: str) -> Optional[int]:
    value = env_optional(name)
    if value is None:
        return None
    return int(value)


def append_optional_arg(args: List[str], flag: str, value: Optional[str]) -> None:
    if value is not None:
        args.extend([flag, value])


def append_flag(args: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        args.append(flag)


def append_save_text_history_override(args: List[str]) -> None:
    """Text history now defaults to on; only append an explicit opt-out."""
    configured = env_optional("SAVE_TEXT_HISTORY")
    if configured is None:
        return
    if configured.strip().lower() in {"0", "false", "no", "off"}:
        args.append("--no_save_text_history")


def enforce_per_agent_prompt_mode() -> None:
    """Reject stale env attempts to force the removed shared-prompt mode."""
    configured = env_optional("PER_AGENT_PROMPTS")
    if configured is None:
        return
    if configured.strip().lower() in {"0", "false", "no", "off"}:
        raise RuntimeError(
            "Shared prompt mode has been removed. "
            "Delete PER_AGENT_PROMPTS=0 and rerun in the default per-agent mode."
        )


def ensure_vllm_installed() -> None:
    if importlib.util.find_spec("vllm") is None:
        raise RuntimeError(
            "vllm is not installed in the current environment. "
            "Install it first, for example with: INSTALL_VLLM=1 bash scripts/setup_textgrad_env.sh"
        )


def read_hf_token(repo_root: Path) -> Optional[str]:
    token_path = repo_root / "token"
    if token_path.exists():
        token = token_path.read_text().strip()
        return token or None
    return None


def load_dotenv_if_present(repo_root: Path) -> Dict[str, str]:
    env_updates: Dict[str, str] = {}
    env_path = repo_root / ".env"
    if not env_path.exists():
        return env_updates

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            env_updates[key] = value
    return env_updates


def build_base_env(repo_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "EMPTY")
    pythonpath_parts: List[str] = []

    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec and vllm_spec.origin:
        vllm_third_party = Path(vllm_spec.origin).resolve().parent / "third_party"
        if vllm_third_party.exists():
            pythonpath_parts.append(str(vllm_third_party))

    pythonpath_parts.append(str(repo_root))
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    hf_token = env.get("HF_TOKEN") or read_hf_token(repo_root)
    if hf_token:
        env["HF_TOKEN"] = hf_token
    return env


def _socket_family_for_host(host: str) -> socket.AddressFamily:
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def is_local_port_available(host: str, port: int) -> bool:
    family = _socket_family_for_host(host)
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
        return True
    except OSError:
        return False


def find_free_local_port(host: str) -> int:
    family = _socket_family_for_host(host)
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def resolve_local_server_port(
    *,
    name: str,
    host: str,
    preferred_port: int,
    should_start: bool,
    explicit_base_url: Optional[str],
) -> int:
    if not should_start or explicit_base_url is not None:
        return preferred_port
    if is_local_port_available(host, preferred_port):
        return preferred_port

    replacement_port = find_free_local_port(host)
    print(
        f"{name} port {preferred_port} is already in use on {host}; "
        f"switching to free port {replacement_port}.",
        flush=True,
    )
    return replacement_port


def split_device_spec(spec: str) -> List[str]:
    return [item for item in re.split(r"[\s,:]+", spec.strip()) if item]


def resolve_visible_devices(
    explicit_devices: Optional[str],
    slot_spec: str,
    job_visible_devices: Optional[str],
) -> str:
    if explicit_devices:
        return ",".join(split_device_spec(explicit_devices))

    if not job_visible_devices:
        return ",".join(split_device_spec(slot_spec))

    allocated = split_device_spec(job_visible_devices)
    slots = split_device_spec(slot_spec)
    resolved: List[str] = []
    for slot_text in slots:
        slot = int(slot_text)
        if slot < 0 or slot >= len(allocated):
            raise ValueError(
                f"GPU slot {slot} is outside allocated devices {job_visible_devices}"
            )
        resolved.append(allocated[slot])
    return ",".join(resolved)


def query_gpu_free_memory_mib() -> Dict[str, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True)
    except FileNotFoundError:
        print(
            "nvidia-smi is unavailable in the current environment; "
            "falling back to allocated GPU order for auto-pick.",
            flush=True,
        )
        return {}
    free_memory: Dict[str, int] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        index_text, free_text = [part.strip() for part in line.split(",", 1)]
        free_memory[index_text] = int(free_text)
    return free_memory


def query_gpu_topology_scores() -> Dict[tuple[str, str], int]:
    command = ["nvidia-smi", "topo", "-m"]
    try:
        output = subprocess.check_output(command, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}

    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    header: List[str] = []
    matrix_rows: List[List[str]] = []
    for line in lines:
        stripped = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        if stripped.startswith("GPU0"):
            header = stripped.split()
            continue
        if not header:
            continue
        if stripped.startswith("Legend:"):
            break
        if stripped.startswith("GPU"):
            matrix_rows.append(stripped.split())

    if not header or not matrix_rows:
        return {}

    score_map = {
        "NV": 6,
        "PIX": 5,
        "PXB": 4,
        "PHB": 3,
        "NODE": 2,
        "SYS": 1,
        "X": 0,
    }
    scores: Dict[tuple[str, str], int] = {}
    gpu_columns = [token for token in header if token.startswith("GPU")]
    for row in matrix_rows:
        row_gpu = row[0]
        if row_gpu not in gpu_columns:
            continue
        for idx, col_gpu in enumerate(gpu_columns, start=1):
            if idx >= len(row):
                break
            relation = row[idx]
            base_relation = relation if relation.startswith("NV") else relation
            score = score_map.get(base_relation, score_map.get(base_relation[:2], -1))
            if score >= 0 and row_gpu != col_gpu:
                scores[tuple(sorted((row_gpu.removeprefix("GPU"), col_gpu.removeprefix("GPU"))))] = score
    return scores


def auto_pick_visible_devices(
    *,
    job_visible_devices: Optional[str],
    count: int,
    prefer: str,
    exclude_devices: Optional[List[str]] = None,
    prefer_topology: bool = False,
    min_free_mib: Optional[int] = None,
) -> str:
    if not job_visible_devices:
        raise ValueError("Auto-pick GPU mode requires CUDA_VISIBLE_DEVICES to be set.")

    allocated = split_device_spec(job_visible_devices)
    excluded = set(exclude_devices or [])
    candidates = [device for device in allocated if device not in excluded]
    if len(candidates) < count:
        raise ValueError(
            f"Not enough candidate GPUs after exclusions: need {count}, have {candidates}"
        )

    free_memory = query_gpu_free_memory_mib()
    if free_memory and min_free_mib is not None:
        candidates = [
            device for device in candidates if free_memory.get(device, -1) >= min_free_mib
        ]
        if len(candidates) < count:
            details = ", ".join(
                f"{device}({free_memory.get(device, -1)} MiB free)"
                for device in allocated
                if device not in excluded
            )
            raise RuntimeError(
                f"Not enough GPUs satisfy the minimum free-memory requirement of "
                f"{min_free_mib} MiB: need {count}, have {len(candidates)} from "
                f"{details or allocated}"
            )
    if prefer == "highest":
        reverse = True
    elif prefer == "lowest":
        reverse = False
    else:
        raise ValueError(f"Unknown GPU auto-pick preference: {prefer}")

    if not free_memory:
        ranked = candidates[:]
        if reverse:
            ranked = list(reversed(ranked))
    elif prefer_topology and count > 1 and prefer == "highest":
        topology_scores = query_gpu_topology_scores()
        ranked_subsets = sorted(
            itertools.combinations(candidates, count),
            key=lambda subset: (
                min(free_memory.get(device, -1) for device in subset),
                min(
                    topology_scores.get(tuple(sorted((left, right))), -1)
                    for left, right in itertools.combinations(subset, 2)
                ),
                sum(free_memory.get(device, -1) for device in subset),
            ),
            reverse=True,
        )
        picked = list(ranked_subsets[0])
        order_index = {device: index for index, device in enumerate(allocated)}
        picked = sorted(picked, key=lambda device: order_index[device])
        print(
            f"Auto-picked GPUs ({prefer} free memory, topology-aware) from {candidates}: "
            + ", ".join(
                f"{device}({free_memory.get(device, 'NA')} MiB free)" for device in picked
            ),
            flush=True,
        )
        return ",".join(picked)
    else:
        ranked = sorted(
            candidates,
            key=lambda device: (free_memory.get(device, -1), device),
            reverse=reverse,
        )
    picked = ranked[:count]
    order_index = {device: index for index, device in enumerate(allocated)}
    picked = sorted(picked, key=lambda device: order_index[device])
    print(
        f"Auto-picked GPUs ({prefer} free memory) from {candidates}: "
        + ", ".join(f"{device}({free_memory.get(device, 'NA')} MiB free)" for device in picked),
        flush=True,
    )
    return ",".join(picked)


def validate_visible_devices_free_memory(
    *,
    name: str,
    visible_devices: str,
    min_free_mib: Optional[int],
) -> None:
    if min_free_mib is None:
        return

    free_memory = query_gpu_free_memory_mib()
    if not free_memory:
        print(
            f"Skipping {name} GPU free-memory validation because nvidia-smi is unavailable.",
            flush=True,
        )
        return

    selected = split_device_spec(visible_devices)
    insufficient = [
        (device, free_memory.get(device, -1))
        for device in selected
        if free_memory.get(device, -1) < min_free_mib
    ]
    if insufficient:
        details = ", ".join(
            f"{device}({free} MiB free)" for device, free in insufficient
        )
        raise RuntimeError(
            f"{name} selected GPUs do not satisfy the minimum free-memory "
            f"requirement of {min_free_mib} MiB: {details}"
        )


def wait_for_http_ready(
    *,
    name: str,
    url: str,
    process: subprocess.Popen,
    timeout_seconds: int,
    log_path: Path,
) -> None:
    print(f"Waiting for {name} health check at {url}...", flush=True)
    waited = 0
    while waited < timeout_seconds:
        try:
            with urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    print(f"{name} server is healthy.", flush=True)
                    return
        except (URLError, HTTPError):
            pass

        if process.poll() is not None:
            tail = read_log_tail(log_path)
            raise RuntimeError(
                f"{name} server exited before becoming healthy.\n"
                f"See {log_path}\n{tail}"
            )

        time.sleep(5)
        waited += 5
        print(f"  waiting for {name}... {waited}/{timeout_seconds}s", flush=True)

    raise RuntimeError(
        f"{name} server did not become healthy within {timeout_seconds}s. "
        f"See {log_path}"
    )


def read_log_tail(path: Path, n_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


@dataclass
class ServerProcess:
    name: str
    process: subprocess.Popen
    log_path: Path
    handle: object


def start_vllm_server(
    *,
    name: str,
    repo_root: Path,
    base_env: Dict[str, str],
    visible_devices: str,
    host: str,
    port: int,
    model_name: str,
    dtype: str,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    tensor_parallel_size: int,
    generation_config: str,
    max_num_batched_tokens: Optional[str],
    distributed_executor_backend: Optional[str],
    max_model_len: Optional[str],
    swap_space: Optional[str],
    cpu_offload_gb: Optional[str],
    disable_custom_all_reduce: bool,
    enforce_eager: bool,
    extra_args: Optional[str],
    log_path: Path,
    dry_run: bool,
) -> Optional[ServerProcess]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model_name,
        "--download-dir",
        "./models",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        str(max_num_seqs),
        "--dtype",
        dtype,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--generation-config",
        generation_config,
    ]
    if max_num_batched_tokens:
        command.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])
    if distributed_executor_backend:
        command.extend(
            ["--distributed-executor-backend", distributed_executor_backend]
        )
    if max_model_len:
        command.extend(["--max-model-len", str(max_model_len)])
    if swap_space:
        command.extend(["--swap-space", str(swap_space)])
    if cpu_offload_gb:
        command.extend(["--cpu-offload-gb", str(cpu_offload_gb)])
    if disable_custom_all_reduce:
        command.append("--disable-custom-all-reduce")
    if enforce_eager:
        command.append("--enforce-eager")
    if extra_args:
        command.extend(shlex.split(extra_args))

    print(
        f"Starting {name} vLLM server on devices {visible_devices} (TP={tensor_parallel_size})...",
        flush=True,
    )
    if dry_run:
        print(shlex.join(command), flush=True)
        return None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w")
    env = base_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ServerProcess(name=name, process=process, log_path=log_path, handle=handle)


def stop_processes(processes: List[ServerProcess]) -> None:
    for server in reversed(processes):
        if server.process.poll() is None:
            server.process.terminate()
    for server in reversed(processes):
        if server.process.poll() is None:
            try:
                server.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.process.kill()
                server.process.wait(timeout=15)
        server.handle.close()


def _redact_command(command: List[str]) -> List[str]:
    redacted = list(command)
    secret_flags = {"--evaluator_api_key"}
    i = 0
    while i < len(redacted):
        if redacted[i] in secret_flags and i + 1 < len(redacted):
            redacted[i + 1] = "<redacted>"
            i += 2
            continue
        i += 1
    return redacted


def run_command(command: List[str], *, cwd: Path, env: Dict[str, str], dry_run: bool) -> None:
    print(shlex.join(_redact_command(command)), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_train(dry_run: bool) -> None:
    apply_process_env_profile_defaults(stage="train")
    enforce_per_agent_prompt_mode()
    repo_root = Path(env_str("REPO_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
    output_dir = Path(env_str("OUTPUT_DIR", "out/tg_mad")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_env = build_base_env(repo_root)
    if env_str("EVALUATOR_TYPE", "local") == "api" and "KIMI_API_KEY" not in base_env:
        dotenv_vars = load_dotenv_if_present(repo_root)
        if "KIMI_API_KEY" in dotenv_vars:
            base_env["KIMI_API_KEY"] = dotenv_vars["KIMI_API_KEY"]
    max_wait_seconds = env_int("MAX_WAIT_SECONDS", 600)
    base_env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", str(max_wait_seconds))

    ensure_vllm_installed()

    host = env_str("SERVER_HOST", "127.0.0.1")
    debater_port = env_int("DEBATER_PORT", 8000)
    evaluator_port = env_int("EVALUATOR_PORT", 8001)
    evaluator_type = env_str("EVALUATOR_TYPE", "local")
    start_debater = env_bool("START_DEBATER_SERVER", True)
    start_evaluator = env_bool(
        "START_EVALUATOR_SERVER",
        evaluator_type != "api",
    )

    job_visible_devices = env_optional("CUDA_VISIBLE_DEVICES")
    debater_model_name = env_str("DEBATER_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
    debater_engine_model = env_optional("DEBATER_ENGINE_MODEL") or (
        f"hosted_vllm/{debater_model_name}"
    )
    evaluator_model_name = env_str("EVALUATOR_MODEL_NAME", "Qwen/Qwen3-8B")
    evaluator_engine_model = env_optional("EVALUATOR_ENGINE_MODEL") or (
        f"hosted_vllm/{evaluator_model_name}"
    )
    debater_tp = env_int("DEBATER_TENSOR_PARALLEL_SIZE", 1)
    evaluator_tp = env_int("EVALUATOR_TENSOR_PARALLEL_SIZE", 1)
    debater_auto_pick = env_bool("DEBATER_AUTO_PICK_GPU", False)
    evaluator_auto_pick = env_bool("EVALUATOR_AUTO_PICK_GPU", False)
    evaluator_first = (
        debater_auto_pick
        and evaluator_auto_pick
        and env_bool("EVALUATOR_AUTO_PICK_BEFORE_DEBATER", False)
    )
    if evaluator_auto_pick and evaluator_first:
        evaluator_visible = auto_pick_visible_devices(
            job_visible_devices=job_visible_devices,
            count=evaluator_tp,
            prefer=env_str("EVALUATOR_AUTO_PICK_PREFERENCE", "highest"),
            prefer_topology=env_bool("EVALUATOR_AUTO_PICK_PREFER_TOPOLOGY", False),
            min_free_mib=env_optional_int("EVALUATOR_MIN_FREE_MIB"),
        )
    elif evaluator_auto_pick:
        evaluator_visible = ""
    else:
        evaluator_visible = resolve_visible_devices(
            env_optional("EVALUATOR_CUDA_VISIBLE_DEVICES"),
            env_str("EVALUATOR_GPU_SLOTS", "1"),
            job_visible_devices,
        )
    if debater_auto_pick:
        debater_visible = auto_pick_visible_devices(
            job_visible_devices=job_visible_devices,
            count=debater_tp,
            prefer=env_str("DEBATER_AUTO_PICK_PREFERENCE", "lowest"),
            exclude_devices=split_device_spec(evaluator_visible),
            min_free_mib=env_optional_int("DEBATER_MIN_FREE_MIB"),
        )
    else:
        debater_visible = resolve_visible_devices(
            env_optional("DEBATER_CUDA_VISIBLE_DEVICES"),
            env_str("DEBATER_GPU_SLOTS", "0"),
            job_visible_devices,
        )
    if evaluator_auto_pick and not evaluator_first:
        evaluator_visible = auto_pick_visible_devices(
            job_visible_devices=job_visible_devices,
            count=evaluator_tp,
            prefer=env_str("EVALUATOR_AUTO_PICK_PREFERENCE", "highest"),
            exclude_devices=split_device_spec(debater_visible),
            prefer_topology=env_bool("EVALUATOR_AUTO_PICK_PREFER_TOPOLOGY", False),
            min_free_mib=env_optional_int("EVALUATOR_MIN_FREE_MIB"),
        )
    validate_visible_devices_free_memory(
        name="debater",
        visible_devices=debater_visible,
        min_free_mib=env_optional_int("DEBATER_MIN_FREE_MIB"),
    )
    validate_visible_devices_free_memory(
        name="evaluator",
        visible_devices=evaluator_visible,
        min_free_mib=env_optional_int("EVALUATOR_MIN_FREE_MIB"),
    )

    print("=== TG-MAD Training ===", flush=True)
    if env_optional("EXPERIMENT_PROFILE"):
        print(f"Experiment profile: {env_optional('EXPERIMENT_PROFILE')}", flush=True)
    print(f"Node: {env_str('SLURMD_NODENAME', os.uname().nodename)}", flush=True)
    print(f"Repo: {repo_root}", flush=True)
    print(
        f"Allocated job CUDA_VISIBLE_DEVICES: {job_visible_devices or '<unset>'}",
        flush=True,
    )

    explicit_debater_url = env_optional("DEBATER_BASE_URL")
    explicit_evaluator_url = env_optional("EVALUATOR_BASE_URL")
    debater_port = resolve_local_server_port(
        name="debater",
        host=host,
        preferred_port=debater_port,
        should_start=start_debater,
        explicit_base_url=explicit_debater_url,
    )
    evaluator_port = resolve_local_server_port(
        name="evaluator",
        host=host,
        preferred_port=evaluator_port,
        should_start=(evaluator_type == "local" and start_evaluator),
        explicit_base_url=explicit_evaluator_url,
    )

    processes: List[ServerProcess] = []
    try:
        debater_url = explicit_debater_url or f"http://{host}:{debater_port}/v1"
        evaluator_url = explicit_evaluator_url or f"http://{host}:{evaluator_port}/v1"

        debater = None
        evaluator = None

        if start_debater:
            debater = start_vllm_server(
                name="debater",
                repo_root=repo_root,
                base_env=base_env,
                visible_devices=debater_visible,
                host=host,
                port=debater_port,
                model_name=debater_model_name,
                dtype=env_str("DEBATER_DTYPE", "bfloat16"),
                gpu_memory_utilization=env_float("DEBATER_GPU_MEMORY", 0.45),
                max_num_seqs=env_int("DEBATER_MAX_NUM_SEQS", 4),
                tensor_parallel_size=debater_tp,
                generation_config=env_str("VLLM_GENERATION_CONFIG", "vllm"),
                max_num_batched_tokens=env_optional(
                    "DEBATER_MAX_NUM_BATCHED_TOKENS"
                ),
                distributed_executor_backend=env_optional(
                    "DEBATER_DISTRIBUTED_EXECUTOR_BACKEND"
                ),
                max_model_len=env_optional("DEBATER_MAX_MODEL_LEN") or "8192",
                swap_space=env_optional("DEBATER_SWAP_SPACE"),
                cpu_offload_gb=env_optional("DEBATER_CPU_OFFLOAD_GB"),
                disable_custom_all_reduce=env_bool(
                    "DEBATER_DISABLE_CUSTOM_ALL_REDUCE", False
                ),
                enforce_eager=env_bool("DEBATER_ENFORCE_EAGER", True),
                extra_args=env_optional("DEBATER_VLLM_EXTRA_ARGS"),
                log_path=output_dir / "vllm_debater_train.log",
                dry_run=dry_run,
            )
            if debater is not None:
                processes.append(debater)

        if evaluator_type == "local" and start_evaluator:
            evaluator = start_vllm_server(
                name="evaluator",
                repo_root=repo_root,
                base_env=base_env,
                visible_devices=evaluator_visible,
                host=host,
                port=evaluator_port,
                model_name=env_str("EVALUATOR_MODEL_NAME", "Qwen/Qwen3-8B"),
                dtype=env_str("EVALUATOR_DTYPE", "bfloat16"),
                gpu_memory_utilization=env_float("EVALUATOR_GPU_MEMORY", 0.55),
                max_num_seqs=env_int("EVALUATOR_MAX_NUM_SEQS", 1),
                tensor_parallel_size=evaluator_tp,
                generation_config=env_str("VLLM_GENERATION_CONFIG", "vllm"),
                max_num_batched_tokens=env_optional(
                    "EVALUATOR_MAX_NUM_BATCHED_TOKENS"
                ),
                distributed_executor_backend=env_optional(
                    "EVALUATOR_DISTRIBUTED_EXECUTOR_BACKEND"
                ),
                max_model_len=env_optional("EVALUATOR_MAX_MODEL_LEN") or "16384",
                swap_space=env_optional("EVALUATOR_SWAP_SPACE"),
                cpu_offload_gb=env_optional("EVALUATOR_CPU_OFFLOAD_GB"),
                disable_custom_all_reduce=env_bool(
                    "EVALUATOR_DISABLE_CUSTOM_ALL_REDUCE", False
                ),
                enforce_eager=env_bool("EVALUATOR_ENFORCE_EAGER", True),
                extra_args=env_optional("EVALUATOR_VLLM_EXTRA_ARGS"),
                log_path=output_dir / "vllm_evaluator_train.log",
                dry_run=dry_run,
            )
            if evaluator is not None:
                processes.append(evaluator)

        if debater is not None:
            wait_for_http_ready(
                name="debater",
                url=f"http://{host}:{debater_port}/health",
                process=debater.process,
                timeout_seconds=max_wait_seconds,
                log_path=debater.log_path,
            )

        if evaluator is not None:
            wait_for_http_ready(
                name="evaluator",
                url=f"http://{host}:{evaluator_port}/health",
                process=evaluator.process,
                timeout_seconds=max_wait_seconds,
                log_path=evaluator.log_path,
            )

        train_cmd = [
            sys.executable,
            "-u",
            "-m",
            "tg_mad.train",
            "--debater_base_url",
            debater_url,
            "--debater_model",
            debater_engine_model,
            "--batch_size",
            str(env_int("TRAIN_BATCH_SIZE", 2)),
            "--num_epochs",
            str(env_int("TRAIN_NUM_EPOCHS", 2)),
            "--train_size",
            str(env_int("TRAIN_SIZE", 10)),
            "--n_agents",
            str(env_int("TRAIN_N_AGENTS", 3)),
            "--n_rounds",
            str(env_int("TRAIN_N_ROUNDS", 3)),
            "--max_new_tokens",
            str(env_int("MAX_NEW_TOKENS", 512)),
            "--evaluator_max_new_tokens",
            str(env_int("EVALUATOR_MAX_NEW_TOKENS", 512)),
            "--seed",
            str(env_int("TRAIN_SEED", 42)),
            "--output_dir",
            str(output_dir),
            "--dataset",
            env_str("DATASET", "hh_rlhf"),
        ]
        append_optional_arg(
            train_cmd,
            "--experiment_profile",
            env_optional("EXPERIMENT_PROFILE"),
        )

        if evaluator_type == "api":
            evaluator_api_key = env_optional("EVALUATOR_API_KEY") or base_env.get(
                "KIMI_API_KEY"
            )
            if evaluator_api_key is None and not dry_run:
                raise RuntimeError(
                    "API evaluator requested but no evaluator API key was found. "
                    "Set EVALUATOR_API_KEY or KIMI_API_KEY."
                )
            train_cmd.extend(["--evaluator_type", "api"])
            append_optional_arg(
                train_cmd,
                "--evaluator_model",
                env_optional("EVALUATOR_API_MODEL"),
            )
            append_optional_arg(
                train_cmd,
                "--evaluator_api_base_url",
                env_optional("EVALUATOR_API_BASE_URL"),
            )
            append_optional_arg(train_cmd, "--evaluator_api_key", evaluator_api_key)
        else:
            train_cmd.extend(["--evaluator_base_url", evaluator_url])
            append_optional_arg(train_cmd, "--evaluator_model", evaluator_engine_model)

        append_save_text_history_override(train_cmd)
        append_flag(
            train_cmd,
            "--allow_failed_generations",
            env_bool("ALLOW_FAILED_GENERATIONS", False),
        )
        append_flag(
            train_cmd,
            "--skip_failed_optimizer_steps",
            env_bool("SKIP_FAILED_OPTIMIZER_STEPS", False),
        )
        append_optional_arg(train_cmd, "--text_history_dir", env_optional("TEXT_HISTORY_DIR"))
        append_optional_arg(train_cmd, "--data_dir", env_optional("DATA_DIR"))
        append_optional_arg(train_cmd, "--existing_data", env_optional("EXISTING_DATA"))
        append_optional_arg(
            train_cmd,
            "--train_existing_data",
            env_optional("TRAIN_EXISTING_DATA"),
        )
        append_optional_arg(
            train_cmd,
            "--eval_existing_data",
            env_optional("EVAL_EXISTING_DATA"),
        )
        append_optional_arg(
            train_cmd,
            "--prompt_history_file",
            env_optional("PROMPT_HISTORY_FILE"),
        )
        append_optional_arg(
            train_cmd,
            "--split_info_file",
            env_optional("SPLIT_INFO_FILE"),
        )
        append_optional_arg(
            train_cmd,
            "--run_config_file",
            env_optional("RUN_CONFIG_FILE"),
        )

        print("Running TG-MAD training...", flush=True)
        run_command(train_cmd, cwd=repo_root, env=base_env, dry_run=dry_run)
    finally:
        if not dry_run:
            stop_processes(processes)


def run_eval(dry_run: bool) -> None:
    apply_process_env_profile_defaults(stage="eval")
    repo_root = Path(env_str("REPO_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
    output_dir = Path(env_str("OUTPUT_DIR", "out/tg_mad")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_env = build_base_env(repo_root)
    max_wait_seconds = env_int("MAX_WAIT_SECONDS", 600)
    base_env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", str(max_wait_seconds))

    ensure_vllm_installed()

    host = env_str("SERVER_HOST", "127.0.0.1")
    debater_port = env_int("DEBATER_PORT", 8000)
    start_debater = env_bool("START_DEBATER_SERVER", True)
    debater_model_name = env_str("DEBATER_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
    debater_engine_model = env_optional("DEBATER_ENGINE_MODEL") or (
        f"hosted_vllm/{debater_model_name}"
    )

    job_visible_devices = env_optional("CUDA_VISIBLE_DEVICES")
    if env_bool("DEBATER_AUTO_PICK_GPU", False):
        debater_visible = auto_pick_visible_devices(
            job_visible_devices=job_visible_devices,
            count=env_int("DEBATER_TENSOR_PARALLEL_SIZE", 1),
            prefer=env_str("DEBATER_AUTO_PICK_PREFERENCE", "lowest"),
            min_free_mib=env_optional_int("DEBATER_MIN_FREE_MIB"),
        )
    else:
        debater_visible = resolve_visible_devices(
            env_optional("DEBATER_CUDA_VISIBLE_DEVICES"),
            env_str("DEBATER_GPU_SLOTS", "0"),
            job_visible_devices,
        )
    validate_visible_devices_free_memory(
        name="debater",
        visible_devices=debater_visible,
        min_free_mib=env_optional_int("DEBATER_MIN_FREE_MIB"),
    )

    print("=== TG-MAD Evaluation ===", flush=True)
    if env_optional("EXPERIMENT_PROFILE"):
        print(f"Experiment profile: {env_optional('EXPERIMENT_PROFILE')}", flush=True)
    print(f"Node: {env_str('SLURMD_NODENAME', os.uname().nodename)}", flush=True)
    print(f"Repo: {repo_root}", flush=True)
    print(
        f"Allocated job CUDA_VISIBLE_DEVICES: {job_visible_devices or '<unset>'}",
        flush=True,
    )

    explicit_debater_url = env_optional("DEBATER_BASE_URL")
    debater_port = resolve_local_server_port(
        name="debater",
        host=host,
        preferred_port=debater_port,
        should_start=start_debater,
        explicit_base_url=explicit_debater_url,
    )

    processes: List[ServerProcess] = []
    try:
        debater_url = explicit_debater_url or f"http://{host}:{debater_port}/v1"
        if start_debater:
            debater = start_vllm_server(
                name="debater",
                repo_root=repo_root,
                base_env=base_env,
                visible_devices=debater_visible,
                host=host,
                port=debater_port,
                model_name=debater_model_name,
                dtype=env_str("DEBATER_DTYPE", "bfloat16"),
                gpu_memory_utilization=env_float("DEBATER_GPU_MEMORY", 0.45),
                max_num_seqs=env_int("DEBATER_MAX_NUM_SEQS", 4),
                tensor_parallel_size=env_int("DEBATER_TENSOR_PARALLEL_SIZE", 1),
                generation_config=env_str("VLLM_GENERATION_CONFIG", "vllm"),
                max_num_batched_tokens=env_optional(
                    "DEBATER_MAX_NUM_BATCHED_TOKENS"
                ),
                distributed_executor_backend=env_optional(
                    "DEBATER_DISTRIBUTED_EXECUTOR_BACKEND"
                ),
                max_model_len=env_optional("DEBATER_MAX_MODEL_LEN") or "8192",
                swap_space=env_optional("DEBATER_SWAP_SPACE"),
                cpu_offload_gb=env_optional("DEBATER_CPU_OFFLOAD_GB"),
                disable_custom_all_reduce=env_bool(
                    "DEBATER_DISABLE_CUSTOM_ALL_REDUCE", False
                ),
                enforce_eager=env_bool("DEBATER_ENFORCE_EAGER", True),
                extra_args=env_optional("DEBATER_VLLM_EXTRA_ARGS"),
                log_path=output_dir / "vllm_debater_eval.log",
                dry_run=dry_run,
            )
            if debater is not None:
                processes.append(debater)
                wait_for_http_ready(
                    name="debater",
                    url=f"http://{host}:{debater_port}/health",
                    process=debater.process,
                    timeout_seconds=max_wait_seconds,
                    log_path=debater.log_path,
                )

        eval_module = (
            "tg_mad.evaluate_sweep"
            if env_bool("EVAL_CHECKPOINT_SWEEP", False)
            else "tg_mad.evaluate"
        )

        eval_cmd = [
            sys.executable,
            "-u",
            "-m",
            eval_module,
            "--debater_base_url",
            debater_url,
            "--debater_model",
            debater_engine_model,
            "--prompt_history",
            env_str("PROMPT_HISTORY_PATH", str(output_dir / "prompt_history.json")),
            "--output_dir",
            str(output_dir),
            "--n_agents",
            str(env_int("EVAL_N_AGENTS", 3)),
            "--n_rounds",
            str(env_int("EVAL_N_ROUNDS", 3)),
            "--max_new_tokens",
            str(env_int("MAX_NEW_TOKENS", 512)),
            "--seed",
            str(env_int("EVAL_SEED", 42)),
            "--dataset",
            env_str("DATASET", "hh_rlhf"),
        ]
        append_optional_arg(
            eval_cmd,
            "--experiment_profile",
            env_optional("EXPERIMENT_PROFILE"),
        )

        append_save_text_history_override(eval_cmd)
        append_flag(
            eval_cmd,
            "--allow_failed_generations",
            env_bool("ALLOW_FAILED_GENERATIONS", False),
        )
        append_optional_arg(eval_cmd, "--text_history_dir", env_optional("TEXT_HISTORY_DIR"))
        append_optional_arg(eval_cmd, "--split_info_file", env_optional("SPLIT_INFO_PATH"))
        append_optional_arg(eval_cmd, "--results_file", env_optional("RESULTS_FILE_PATH"))
        append_optional_arg(eval_cmd, "--run_config_file", env_optional("RUN_CONFIG_FILE_PATH"))
        append_optional_arg(
            eval_cmd,
            "--max_test_samples",
            env_optional("EVAL_MAX_TEST_SAMPLES"),
        )
        append_optional_arg(eval_cmd, "--data_dir", env_optional("DATA_DIR"))
        append_optional_arg(eval_cmd, "--existing_data", env_optional("EXISTING_DATA"))
        append_optional_arg(
            eval_cmd,
            "--train_existing_data",
            env_optional("TRAIN_EXISTING_DATA"),
        )
        append_optional_arg(
            eval_cmd,
            "--eval_existing_data",
            env_optional("EVAL_EXISTING_DATA"),
        )
        if eval_module == "tg_mad.evaluate":
            append_optional_arg(eval_cmd, "--prompt_index", env_optional("EVAL_PROMPT_INDEX"))
        else:
            append_optional_arg(
                eval_cmd,
                "--checkpoint_stride",
                env_optional("EVAL_SWEEP_CHECKPOINT_STRIDE"),
            )
            append_optional_arg(
                eval_cmd,
                "--checkpoint_indices",
                env_optional("EVAL_SWEEP_PROMPT_INDICES"),
            )
            append_optional_arg(
                eval_cmd,
                "--plot_file",
                env_optional("EVAL_SWEEP_PLOT_FILE"),
            )

        print("Running TG-MAD evaluation...", flush=True)
        run_command(eval_cmd, cwd=repo_root, env=base_env, dry_run=dry_run)
    finally:
        if not dry_run:
            stop_processes(processes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TG-MAD SLURM job runner")
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch plan and commands without starting servers or running train/eval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_train(dry_run=args.dry_run)
    else:
        run_eval(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
