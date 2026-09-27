# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal MPI runner used by the standalone MegaMoE benchmark."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

_KERNEL_PATH = Path(__file__).with_name("kernel.py")
_BENCH_LINE = re.compile(
    r"\[rank (?P<rank>\d+)/(?P<ranks>\d+)\] BENCH .*?\|\s*"
    r"(?P<latency_us>[0-9.]+) us"
)


@dataclass(frozen=True)
class MegaMoEConfig:
    worker: bool = False
    num_ranks: int = 2
    tokens: int = 128
    hidden_size: int = 256
    intermediate_size: int = 128
    num_experts: int = 16
    topk: int = 4
    stages: int = 2
    num_sms: int = 0
    max_recv: int = 512
    drop_rate: float = 0.1
    benchmark: bool = False
    warmup: int = 5
    iterations: int = 20
    reduce: str = "median"
    gpu_start_barrier: bool = False
    data_dir: Path | None = None
    mpirun: str | None = None
    timeout: int = 600
    cuda_home: Path | None = None
    nvshmem_home: Path | None = None


def _config_parser() -> argparse.ArgumentParser:
    defaults = MegaMoEConfig()
    parser = argparse.ArgumentParser(description="Run the Hopper MegaMoE benchmark.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-ranks", type=int, default=defaults.num_ranks)
    parser.add_argument("--tokens", type=int, default=defaults.tokens)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument(
        "--intermediate-size", type=int, default=defaults.intermediate_size
    )
    parser.add_argument("--num-experts", type=int, default=defaults.num_experts)
    parser.add_argument("--topk", type=int, default=defaults.topk)
    parser.add_argument("--stages", type=int, default=defaults.stages)
    parser.add_argument("--num-sms", type=int, default=defaults.num_sms)
    parser.add_argument("--max-recv", type=int, default=defaults.max_recv)
    parser.add_argument("--drop-rate", type=float)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=defaults.warmup)
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--reduce", choices=("mean", "median"), default=defaults.reduce)
    parser.add_argument("--gpu-start-barrier", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--mpirun")
    parser.add_argument("--timeout", type=int, default=defaults.timeout)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--nvshmem-home", type=Path)
    return parser


def parse_megamoe_config(argv: list[str] | None = None) -> MegaMoEConfig:
    args = _config_parser().parse_args(argv)
    if args.drop_rate is None:
        args.drop_rate = 0.0 if args.data_dir is not None else 0.1
    return MegaMoEConfig(**vars(args))


def _worker_cli_args(config: MegaMoEConfig) -> list[str]:
    args = [
        "--worker",
        "--num-ranks",
        str(config.num_ranks),
        "--tokens",
        str(config.tokens),
        "--hidden-size",
        str(config.hidden_size),
        "--intermediate-size",
        str(config.intermediate_size),
        "--num-experts",
        str(config.num_experts),
        "--topk",
        str(config.topk),
        "--stages",
        str(config.stages),
        "--num-sms",
        str(config.num_sms),
        "--max-recv",
        str(config.max_recv),
        "--drop-rate",
        str(config.drop_rate),
        "--warmup",
        str(config.warmup),
        "--iterations",
        str(config.iterations),
        "--reduce",
        config.reduce,
    ]
    if config.benchmark:
        args.append("--benchmark")
    if config.gpu_start_barrier:
        args.append("--gpu-start-barrier")
    if config.data_dir is not None:
        args.extend(("--data-dir", str(config.data_dir)))
    if config.cuda_home is not None:
        args.extend(("--cuda-home", str(config.cuda_home)))
    if config.nvshmem_home is not None:
        args.extend(("--nvshmem-home", str(config.nvshmem_home)))
    return args


def _resolve_runtime_paths(config: MegaMoEConfig) -> MegaMoEConfig:
    if __package__:
        from .runtime import _find_cuda_home, _find_nvshmem_home
    else:
        from runtime import _find_cuda_home, _find_nvshmem_home

    return replace(
        config,
        cuda_home=_find_cuda_home(config.cuda_home),
        nvshmem_home=_find_nvshmem_home(config.nvshmem_home),
    )


def _slowest_rank_latency(stdout: str, num_ranks: int) -> float:
    latencies = {
        int(match["rank"]): float(match["latency_us"]) / 1e3
        for match in _BENCH_LINE.finditer(stdout)
        if int(match["ranks"]) == num_ranks
    }
    assert set(latencies) == set(range(num_ranks))
    return max(latencies.values())


def launch_megamoe(config: MegaMoEConfig) -> float | None:
    config = _resolve_runtime_paths(config)
    mpirun = config.mpirun or shutil.which("mpirun")
    assert mpirun is not None

    command = [
        mpirun,
        "--allow-run-as-root",
        "-np",
        str(config.num_ranks),
        sys.executable,
        str(_KERNEL_PATH),
        *_worker_cli_args(config),
    ]
    env = os.environ.copy()
    env.pop("TRITON_CACHE_DIR", None)
    env.update({"NVSHMEM_BOOTSTRAP": "MPI", "CUDA_HOME": str(config.cuda_home)})
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=config.timeout)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, output=stdout)
    return _slowest_rank_latency(stdout, config.num_ranks) if config.benchmark else None


def launch_megamoe_cli(config: MegaMoEConfig) -> int:
    launch_megamoe(config)
    return 0
