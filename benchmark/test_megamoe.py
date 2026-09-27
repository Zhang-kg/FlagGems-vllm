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

"""Benchmark the multi-rank Hopper MegaMoE kernel on real FP8 data."""

import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamoe import (
    MEGAMOE_KERNEL_PATH,
)

from . import base
from .conftest import Config
from .consts import DEFAULT_ITER_TIME, DEFAULT_WARMUP_TIME, BenchMode

OP_NAME = "megamoe"
NUM_RANKS = int(os.environ.get("MEGAMOE_NP", 8))
HIDDEN = int(os.environ.get("W_K", 4096))
INTERMEDIATE = int(os.environ.get("W_INTER", 1536))
NUM_EXPERTS = int(os.environ.get("W_NEXP", 128))
TOPK = int(os.environ.get("W_TOPK", 8))
STAGES = int(os.environ.get("W_STAGES", 4))

DEFAULT_TOKENS = (512, 1024, 2048)
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 30
DEFAULT_TIMEOUT_S = 1800

BENCH_LINE = re.compile(
    r"\[rank (?P<rank>\d+)/(?P<ranks>\d+)\] BENCH .*?\|\s*"
    r"(?P<latency_us>[0-9.]+) us"
)


def _token_sweep():
    raw = os.environ.get("MEGAMOE_BENCH_TOKENS", "").strip()
    if not raw:
        return DEFAULT_TOKENS
    return tuple(sorted({int(token) for token in raw.replace(",", " ").split()}))


def _resolve_mpirun():
    return shutil.which(os.environ.get("MEGAMOE_MPIRUN", "mpirun"))


def _iteration_count(option, configured, default, fallback):
    explicitly_set = any(
        arg == option or arg.startswith(f"{option}=") for arg in sys.argv
    )
    return configured if explicitly_set or configured != default else fallback


def _terminate_process_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if proc.poll() is None:
        proc.wait()


def _run_megamoe(tokens, hidden, intermediate, num_experts, topk, num_ranks):
    data_dir = Path(os.environ["MEGAMOE_SHARED_DATA_DIR"]).expanduser().resolve()
    launcher = _resolve_mpirun()
    timeout_s = int(os.environ.get("MEGAMOE_BENCH_TIMEOUT", DEFAULT_TIMEOUT_S))
    warmup = _iteration_count(
        "--warmup", Config.warm_up, DEFAULT_WARMUP_TIME, DEFAULT_WARMUP
    )
    iters = _iteration_count(
        "--iter", Config.repetition, DEFAULT_ITER_TIME, DEFAULT_ITERS
    )

    env = os.environ.copy()
    env.update(
        {
            "MEGAMOE_NP": str(num_ranks),
            "MEGAMOE_MPIRUN": launcher,
            "MEGAMOE_SHARED_DATA_DIR": str(data_dir),
            "W_NTOK": str(tokens),
            "W_TOPK": str(topk),
            "W_NEXP": str(num_experts),
            "W_K": str(hidden),
            "W_INTER": str(intermediate),
            "W_DROP": "0",
            "W_STAGES": str(STAGES),
            "W_BENCH": "1",
            "W_WARMUP": str(warmup),
            "W_ITERS": str(iters),
            "W_BENCH_REDUCE": "mean",
            "W_GPU_START_BARRIER": "1",
            "W_TIMEOUT": str(timeout_s),
        }
    )
    env.pop("TRITON_CACHE_DIR", None)

    python = os.environ.get("TLE_PYTHON_OVERRIDE", sys.executable)
    proc = subprocess.Popen(
        [python, str(MEGAMOE_KERNEL_PATH)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s + 120)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"MegaMoE timed out after {timeout_s + 120}s") from None
    else:
        if proc.returncode != 0:
            detail = (stderr.strip() or stdout.strip())[-1000:]
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"MegaMoE exited with status {proc.returncode}{suffix}")

        rank_latencies = {}
        for match in BENCH_LINE.finditer(stdout):
            if int(match["ranks"]) == num_ranks:
                rank_latencies[int(match["rank"])] = float(match["latency_us"]) / 1e3

        expected_ranks = set(range(num_ranks))
        if set(rank_latencies) != expected_ranks:
            raise RuntimeError(
                "MegaMoE did not report latency for every rank: "
                f"expected {sorted(expected_ranks)}, got {sorted(rank_latencies)}"
            )
        return max(rank_latencies.values())
    finally:
        if proc.poll() is None or sys.exc_info()[0] is not None:
            _terminate_process_group(proc)


class MegaMoEBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency"]
    DEFAULT_DTYPES = [torch.float8_e4m3fn]
    DEFAULT_SHAPE_DESC = "tokens, hidden, intermediate, num_experts, topk, num_ranks"

    def __init__(self):
        super().__init__(
            op_name=OP_NAME,
            torch_op=_run_megamoe,
            gems_op=_run_megamoe,
            dtypes=self.DEFAULT_DTYPES,
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (tokens, HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOPK, NUM_RANKS)
            for tokens in _token_sweep()
        ]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, dtype):
        yield from self.shapes

    def get_latency(self, op, *shape, **kwargs):
        if Config.mode is not BenchMode.KERNEL:
            raise ValueError("MegaMoE only supports --mode kernel")
        return op(*shape)


@pytest.mark.megamoe
def test_megamoe():
    if Config.query:
        MegaMoEBenchmark().run()
        return
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip(f"requires SM90, got SM{major}0")
    if torch.cuda.device_count() < NUM_RANKS:
        pytest.skip(f"requires {NUM_RANKS} visible GPUs")
    if _resolve_mpirun() is None:
        pytest.skip("requires OpenMPI; set MEGAMOE_MPIRUN if it is not on PATH")
    data_dir = os.environ.get("MEGAMOE_SHARED_DATA_DIR", "").strip()
    if not data_dir or not Path(data_dir).expanduser().is_dir():
        pytest.skip("set MEGAMOE_SHARED_DATA_DIR to the real FP8 dataset")

    MegaMoEBenchmark().run()
