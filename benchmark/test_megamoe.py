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

"""Benchmark the multi-rank Hopper MegaMoE kernel on representative shapes."""

import os
import re
import shutil
import signal
import subprocess
import sys

import pytest
import torch

from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamoe import (
    MEGAMOE_KERNEL_PATH,
)

from . import base
from .conftest import Config
from .consts import DEFAULT_ITER_TIME, DEFAULT_WARMUP_TIME, BenchMode

OP_NAME = "megamoe"
NUM_RANKS = 8
STAGES = 4
DEFAULT_SHAPES = [
    (512, 2048, 512, 256, 8, NUM_RANKS),
    (1024, 2048, 512, 256, 8, NUM_RANKS),
    (2048, 2048, 512, 256, 8, NUM_RANKS),
]
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 30
DEFAULT_TIMEOUT_S = 1800

BENCH_LINE = re.compile(
    r"\[rank (?P<rank>\d+)/(?P<ranks>\d+)\] BENCH .*?\|\s*"
    r"(?P<latency_us>[0-9.]+) us"
)


def _resolve_mpirun():
    return shutil.which("mpirun")


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
    launcher = _resolve_mpirun()
    if launcher is None:
        raise RuntimeError("MegaMoE requires mpirun on PATH")

    warmup = _iteration_count(
        "--warmup", Config.warm_up, DEFAULT_WARMUP_TIME, DEFAULT_WARMUP
    )
    iters = _iteration_count(
        "--iter", Config.repetition, DEFAULT_ITER_TIME, DEFAULT_ITERS
    )
    command = [
        sys.executable,
        str(MEGAMOE_KERNEL_PATH),
        "--num-ranks",
        str(num_ranks),
        "--tokens",
        str(tokens),
        "--hidden-size",
        str(hidden),
        "--intermediate-size",
        str(intermediate),
        "--num-experts",
        str(num_experts),
        "--topk",
        str(topk),
        "--stages",
        str(STAGES),
        "--drop-rate",
        "0",
        "--benchmark",
        "--warmup",
        str(warmup),
        "--iterations",
        str(iters),
        "--reduce",
        "mean",
        "--gpu-start-barrier",
        "--timeout",
        str(DEFAULT_TIMEOUT_S),
        "--mpirun",
        launcher,
    ]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=DEFAULT_TIMEOUT_S + 120)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"MegaMoE timed out after {DEFAULT_TIMEOUT_S + 120}s"
        ) from None
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
        self.shapes = DEFAULT_SHAPES
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
        pytest.skip("requires mpirun on PATH")

    MegaMoEBenchmark().run()
