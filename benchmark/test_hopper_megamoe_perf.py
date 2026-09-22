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

"""Eight-rank Hopper Triton/TLE MegaMoE benchmark on real Qwen3 FP8 data.

The operator is one persistent Triton/TLE kernel launch per rank; it owns the
dispatch, the FP8 L1 gate/up GEMM, SwiGLU, the L2 down projection, the remote
L2 scatter and the local top-k combine.  It needs NVSHMEM, one GPU per rank and
an MPI launcher, so it cannot be driven in-process the way a single-device
operator is and it does not use ``benchmark.base.Benchmark``.  This file drives
the operator's own multi-rank entry point as a subprocess -- one subprocess per
benchmarked token count -- and parses the per-rank ``BENCH`` line it prints.

The latency of a collective is set by its slowest rank, so the recorded latency
is the maximum over ranks.  Every rank is printed as well, because rank spread
is the first thing to look at when a number moves.

Inputs, routing, expert weights and scales come from the immutable Qwen3 FP8
dataset.  The kernel's synthetic path exists for legacy regression runs only
and is never benchmarked here: it is not a real-data comparison.

Requirements
------------

- SM90 (H100) with at least eight visible GPUs;
- a FlagTree/TLE Triton build providing ``buffered_tensor.subslice``, multiple
  pure-TMA writers on one ``tle.pipe``, and multi-writer TMA token
  ``full_count`` lowering;
- CUDA, OpenMPI and NVSHMEM;
- the Qwen3 FP8 shared dataset: ``manifest.json``, ``shared.pt`` and one
  ``rankNN.pt`` per rank.

Environment
-----------

- ``MEGAMOE_SHARED_DATA_DIR``: Qwen3 FP8 dataset root.  Required.
- ``MEGAMOE_BENCH_TOKENS``: tokens per rank to sweep.  Default ``512 1024 2048``.
- ``MEGAMOE_BENCH_WARMUP``: warm-up iterations per token count.  Default 10.
- ``MEGAMOE_BENCH_ITERS``: measured iterations per token count.  Default 30.
- ``MEGAMOE_BENCH_TIMEOUT``: seconds allowed per token count.  Default 1800.
- ``MEGAMOE_MPIRUN``: OpenMPI launcher.  Default: ``mpirun`` on ``PATH``.
- ``TLE_PYTHON_OVERRIDE``: FlagTree/TLE interpreter that runs the entry
  point and its MPI workers.  Defaults to the one running pytest.
- ``MEGAMOE_TORCH_SITE_PACKAGES``: site-packages holding PyTorch/NVSHMEM.
- ``NVSHMEM_HOME``: NVSHMEM root containing ``include/`` and ``lib/``.
- ``CUDA_HOME``: CUDA toolkit root.  Defaults to ``/usr/local/cuda-12.8``.
- ``CLANG``: clang used by TLE raw CUDA compilation.
- ``W_CACHE_ROOT``: per-rank Triton cache root.
- ``MEGAMOE_BUILD_DIR``: cache for the host-side NVSHMEM wrapper.

Run it with the intended FlagTree Python, and do not wrap the command in
another ``mpirun``: the entry point launches one worker per rank itself::

    MEGAMOE_SHARED_DATA_DIR=/data/qwen3-235b-a22b-fp8/layer0_np8_t4096 \\
    PYTHONPATH=src pytest -q -s benchmark/test_hopper_megamoe_perf.py

Only benchmark after correctness passes on the same node and compiler.  The
eight-rank correctness run is the same entry point with ``W_BENCH=0``; every
rank must report ``scatter_bad=0 ... errors=0 ... -> PASS``::

    MEGAMOE_SHARED_DATA_DIR=/data/qwen3-235b-a22b-fp8/layer0_np8_t4096 \\
    MEGAMOE_NP=8 W_NTOK=512 W_TOPK=8 W_NEXP=128 W_K=4096 W_INTER=1536 \\
    W_DROP=0 W_STAGES=4 W_BENCH=0 W_TIMEOUT=900 \\
    python src/flaggems_vllm/runtime/backend/_nvidia/hopper/mega/megamoe/kernel.py

Keep the exact FlagTree/TLE compiler revision next to every number, and do not
compare results collected with different inputs, compiler revisions, clocks or
rank-reduction rules.  When reproducing on the historical environment-gated
PR837 compiler build, set ``TLE_MULTI_TMA_WRITERS=1`` externally; the operator
no longer sets that legacy compiler switch itself.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from .conftest import Config, emit_record_logger, update_result
from .consts import BenchmarkMetrics, BenchmarkResult

try:
    from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamoe import (
        MEGAMOE_KERNEL_PATH,
    )
except Exception:  # non-NVIDIA builds do not ship the Hopper mega kernels
    MEGAMOE_KERNEL_PATH = None

OP_NAME = "hopper_megamoe"

# The previously validated UserHopper-aligned launch shape.  The candidate has
# no autotune table and no registered generic API, so the shape is fixed here
# rather than read from core_shapes.yaml.
NUM_RANKS = 8
HIDDEN = 4096
INTERMEDIATE = 1536
NUM_EXPERTS = 128
TOPK = 8
STAGES = 4

# kernel.py sizes the per (local expert, source rank) dispatch queue with this
# constant.  Real Qwen3 routing is imbalanced, so a token count whose hottest
# queue would exceed it is dropped from the sweep instead of overrunning it.
MAX_RECV = 512

DEFAULT_TOKENS = (512, 1024, 2048)
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 30
DEFAULT_TIMEOUT_S = 1800

# "[rank 0/8] BENCH h=4096 ih=1536 E=128 k=8 tokens=512 recv=4096 experts=16 |
#     351.2 us   123.4 TFLOPS  ws=True ... data=Qwen/Qwen3-235B-A22B-FP8@...:layer0"
BENCH_LINE = re.compile(
    r"\[rank (?P<rank>\d+)/(?P<ranks>\d+)\] BENCH "
    r"h=(?P<hidden>\d+) ih=(?P<intermediate>\d+) E=(?P<experts>\d+) "
    r"k=(?P<topk>\d+) tokens=(?P<tokens>\d+) recv=(?P<recv>\d+) "
    r"experts=(?P<experts_per_rank>\d+) \|\s*(?P<us>[0-9.]+) us\s+"
    r"(?P<tflops>[0-9.]+) TFLOPS"
)
DATA_LABEL = re.compile(r"data=(?P<label>\S+)")


def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _resolve_mpirun():
    override = os.environ.get("MEGAMOE_MPIRUN", "").strip()
    if override:
        return shutil.which(override)
    return shutil.which("mpirun")


def _interpreter():
    """Python that runs the entry point and, through it, the MPI workers.

    The entry point imports the FlagTree/TLE Triton at module scope, so it has
    to run under the TLE interpreter.  That is usually not the interpreter
    running pytest, which only needs torch: TLE_PYTHON_OVERRIDE names it, and
    the entry point already uses the same variable for its workers.
    """
    override = os.environ.get("TLE_PYTHON_OVERRIDE", "").strip()
    return override or sys.executable


def _token_sweep():
    raw = os.environ.get("MEGAMOE_BENCH_TOKENS", "").strip()
    if not raw:
        return list(DEFAULT_TOKENS)
    return sorted({int(token) for token in raw.replace(",", " ").split()})


def _skip_reason():
    if MEGAMOE_KERNEL_PATH is None or not MEGAMOE_KERNEL_PATH.is_file():
        return "the Hopper MegaMoE entry point is not available in this build"
    if not torch.cuda.is_available():
        return "requires cuda"
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        return f"requires SM90, got SM{major}0"
    visible = torch.cuda.device_count()
    if visible < NUM_RANKS:
        return f"requires {NUM_RANKS} visible GPUs, got {visible}"
    if _resolve_mpirun() is None:
        return "an OpenMPI launcher is required; set MEGAMOE_MPIRUN"
    if not os.environ.get("MEGAMOE_SHARED_DATA_DIR", "").strip():
        return (
            "set MEGAMOE_SHARED_DATA_DIR to the immutable Qwen3 FP8 dataset; "
            "the synthetic path is a regression mode, not a benchmark"
        )
    return None


def _dataset_root():
    root = Path(os.environ["MEGAMOE_SHARED_DATA_DIR"].strip()).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        # The variable was set on purpose, so a bad path is a configuration
        # error and not a reason to silently report nothing.
        pytest.fail(f"MEGAMOE_SHARED_DATA_DIR={root} has no manifest.json")
    return root, json.loads(manifest_path.read_text(encoding="utf-8"))


def _dataset_tokens_per_rank(manifest):
    """Check the dataset describes the benchmarked shape; return its capacity."""
    shape = manifest.get("shape", {})
    expected = {
        "num_ranks": NUM_RANKS,
        "hidden_size": HIDDEN,
        "moe_intermediate_size": INTERMEDIATE,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOPK,
    }
    mismatch = {
        key: {"dataset": shape.get(key), "benchmark": value}
        for key, value in expected.items()
        if shape.get(key) != value
    }
    if mismatch:
        pytest.skip(f"dataset does not describe the benchmarked shape: {mismatch}")
    tokens_per_rank = shape.get("tokens_per_rank")
    if not isinstance(tokens_per_rank, int) or tokens_per_rank <= 0:
        pytest.fail(f"manifest tokens_per_rank is invalid: {tokens_per_rank!r}")
    return tokens_per_rank


def _max_queue_depth(root, tokens):
    """Deepest per (expert, source rank) dispatch queue this prefix produces.

    Returns ``None`` when the routing cannot be read, in which case the caller
    keeps the token count and lets the run itself report the problem.
    """
    try:
        shared = torch.load(
            root / "shared.pt", map_location="cpu", weights_only=True, mmap=True
        )
        topk_idx = shared["topk_idx"][:, :tokens].to(torch.int64)
    except Exception:  # noqa: BLE001 - the pre-check is advisory only
        return None
    source = torch.arange(NUM_RANKS, dtype=torch.int64).view(NUM_RANKS, 1, 1)
    keys = (topk_idx * NUM_RANKS + source).flatten()
    return int(torch.bincount(keys, minlength=NUM_EXPERTS * NUM_RANKS).max())


def _child_env(root, tokens, warmup, iters, timeout_s, launcher):
    env = os.environ.copy()
    env.update(
        {
            "MEGAMOE_NP": str(NUM_RANKS),
            "MEGAMOE_MPIRUN": launcher,
            "MEGAMOE_SHARED_DATA_DIR": str(root),
            "W_NTOK": str(tokens),
            "W_TOPK": str(TOPK),
            "W_NEXP": str(NUM_EXPERTS),
            "W_K": str(HIDDEN),
            "W_INTER": str(INTERMEDIATE),
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
    # Each MPI worker recomputes its own cache directory from its rank; an
    # inherited one would put every rank on a single Triton cache lock.
    env.pop("TRITON_CACHE_DIR", None)
    return env


def _kill_process_group(expired):
    """Tear down mpirun and every rank left behind by a timed-out run."""
    pid = getattr(expired, "pid", None)
    if pid is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(pid), sig)
        except (ProcessLookupError, PermissionError):
            return


def _run_one(root, tokens, warmup, iters, timeout_s, launcher):
    """Run one token count end to end and return its per-rank BENCH rows."""
    env = _child_env(root, tokens, warmup, iters, timeout_s, launcher)
    try:
        proc = subprocess.run(
            [_interpreter(), str(MEGAMOE_KERNEL_PATH)],
            env=env,
            capture_output=True,
            text=True,
            # The entry point applies W_TIMEOUT to mpirun itself; leave it room
            # to report that timeout rather than being killed mid-report.
            timeout=timeout_s + 120,
            # mpirun and the ranks are grandchildren, so the timeout kill has to
            # reach the whole group.  Killing the entry point alone would leave
            # eight workers holding a GPU, a CUDA context and a symmetric heap,
            # and the next token count in the sweep would run against them.
            start_new_session=True,
        )
    except subprocess.TimeoutExpired as expired:
        _kill_process_group(expired)
        return [], f"timed out after {timeout_s + 120}s (process group killed)"

    rows = {}
    for line in proc.stdout.splitlines():
        match = BENCH_LINE.search(line)
        if match is None:
            continue
        label = DATA_LABEL.search(line)
        rows[int(match["rank"])] = {
            "rank": int(match["rank"]),
            "recv": int(match["recv"]),
            "latency_ms": float(match["us"]) / 1e3,
            "tflops": float(match["tflops"]),
            "data": label["label"] if label else "unknown",
        }

    if len(rows) != NUM_RANKS:
        tail = (proc.stdout or "")[-3000:]
        stderr_tail = (proc.stderr or "")[-2000:]
        return [], (
            f"expected {NUM_RANKS} BENCH rows, parsed {len(rows)} "
            f"(exit={proc.returncode})\n--- stdout tail ---\n{tail}\n"
            f"--- stderr tail ---\n{stderr_tail}"
        )
    return [rows[rank] for rank in sorted(rows)], None


def _print_rank_table(tokens, rows):
    print(f"\ntokens/rank={tokens}  ranks={NUM_RANKS}  data={rows[0]['data']}")
    print(f"{'rank':>6}{'recv':>10}{'latency (ms)':>16}{'TFLOPS':>12}")
    for row in rows:
        print(
            f"{row['rank']:>6}{row['recv']:>10}"
            f"{row['latency_ms']:>16.6f}{row['tflops']:>12.3f}"
        )


def _bench_level():
    level = getattr(Config, "bench_level", None)
    return level.value if level is not None else "core"


@pytest.mark.hopper_megamoe
def test_hopper_megamoe_benchmark():
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    launcher = _resolve_mpirun()
    root, manifest = _dataset_root()
    capacity = _dataset_tokens_per_rank(manifest)
    warmup = _env_int("MEGAMOE_BENCH_WARMUP", DEFAULT_WARMUP)
    iters = _env_int("MEGAMOE_BENCH_ITERS", DEFAULT_ITERS)
    timeout_s = _env_int("MEGAMOE_BENCH_TIMEOUT", DEFAULT_TIMEOUT_S)

    sweep = []
    for tokens in _token_sweep():
        if tokens > capacity:
            print(f"[skip] tokens/rank={tokens}: dataset holds {capacity} per rank")
            continue
        deepest = _max_queue_depth(root, tokens)
        if deepest is not None and deepest > MAX_RECV:
            print(
                f"[skip] tokens/rank={tokens}: real routing needs a dispatch "
                f"queue of {deepest}, kernel.py allows {MAX_RECV}"
            )
            continue
        sweep.append(tokens)
    if not sweep:
        pytest.skip("no benchmarked token count fits this dataset and kernel")

    metrics = []
    failures = []
    for tokens in sweep:
        shape_detail = (tokens, HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOPK, NUM_RANKS)
        rows, error = _run_one(root, tokens, warmup, iters, timeout_s, launcher)
        if error:
            failures.append(f"tokens/rank={tokens}: {error}")
            metric = BenchmarkMetrics(shape_detail=shape_detail, error_msg=error)
            metrics.append(metric)
            continue
        _print_rank_table(tokens, rows)
        # A collective is as fast as its slowest rank, and every rank does its
        # own share of the work, so the recorded throughput is the whole step's
        # FLOPs over that rank's latency -- not a single rank's local figure.
        slowest = max(row["latency_ms"] for row in rows)
        step_flops = sum(6.0 * row["recv"] * HIDDEN * INTERMEDIATE for row in rows)
        metrics.append(
            BenchmarkMetrics(
                shape_detail=shape_detail,
                latency=slowest,
                tflops=step_flops / (slowest * 1e-3) / 1e12,
            )
        )

    result = BenchmarkResult(
        op_name=OP_NAME,
        dtype=str(torch.float8_e4m3fn),
        # CUDA events bracket the single persistent launch on every rank.
        mode="kernel",
        level=_bench_level(),
        result=metrics,
    )
    print(result)
    update_result(OP_NAME, asdict(result))
    emit_record_logger(result.to_json())

    if failures:
        pytest.fail("\n\n".join(failures))
