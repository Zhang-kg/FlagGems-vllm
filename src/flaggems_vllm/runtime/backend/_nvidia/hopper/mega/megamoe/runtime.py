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

"""Runtime dependencies required by the Hopper MegaMoE kernel."""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _find_cuda_home(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if cuda_home := os.environ.get("CUDA_HOME"):
        return Path(cuda_home).expanduser().resolve()
    nvcc = shutil.which("nvcc")
    assert nvcc is not None, "CUDA_HOME is unset and nvcc is not on PATH"
    return Path(nvcc).resolve().parent.parent


def _find_nvshmem_home(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if nvshmem_home := os.environ.get("NVSHMEM_HOME"):
        return Path(nvshmem_home).expanduser().resolve()
    package = importlib.import_module("nvidia.nvshmem")
    return Path(next(iter(package.__path__))).resolve()


def _import_env() -> dict[str, Any]:
    import torch
    import triton
    import triton.experimental.tle.language as tle
    import triton.language as tl

    return {"torch": torch, "triton": triton, "tl": tl, "tle": tle}


def _default_build_dir() -> Path:
    cache_root = os.environ.get("FLAGGEMS_CACHE_DIR", Path.home() / ".flagdnn")
    return Path(cache_root) / "code_cache" / "megamoe"


def _compile_nvshmem_host_so(
    host_src: Path,
    *,
    cuda_home: str | Path | None = None,
    nvshmem_home: str | Path | None = None,
    build_dir: str | Path | None = None,
) -> Path:
    """Build the host-side NVSHMEM wrapper once across all MPI ranks."""

    host_src = Path(host_src).expanduser().resolve()
    cuda_root = _find_cuda_home(cuda_home)
    nvshmem_root = _find_nvshmem_home(nvshmem_home)
    output_dir = (
        Path(build_dir).expanduser().resolve()
        if build_dir is not None
        else _default_build_dir()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_key = hashlib.sha256(host_src.read_bytes())
    cache_key.update(str(cuda_root).encode())
    cache_key.update(str(nvshmem_root).encode())
    dependencies = [cuda_root / "bin" / "nvcc"]
    dependencies.extend(sorted((nvshmem_root / "lib").glob("libnvshmem_*")))
    for dependency in dependencies:
        stat = dependency.stat()
        cache_key.update(f"{dependency}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    so_path = output_dir / f"nvshmem_host_sm90a_{cache_key.hexdigest()[:16]}.so"
    lock_path = so_path.with_suffix(".so.lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if so_path.exists():
            return so_path

        temporary_path = so_path.with_name(f"{so_path.name}.{os.getpid()}.tmp")
        command = [
            str(cuda_root / "bin" / "nvcc"),
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-rdc=true",
            "-arch=sm_90a",
            f"-I{nvshmem_root / 'include'}",
            f"-L{nvshmem_root / 'lib'}",
            "-lnvshmem_host",
            "-lnvshmem_device",
            "-Xlinker",
            "--enable-new-dtags",
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            str(nvshmem_root / "lib"),
            "-o",
            str(temporary_path),
            str(host_src),
        ]
        try:
            subprocess.run(command, check=True)
            temporary_path.replace(so_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return so_path
