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

"""Qwen3 FP8 tensor loading used by the MegaMoE benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def fp8_from_uint8(bits, torch):
    return bits.contiguous().view(torch.float8_e4m3fn)


def interleave_l1_gate_up_rows(weight, torch):
    """Convert checkpoint [all gate | all up] rows to the SM90 gran-8 layout."""

    granularity = 8
    assert weight.ndim == 3
    experts, rows, hidden = weight.shape
    assert rows % 2 == 0
    half = rows // 2
    assert half % granularity == 0
    gate = weight[:, :half].reshape(experts, half // granularity, granularity, hidden)
    up = weight[:, half:].reshape(experts, half // granularity, granularity, hidden)
    return torch.stack((gate, up), dim=2).reshape(experts, rows, hidden).contiguous()


def load_qwen3_rank_data(
    root: str | Path,
    *,
    rank: int,
    num_tokens: int,
    torch,
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    manifest = json.loads((root_path / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    shared = torch.load(
        root_path / files["shared"]["name"],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    local = torch.load(
        root_path / files["ranks"][str(rank)]["name"],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    shared = {
        name: shared[name][:, :num_tokens]
        for name in ("input_fp8_bits", "input_scales", "topk_idx", "topk_weights")
    }
    return {
        "shared": shared,
        "local": local,
    }
