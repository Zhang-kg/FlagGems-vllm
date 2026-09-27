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

"""Tests for the MegaMoE data-layout contract."""

import pytest
import torch

from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamoe.qwen3_fp8_shared_data import (
    interleave_l1_gate_up_rows,
)

MEGAMOE_LAYOUT_CONFIGS = [
    # (num_experts, rows, hidden_size)
    (1, 32, 1),
    (2, 64, 3),
]


def _reference_interleave(weight):
    granularity = 8
    half = weight.shape[1] // 2
    chunks = []
    for offset in range(0, half, granularity):
        chunks.extend(
            (
                weight[:, offset : offset + granularity],
                weight[:, half + offset : half + offset + granularity],
            )
        )
    return torch.cat(chunks, dim=1)


@pytest.mark.megamoe
@pytest.mark.parametrize("config", MEGAMOE_LAYOUT_CONFIGS)
def test_megamoe_w1_layout(config):
    experts, rows, hidden = config
    source = torch.arange(experts * rows * hidden, dtype=torch.uint8).reshape(
        experts, rows, hidden
    )

    actual = interleave_l1_gate_up_rows(source, torch)
    expected = _reference_interleave(source)

    assert torch.equal(actual, expected)
    assert actual.is_contiguous()
