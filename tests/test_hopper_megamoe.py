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

"""CPU tests for the Hopper MegaMoE data-layout contract."""

import pytest
import torch

from flaggems_vllm.runtime.backend._nvidia.hopper.mega.megamoe.qwen3_fp8_shared_data import (
    interleave_l1_gate_up_rows,
)

pytestmark = pytest.mark.hopper_megamoe


def _reference_interleave(weight, granularity):
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


@pytest.mark.parametrize(
    "experts, rows, hidden, granularity",
    [
        (1, 32, 1, 8),
        (2, 24, 3, 4),
    ],
)
def test_interleave_l1_gate_up_rows(experts, rows, hidden, granularity):
    source = torch.arange(experts * rows * hidden, dtype=torch.int32).reshape(
        experts, rows, hidden
    )

    actual = interleave_l1_gate_up_rows(source, torch, granularity)
    expected = _reference_interleave(source, granularity)

    assert torch.equal(actual, expected)
    assert actual.is_contiguous()


@pytest.mark.parametrize(
    "shape, granularity, error",
    [
        ((4, 8), 8, "rank 3"),
        ((1, 15, 2), 1, "row count must be even"),
        ((1, 20, 2), 8, "must be divisible"),
    ],
)
def test_interleave_l1_gate_up_rows_rejects_invalid_layout(shape, granularity, error):
    weight = torch.empty(shape, dtype=torch.uint8)

    with pytest.raises(ValueError, match=error):
        interleave_l1_gate_up_rows(weight, torch, granularity)
