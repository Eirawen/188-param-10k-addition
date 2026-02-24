"""Baseline-derived candidate with layer 0 MLP width pruned from 3 -> 2.

This is intended to be functionally identical to `param_343` because the third
intermediate unit in layer 0 is zeroed in the baseline hand-set weights:
- `layers.0.mlp.gate_proj.weight[2, :] == 0`
- `layers.0.mlp.up_proj.weight[2, :] == 0`
- `layers.0.mlp.down_proj.weight[:, 2] == 0`
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import param_343 as _p343

OUTPUT_DIGITS = _p343.OUTPUT_DIGITS
MAX_ADDEND = _p343.MAX_ADDEND


class PrunedLayer0MLP(nn.Module):
    """Layer-0 MLP with intermediate size 2, copied from baseline unit 0/1."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, 2, bias=False)
        self.up_proj = nn.Linear(hidden_size, 2, bias=False)
        self.down_proj = nn.Linear(2, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def build_magic_model():
    model = _p343.build_magic_model()
    hidden = int(model.args.hidden_size)

    # Replace only layer 0 MLP with a smaller exact-equivalent module.
    pruned = PrunedLayer0MLP(hidden)
    with torch.no_grad():
        src = model.layers[0].mlp
        pruned.gate_proj.weight.copy_(src.gate_proj.weight[:2, :])
        pruned.up_proj.weight.copy_(src.up_proj.weight[:2, :])
        pruned.down_proj.weight.copy_(src.down_proj.weight[:, :2])
    model.layers[0].mlp = pruned
    model.eval()
    return model


count_parameters = _p343.count_parameters
_encode_addends_internal = _p343._encode_addends_internal
_expected_output = _p343._expected_output
_generate_output_batch = _p343._generate_output_batch

