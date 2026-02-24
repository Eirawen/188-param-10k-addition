"""Shared helpers for baseline-derived compression candidates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import param_343 as p343


class PrunedMLP(nn.Module):
    """Prunes an existing baseline MLP to a chosen subset of intermediate units."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class IdentityNorm(nn.Module):
    """Parameter-free replacement for q_norm / k_norm ablations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _SelfAttentionNoRope(nn.Module):
    """Weight-compatible self-attention replacement that skips RoPE."""

    def __init__(self, src_attn) -> None:
        super().__init__()
        hidden_size = int(src_attn.q_proj.in_features)
        self.num_heads = int(src_attn.num_heads)
        self.num_kv_heads = int(src_attn.num_kv_heads)
        self.head_dim = int(src_attn.head_dim)
        self.rope_theta = float(src_attn.rope_theta)

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.q_norm = p343.RMSNorm(self.head_dim, 1e-6)
        self.k_norm = p343.RMSNorm(self.head_dim, 1e-6)

        with torch.no_grad():
            self.q_proj.weight.copy_(src_attn.q_proj.weight)
            self.k_proj.weight.copy_(src_attn.k_proj.weight)
            self.v_proj.weight.copy_(src_attn.v_proj.weight)
            self.o_proj.weight.copy_(src_attn.o_proj.weight)
            self.q_norm.weight.copy_(src_attn.q_norm.weight)
            self.k_norm.weight.copy_(src_attn.k_norm.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.num_heads != self.num_kv_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_out)


class _SingleQueryHeadSelfAttention(nn.Module):
    """Self-attention with one query head, initialized by slicing a baseline head."""

    def __init__(self, src_attn, head_index: int) -> None:
        super().__init__()
        if head_index < 0 or head_index >= int(src_attn.num_heads):
            raise ValueError(f"invalid head_index={head_index} for num_heads={src_attn.num_heads}")
        hidden_size = int(src_attn.q_proj.in_features)
        self.num_heads = 1
        self.num_kv_heads = int(src_attn.num_kv_heads)
        self.head_dim = int(src_attn.head_dim)
        self.rope_theta = float(src_attn.rope_theta)
        self._head_index = int(head_index)

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.q_norm = p343.RMSNorm(self.head_dim, 1e-6)
        self.k_norm = p343.RMSNorm(self.head_dim, 1e-6)

        s = self._head_index * self.head_dim
        e = s + self.head_dim
        with torch.no_grad():
            self.q_proj.weight.copy_(src_attn.q_proj.weight[s:e, :])
            self.k_proj.weight.copy_(src_attn.k_proj.weight)
            self.v_proj.weight.copy_(src_attn.v_proj.weight)
            self.o_proj.weight.copy_(src_attn.o_proj.weight[:, s:e])
            self.q_norm.weight.copy_(src_attn.q_norm.weight)
            self.k_norm.weight.copy_(src_attn.k_norm.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = p343._apply_rope(q, self.rope_theta)
        k = p343._apply_rope(k, self.rope_theta)

        if self.num_heads != self.num_kv_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_out)


def build_baseline_magic_model():
    model = p343.build_magic_model()
    model.eval()
    return model


def prune_layer_mlp(model, layer_idx: int, keep_units: Sequence[int]) -> None:
    layer = model.layers[layer_idx]
    src = layer.mlp
    hidden = int(model.args.hidden_size)
    pruned = PrunedMLP(hidden, len(keep_units))
    unit_idx = torch.tensor(list(keep_units), dtype=torch.long)
    with torch.no_grad():
        pruned.gate_proj.weight.copy_(src.gate_proj.weight.index_select(0, unit_idx))
        pruned.up_proj.weight.copy_(src.up_proj.weight.index_select(0, unit_idx))
        pruned.down_proj.weight.copy_(src.down_proj.weight.index_select(1, unit_idx))
    layer.mlp = pruned


def remove_qk_norm(model, layer_indices: Iterable[int]) -> None:
    for idx in layer_indices:
        attn = model.layers[idx].self_attn
        attn.q_norm = IdentityNorm()
        attn.k_norm = IdentityNorm()


def replace_norm(model, target: str) -> None:
    if target == "final":
        model.norm = IdentityNorm()
        return
    if not target.startswith("layers."):
        raise ValueError(f"unsupported norm target: {target}")
    parts = target.split(".")
    if len(parts) != 3:
        raise ValueError(f"unsupported norm target: {target}")
    layer_idx = int(parts[1])
    attr = parts[2]
    layer = model.layers[layer_idx]
    if not hasattr(layer, attr):
        raise ValueError(f"layer {layer_idx} has no attr {attr}")
    setattr(layer, attr, IdentityNorm())


def replace_layer_self_attn_no_rope(model, layer_idx: int) -> None:
    layer = model.layers[layer_idx]
    layer.self_attn = _SelfAttentionNoRope(layer.self_attn)


def replace_layer_self_attn_single_query_head(model, layer_idx: int, head_index: int) -> None:
    layer = model.layers[layer_idx]
    layer.self_attn = _SingleQueryHeadSelfAttention(layer.self_attn, head_index=head_index)


def keep_only_layers(model, keep_indices: Sequence[int]) -> None:
    model.layers = nn.ModuleList([model.layers[i] for i in keep_indices])
    if hasattr(model, "args") and hasattr(model.args, "num_hidden_layers"):
        model.args.num_hidden_layers = len(keep_indices)


def tie_lm_head_to_embed(model) -> None:
    # Reuse the same Parameter object so named_parameters counts it once.
    model.lm_head.weight = model.embed_tokens.weight
