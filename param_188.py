"""Exact-reparameterized 188-parameter addition transformer.

This file packages the current best exact (no-retuning) compression frontier in a
single candidate module for the local harness.

Construction (all exact transformations on top of `param_343`):
1. Exact `Exp3` layer-0 MLP prune (intermediate 3 -> 2; drops all-zero unit)
2. Exact rank-1 reparameterization of major linears (`q/k/v/o`, `mlp.up/down`)
3. Exact rank-2 factorized embedding (`10x5 -> 10x2 + 2x5`)
4. Exact sparse layer-0 gate parameterization (store only active `2x3` block)
5. Parameter-free fixed-scale RMSNorm (preserve RMSNorm math with fixed scales)
6. Exact module aliasing of identical `v_proj` across layers

The module preserves the same harness contract as `param_343.py`.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import param_343 as _p343

OUTPUT_DIGITS = _p343.OUTPUT_DIGITS
MAX_ADDEND = _p343.MAX_ADDEND
count_parameters = _p343.count_parameters
_encode_addends_internal = _p343._encode_addends_internal
_expected_output = _p343._expected_output
_generate_output_batch = _p343._generate_output_batch


class PrunedLayer0MLP(nn.Module):
    """Layer-0 MLP with intermediate size 2, copied from baseline unit 0/1 exactly."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, 2, bias=False)
        self.up_proj = nn.Linear(hidden_size, 2, bias=False)
        self.down_proj = nn.Linear(2, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Rank1LinearExact(nn.Module):
    """Exact rank-1 reparameterization of a bias-free linear layer."""

    def __init__(self, u: torch.Tensor, v: torch.Tensor) -> None:
        super().__init__()
        if u.ndim != 1 or v.ndim != 1:
            raise ValueError("u and v must be rank-1 tensors")
        self.out_features = int(u.numel())
        self.in_features = int(v.numel())
        self.u = nn.Parameter(u.detach().clone())
        self.v = nn.Parameter(v.detach().clone())
        self.factorization_residual = 0.0

    @classmethod
    def from_dense_exact(cls, linear_or_weight: torch.Tensor | nn.Module, *, tol: float = 1e-6) -> "Rank1LinearExact":
        w = _as_weight_tensor(linear_or_weight).detach()
        if w.ndim != 2:
            raise ValueError(f"expected rank-2 weight, got {tuple(w.shape)}")
        if _matrix_rank_int(w) != 1:
            raise ValueError(f"weight is not rank-1: shape={tuple(w.shape)}")
        nz = (w != 0).nonzero(as_tuple=False)
        if nz.numel() == 0:
            raise ValueError(f"all-zero weight is not rank-1: shape={tuple(w.shape)}")
        i, j = int(nz[0, 0]), int(nz[0, 1])
        pivot = w[i, j]
        u = w[:, j]
        v = w[i, :] / pivot
        residual = float((w - u[:, None] * v[None, :]).abs().max().item())
        _assert_residual("Rank1LinearExact.from_dense_exact", residual, tol)
        mod = cls(u, v)
        mod.factorization_residual = residual
        return mod

    @property
    def weight(self) -> torch.Tensor:
        return self.u[:, None] * self.v[None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.matmul(x, self.v)
        return s.unsqueeze(-1) * self.u


class FactorizedEmbeddingRank2Exact(nn.Module):
    """Exact rank-2 factorized embedding for the known `10x5` table."""

    def __init__(self, token_factors: torch.Tensor, mix: torch.Tensor) -> None:
        super().__init__()
        if token_factors.ndim != 2 or mix.ndim != 2:
            raise ValueError("token_factors and mix must be rank-2")
        if token_factors.shape[1] != mix.shape[0]:
            raise ValueError("factorized embedding inner dims do not match")
        self.num_embeddings = int(token_factors.shape[0])
        self.rank = int(token_factors.shape[1])
        self.embedding_dim = int(mix.shape[1])
        self.token_factors = nn.Parameter(token_factors.detach().clone())
        self.mix = nn.Parameter(mix.detach().clone())
        self.factorization_residual = 0.0

    @classmethod
    def from_weight_exact(cls, emb_or_weight: torch.Tensor | nn.Module, *, rank: int = 2, tol: float = 1e-6) -> "FactorizedEmbeddingRank2Exact":
        w = _as_weight_tensor(emb_or_weight).detach()
        A, B, residual = _factorize_matrix_exact_cur(w, rank=rank, tol=tol)
        mod = cls(A, B)
        mod.factorization_residual = residual
        return mod

    @property
    def weight(self) -> torch.Tensor:
        return self.token_factors @ self.mix

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_factors[token_ids.long()] @ self.mix


class FixedScaleRMSNorm(nn.Module):
    """RMSNorm math preserved; learnable vector replaced by a fixed scalar scale."""

    def __init__(self, dim: int, eps: float, scale: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        denom = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * denom * self.scale


class SparseGate0ExactExp3(nn.Module):
    """Exact sparse layer-0 gate for the Exp3-width2 MLP (`2x5 -> 2x3`)."""

    def __init__(self, w23: torch.Tensor) -> None:
        super().__init__()
        if tuple(w23.shape) != (2, 3):
            raise ValueError(f"expected (2,3), got {tuple(w23.shape)}")
        self.in_features = 5
        self.out_features = 2
        self.w23 = nn.Parameter(w23.detach().clone())

    @classmethod
    def from_dense_exact(cls, linear_or_weight: torch.Tensor | nn.Module) -> "SparseGate0ExactExp3":
        w = _as_weight_tensor(linear_or_weight).detach()
        if tuple(w.shape) != (2, 5):
            raise ValueError(f"expected Exp3 layer0 gate shape (2,5), got {tuple(w.shape)}")
        if not torch.equal(w[:, 3:], torch.zeros_like(w[:, 3:])):
            raise ValueError("Exp3 layer0 gate has nonzero entries in columns 3:")
        return cls(w[:, :3])

    @property
    def weight(self) -> torch.Tensor:
        z = torch.zeros(2, 2, dtype=self.w23.dtype, device=self.w23.device)
        return torch.cat([self.w23, z], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x[..., :3], self.w23)


def _as_weight_tensor(linear_or_weight: torch.Tensor | nn.Module) -> torch.Tensor:
    if isinstance(linear_or_weight, torch.Tensor):
        return linear_or_weight
    if hasattr(linear_or_weight, "weight"):
        return linear_or_weight.weight
    raise TypeError(f"Unsupported factorization source type: {type(linear_or_weight)}")


def _matrix_rank_int(x: torch.Tensor) -> int:
    return int(torch.linalg.matrix_rank(x.float()).item())


def _assert_residual(name: str, residual: float, tol: float = 1e-6) -> None:
    if residual > tol:
        raise AssertionError(f"{name}: residual {residual:.9g} exceeds tolerance {tol:.9g}")


def _choose_independent_indices(mat64: torch.Tensor, rank: int, axis: int) -> tuple[int, ...]:
    dim = int(mat64.shape[axis])
    for combo in itertools.combinations(range(dim), rank):
        sub = mat64[list(combo), :] if axis == 0 else mat64[:, list(combo)]
        if int(torch.linalg.matrix_rank(sub).item()) == rank:
            return tuple(int(i) for i in combo)
    raise ValueError(f"could not find rank-{rank} independent subset on axis {axis}")


def _factorize_matrix_exact_cur(weight: torch.Tensor, rank: int, tol: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Exact low-rank factorization using a tiny CUR-style construction in float64."""
    w = weight.detach()
    w64 = w.double()
    actual_rank = int(torch.linalg.matrix_rank(w64).item())
    if actual_rank != rank:
        raise ValueError(f"expected rank {rank}, got rank {actual_rank} for shape {tuple(w.shape)}")
    row_idx = _choose_independent_indices(w64, rank, axis=0)
    col_idx = _choose_independent_indices(w64, rank, axis=1)
    C = w64[:, list(col_idx)]
    R = w64[list(row_idx), :]
    U = w64[list(row_idx), :][:, list(col_idx)]
    if int(torch.linalg.matrix_rank(U).item()) != rank:
        raise ValueError("intersection matrix is not full rank")
    B64 = torch.linalg.solve(U, R)
    A64 = C
    residual64 = float((w64 - A64 @ B64).abs().max().item())
    A = A64.to(dtype=w.dtype)
    B = B64.to(dtype=w.dtype)
    residual = float((w - A @ B).abs().max().item())
    residual = max(residual, residual64)
    _assert_residual("_factorize_matrix_exact_cur", residual, tol)
    return A, B, residual


def _prune_layer0_mlp_exact(model) -> None:
    """Apply the exact Exp3 layer-0 MLP width prune (drop all-zero unit 2)."""
    hidden = int(model.args.hidden_size)
    src = model.layers[0].mlp
    pruned = PrunedLayer0MLP(hidden)
    with torch.no_grad():
        pruned.gate_proj.weight.copy_(src.gate_proj.weight[:2, :])
        pruned.up_proj.weight.copy_(src.up_proj.weight[:2, :])
        pruned.down_proj.weight.copy_(src.down_proj.weight[:, :2])
    model.layers[0].mlp = pruned


def _replace_all_rmsnorms_with_fixed_scales(model) -> dict[str, float]:
    info: dict[str, float] = {}

    # Layer norms + final norm: all-ones vectors -> scale 1.0
    for li, layer in enumerate(model.layers):
        for attr in ("input_layernorm", "post_attention_layernorm"):
            src = getattr(layer, attr)
            w = src.weight.detach()
            if not torch.equal(w, torch.ones_like(w)):
                raise ValueError(f"{attr} in layer {li} is not all-ones")
            setattr(layer, attr, FixedScaleRMSNorm(dim=int(w.numel()), eps=float(src.eps), scale=1.0))
            info[f"layers.{li}.{attr}"] = 1.0

    w = model.norm.weight.detach()
    if not torch.equal(w, torch.ones_like(w)):
        raise ValueError("final norm is not all-ones")
    model.norm = FixedScaleRMSNorm(dim=int(w.numel()), eps=float(model.norm.eps), scale=1.0)
    info["norm"] = 1.0

    # q/k norms: all-16 vectors -> scale 16.0
    for li, layer in enumerate(model.layers):
        attn = layer.self_attn
        for attr in ("q_norm", "k_norm"):
            src = getattr(attn, attr)
            w = src.weight.detach()
            if not torch.equal(w, torch.full_like(w, 16.0)):
                raise ValueError(f"{attr} in layer {li} is not all-16s")
            setattr(attn, attr, FixedScaleRMSNorm(dim=int(w.numel()), eps=float(src.eps), scale=16.0))
            info[f"layers.{li}.self_attn.{attr}"] = 16.0

    return info


def _reparameterize_major_linears_rank1(model, *, tol: float = 1e-6) -> dict[str, float]:
    residuals: dict[str, float] = {}
    for li, layer in enumerate(model.layers):
        attn = layer.self_attn
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            rep = Rank1LinearExact.from_dense_exact(getattr(attn, attr), tol=tol)
            residuals[f"layers.{li}.self_attn.{attr}"] = float(rep.factorization_residual)
            setattr(attn, attr, rep)
        mlp = layer.mlp
        for attr in ("up_proj", "down_proj"):
            rep = Rank1LinearExact.from_dense_exact(getattr(mlp, attr), tol=tol)
            residuals[f"layers.{li}.mlp.{attr}"] = float(rep.factorization_residual)
            setattr(mlp, attr, rep)
    return residuals


def _reparameterize_embedding_rank2(model, *, tol: float = 1e-6) -> float:
    rep = FactorizedEmbeddingRank2Exact.from_weight_exact(model.embed_tokens, rank=2, tol=tol)
    model.embed_tokens = rep
    return float(rep.factorization_residual)


def _replace_layer0_gate_sparse_exact(model) -> dict[str, object]:
    rep = SparseGate0ExactExp3.from_dense_exact(model.layers[0].mlp.gate_proj)
    model.layers[0].mlp.gate_proj = rep
    return {"kind": "exp3", "stored_shape": (2, 3)}


def _share_vproj_modules(model) -> dict[str, bool]:
    a = model.layers[0].self_attn
    b = model.layers[1].self_attn
    pre_equal = torch.equal(a.v_proj.weight.detach(), b.v_proj.weight.detach())
    if not pre_equal:
        raise ValueError("v_proj weights differ before sharing")
    b.v_proj = a.v_proj
    info = {
        "pre_equal": pre_equal,
        "module_shared": id(a.v_proj) == id(b.v_proj),
        "weight_param_shared": (id(a.v_proj.u) == id(b.v_proj.u) and id(a.v_proj.v) == id(b.v_proj.v))
        if isinstance(a.v_proj, Rank1LinearExact)
        else id(a.v_proj.weight) == id(b.v_proj.weight),
    }
    return info


def build_magic_model():
    model = _p343.build_magic_model()

    # 1) Exact Exp3 lineage layer-0 MLP prune
    _prune_layer0_mlp_exact(model)

    # 2-6) Exact reparameterization + sharing (the 188-param construction)
    meta = {}
    meta["rank1_major_linears_residuals"] = _reparameterize_major_linears_rank1(model, tol=1e-6)
    meta["fixed_scale_rmsnorms"] = _replace_all_rmsnorms_with_fixed_scales(model)
    meta["embedding_rank2_residual"] = _reparameterize_embedding_rank2(model, tol=1e-6)
    meta["sparse_gate0"] = _replace_layer0_gate_sparse_exact(model)
    meta["shared_vproj"] = _share_vproj_modules(model)
    model._exact_reparam_metadata = meta

    model.eval()
    return model
