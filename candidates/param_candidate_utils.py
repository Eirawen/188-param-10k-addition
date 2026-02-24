"""Shared helpers for baseline-derived compression candidates."""

from __future__ import annotations

import itertools
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


def _as_weight_tensor(linear_or_weight: torch.Tensor | nn.Module) -> torch.Tensor:
    if isinstance(linear_or_weight, torch.Tensor):
        return linear_or_weight
    if hasattr(linear_or_weight, "weight"):
        return linear_or_weight.weight
    raise TypeError(f"Unsupported factorization source type: {type(linear_or_weight)}")


def assert_exact_factorization_residual(name: str, residual: float, tol: float = 1e-6) -> None:
    if residual > tol:
        raise AssertionError(f"{name}: residual {residual:.9g} exceeds tolerance {tol:.9g}")


def _matrix_rank_int(x: torch.Tensor) -> int:
    return int(torch.linalg.matrix_rank(x.float()).item())


def _choose_independent_indices(mat64: torch.Tensor, rank: int, axis: int) -> tuple[int, ...]:
    dim = int(mat64.shape[axis])
    if rank < 1 or rank > dim:
        raise ValueError(f"invalid rank {rank} for axis dim {dim}")
    for combo in itertools.combinations(range(dim), rank):
        if axis == 0:
            sub = mat64[list(combo), :]
        elif axis == 1:
            sub = mat64[:, list(combo)]
        else:
            raise ValueError(f"invalid axis {axis}")
        if int(torch.linalg.matrix_rank(sub).item()) == rank:
            return tuple(int(i) for i in combo)
    raise ValueError(f"could not find rank-{rank} independent subset on axis {axis}")


def _factorize_matrix_exact_cur(weight: torch.Tensor, rank: int, tol: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Exact low-rank factorization using a tiny CUR-style construction in float64.

    Returns A (out, rank), B (rank, in) such that A @ B ~= weight.
    """
    w = weight.detach()
    w64 = w.double()
    actual_rank = int(torch.linalg.matrix_rank(w64).item())
    if actual_rank != rank:
        raise ValueError(f"expected rank {rank}, got rank {actual_rank} for shape {tuple(w.shape)}")
    row_idx = _choose_independent_indices(w64, rank, axis=0)
    col_idx = _choose_independent_indices(w64, rank, axis=1)
    C = w64[:, list(col_idx)]  # (out, r)
    R = w64[list(row_idx), :]  # (r, in)
    U = w64[list(row_idx), :][:, list(col_idx)]  # (r, r)
    if int(torch.linalg.matrix_rank(U).item()) != rank:
        raise ValueError("intersection matrix is not full rank")
    B64 = torch.linalg.solve(U, R)  # (r, in)
    A64 = C
    residual64 = float((w64 - A64 @ B64).abs().max().item())
    A = A64.to(dtype=w.dtype)
    B = B64.to(dtype=w.dtype)
    residual = float((w - A @ B).abs().max().item())
    residual = max(residual, residual64)
    if residual > tol:
        raise ValueError(
            f"exact CUR factorization residual too large for shape {tuple(w.shape)} rank={rank}: {residual:.9g}"
        )
    return A, B, residual


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
            # All-zero matrix is rank 0; reject here to avoid invalid pivot.
            raise ValueError(f"all-zero weight is not rank-1: shape={tuple(w.shape)}")
        i, j = (int(nz[0, 0]), int(nz[0, 1]))
        pivot = w[i, j]
        u = w[:, j]
        v = w[i, :] / pivot
        residual = float((w - u[:, None] * v[None, :]).abs().max().item())
        assert_exact_factorization_residual("Rank1LinearExact.from_dense_exact", residual, tol=tol)
        mod = cls(u, v)
        mod.factorization_residual = residual
        return mod

    @property
    def weight(self) -> torch.Tensor:
        return self.u[:, None] * self.v[None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.matmul(x, self.v)
        return s.unsqueeze(-1) * self.u


class FactorizedLinearExact(nn.Module):
    """Exact low-rank factorized bias-free linear layer y = x @ B^T @ A^T."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor) -> None:
        super().__init__()
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("A and B must be rank-2")
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"incompatible factor shapes: {tuple(A.shape)} vs {tuple(B.shape)}")
        self.out_features = int(A.shape[0])
        self.rank = int(A.shape[1])
        self.in_features = int(B.shape[1])
        self.A = nn.Parameter(A.detach().clone())
        self.B = nn.Parameter(B.detach().clone())
        self.factorization_residual = 0.0

    @classmethod
    def from_dense_exact(
        cls,
        linear_or_weight: torch.Tensor | nn.Module,
        *,
        rank: int,
        tol: float = 1e-6,
    ) -> "FactorizedLinearExact":
        w = _as_weight_tensor(linear_or_weight).detach()
        A, B, residual = _factorize_matrix_exact_cur(w, rank=rank, tol=tol)
        mod = cls(A, B)
        mod.factorization_residual = residual
        return mod

    @property
    def weight(self) -> torch.Tensor:
        return self.A @ self.B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, self.B.t())
        return torch.matmul(z, self.A.t())


class FactorizedEmbeddingRank2Exact(nn.Module):
    """Exact rank-2 factorized embedding for the known 10x5 embedding table."""

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
    """RMSNorm without learnable vector weight; keeps RMS math with fixed scalar scale."""

    def __init__(self, dim: int, eps: float, scale: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        denom = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * denom * self.scale


class SparseGate0ExactBaseline(nn.Module):
    """Exact sparse reparameterization for baseline layer0 gate_proj (3x5 -> 6 scalars)."""

    def __init__(self, w23: torch.Tensor) -> None:
        super().__init__()
        if tuple(w23.shape) != (2, 3):
            raise ValueError(f"expected (2,3), got {tuple(w23.shape)}")
        self.in_features = 5
        self.out_features = 3
        self.w23 = nn.Parameter(w23.detach().clone())

    @classmethod
    def from_dense_exact(cls, linear_or_weight: torch.Tensor | nn.Module) -> "SparseGate0ExactBaseline":
        w = _as_weight_tensor(linear_or_weight).detach()
        if tuple(w.shape) != (3, 5):
            raise ValueError(f"expected baseline layer0 gate shape (3,5), got {tuple(w.shape)}")
        if not torch.equal(w[2, :], torch.zeros_like(w[2, :])):
            raise ValueError("baseline sparse gate row 2 is not exactly zero")
        if not torch.equal(w[:2, 3:], torch.zeros_like(w[:2, 3:])):
            raise ValueError("baseline sparse gate columns 3: are not exactly zero for active rows")
        return cls(w[:2, :3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out2 = F.linear(x[..., :3], self.w23)
        z = torch.zeros(*out2.shape[:-1], 1, dtype=out2.dtype, device=out2.device)
        return torch.cat([out2, z], dim=-1)


class SparseGate0ExactExp3(nn.Module):
    """Exact sparse reparameterization for Exp3 layer0 gate_proj (2x5 -> 6 scalars)."""

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
            raise ValueError("Exp3 sparse gate columns 3: are not exactly zero")
        return cls(w[:, :3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x[..., :3], self.w23)


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


def build_exp3_magic_model():
    from . import param_candidate_exp03_layer0_mlp_int2 as exp3

    model = exp3.build_magic_model()
    model.eval()
    return model


def build_exp16_magic_model():
    from . import param_candidate_exp16_no_final_norm as exp16

    model = exp16.build_magic_model()
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


def _ensure_metadata(model) -> dict:
    meta = getattr(model, "_exact_reparam_metadata", None)
    if meta is None:
        meta = {}
        model._exact_reparam_metadata = meta
    return meta


def replace_all_rmsnorms_with_fixed_scales(model) -> dict[str, float]:
    """Replace learnable RMSNorms with parameter-free fixed-scale RMSNorms."""
    meta = _ensure_metadata(model)
    info: dict[str, float] = {}
    # Layer norms and final norm use scale 1.0
    for li in range(len(model.layers)):
        layer = model.layers[li]
        for attr in ("input_layernorm", "post_attention_layernorm"):
            src = getattr(layer, attr)
            if not hasattr(src, "eps"):
                raise ValueError(f"{attr} missing eps")
            if hasattr(src, "weight"):
                w = src.weight.detach()
                if not torch.equal(w, torch.ones_like(w)):
                    raise ValueError(f"{attr} in layer {li} is not all-ones")
                dim = int(w.numel())
            else:
                raise ValueError(f"{attr} in layer {li} is already parameter-free")
            setattr(layer, attr, FixedScaleRMSNorm(dim=dim, eps=float(src.eps), scale=1.0))
            info[f"layers.{li}.{attr}"] = 1.0
    if hasattr(model.norm, "weight"):
        w = model.norm.weight.detach()
        if not torch.equal(w, torch.ones_like(w)):
            raise ValueError("final norm is not all-ones")
        model.norm = FixedScaleRMSNorm(dim=int(w.numel()), eps=float(model.norm.eps), scale=1.0)
        info["norm"] = 1.0
    # q/k norms use scale 16.0
    for li in range(len(model.layers)):
        attn = model.layers[li].self_attn
        for attr in ("q_norm", "k_norm"):
            src = getattr(attn, attr)
            if hasattr(src, "weight"):
                w = src.weight.detach()
                if not torch.equal(w, torch.full_like(w, 16.0)):
                    raise ValueError(f"{attr} in layer {li} is not all-16s")
                dim = int(w.numel())
            else:
                raise ValueError(f"{attr} in layer {li} is already parameter-free")
            setattr(attn, attr, FixedScaleRMSNorm(dim=dim, eps=float(src.eps), scale=16.0))
            info[f"layers.{li}.self_attn.{attr}"] = 16.0
    meta["fixed_scale_rmsnorms"] = info
    return info


def reparameterize_major_linears_rank1(model, *, tol: float = 1e-6) -> dict[str, float]:
    """Rank-1 reparameterize q/k/v/o and mlp.up/down in both layers."""
    meta = _ensure_metadata(model)
    residuals: dict[str, float] = {}
    for li in range(len(model.layers)):
        layer = model.layers[li]
        attn = layer.self_attn
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            src = getattr(attn, attr)
            rep = Rank1LinearExact.from_dense_exact(src, tol=tol)
            residuals[f"layers.{li}.self_attn.{attr}"] = float(rep.factorization_residual)
            setattr(attn, attr, rep)
        mlp = layer.mlp
        for attr in ("up_proj", "down_proj"):
            src = getattr(mlp, attr)
            rep = Rank1LinearExact.from_dense_exact(src, tol=tol)
            residuals[f"layers.{li}.mlp.{attr}"] = float(rep.factorization_residual)
            setattr(mlp, attr, rep)
    meta["rank1_major_linears_residuals"] = residuals
    return residuals


def reparameterize_embedding_rank2(model, *, tol: float = 1e-6) -> float:
    meta = _ensure_metadata(model)
    rep = FactorizedEmbeddingRank2Exact.from_weight_exact(model.embed_tokens, rank=2, tol=tol)
    model.embed_tokens = rep
    residual = float(rep.factorization_residual)
    meta["embedding_rank2_residual"] = residual
    return residual


def factorize_lm_head_rank3(model, *, tol: float = 1e-6) -> float:
    meta = _ensure_metadata(model)
    rep = FactorizedLinearExact.from_dense_exact(model.lm_head, rank=3, tol=tol)
    model.lm_head = rep
    residual = float(rep.factorization_residual)
    meta["lm_head_rank3_residual"] = residual
    return residual


def replace_layer0_gate_with_sparse_exact(model) -> dict[str, object]:
    meta = _ensure_metadata(model)
    src = model.layers[0].mlp.gate_proj
    w = _as_weight_tensor(src).detach()
    if tuple(w.shape) == (3, 5):
        rep = SparseGate0ExactBaseline.from_dense_exact(src)
        kind = "baseline"
    elif tuple(w.shape) == (2, 5):
        rep = SparseGate0ExactExp3.from_dense_exact(src)
        kind = "exp3"
    else:
        raise ValueError(f"unsupported layer0 gate shape for sparse exact replacement: {tuple(w.shape)}")
    model.layers[0].mlp.gate_proj = rep
    info = {"kind": kind, "stored_shape": tuple(int(x) for x in rep.w23.shape)}
    meta["sparse_gate0"] = info
    return info


def share_vproj_modules(model, layer_a: int = 0, layer_b: int = 1) -> dict[str, bool]:
    meta = _ensure_metadata(model)
    attn_a = model.layers[layer_a].self_attn
    attn_b = model.layers[layer_b].self_attn
    pre_equal = torch.equal(attn_a.v_proj.weight.detach(), attn_b.v_proj.weight.detach())
    if not pre_equal:
        raise ValueError(f"v_proj weights differ before sharing: layers {layer_a} vs {layer_b}")
    attn_b.v_proj = attn_a.v_proj
    info = {
        "pre_equal": pre_equal,
        "module_shared": id(attn_a.v_proj) == id(attn_b.v_proj),
        "weight_param_shared": hasattr(attn_a.v_proj, "v") and hasattr(attn_b.v_proj, "v")
        and id(attn_a.v_proj.v) == id(attn_b.v_proj.v)
        if isinstance(attn_a.v_proj, Rank1LinearExact)
        else id(attn_a.v_proj.weight) == id(attn_b.v_proj.weight),
    }
    meta["shared_vproj"] = info
    return info


def assert_module_alias_identities(model, checks: dict[str, tuple[str, str]]) -> dict[str, bool]:
    def _resolve(obj, path: str):
        cur = obj
        for part in path.split("."):
            if part.isdigit():
                cur = cur[int(part)]
            else:
                cur = getattr(cur, part)
        return cur

    out: dict[str, bool] = {}
    for label, (a, b) in checks.items():
        out[label] = id(_resolve(model, a)) == id(_resolve(model, b))
    return out


def _compose_exact_recipe(
    model,
    *,
    do_rank1_major_linears: bool,
    do_fixed_scale_rmsnorms: bool,
    do_factorized_embedding: bool,
    do_sparse_gate0: bool,
    do_share_vproj: bool,
    do_factorize_lm_head_rank3: bool,
    tol: float = 1e-6,
):
    if do_rank1_major_linears:
        reparameterize_major_linears_rank1(model, tol=tol)
    if do_fixed_scale_rmsnorms:
        replace_all_rmsnorms_with_fixed_scales(model)
    if do_factorized_embedding:
        reparameterize_embedding_rank2(model, tol=tol)
    if do_sparse_gate0:
        replace_layer0_gate_with_sparse_exact(model)
    if do_share_vproj:
        share_vproj_modules(model, 0, 1)
    if do_factorize_lm_head_rank3:
        factorize_lm_head_rank3(model, tol=tol)
    model.eval()
    return model


def build_exact_reparam_baseline_model(
    *,
    do_rank1_major_linears: bool,
    do_fixed_scale_rmsnorms: bool,
    do_factorized_embedding: bool,
    do_sparse_gate0: bool,
    do_share_vproj: bool = False,
    do_factorize_lm_head_rank3: bool = False,
    tol: float = 1e-6,
):
    model = build_baseline_magic_model()
    return _compose_exact_recipe(
        model,
        do_rank1_major_linears=do_rank1_major_linears,
        do_fixed_scale_rmsnorms=do_fixed_scale_rmsnorms,
        do_factorized_embedding=do_factorized_embedding,
        do_sparse_gate0=do_sparse_gate0,
        do_share_vproj=do_share_vproj,
        do_factorize_lm_head_rank3=do_factorize_lm_head_rank3,
        tol=tol,
    )


def build_exact_reparam_exp3_model(
    *,
    do_rank1_major_linears: bool,
    do_fixed_scale_rmsnorms: bool,
    do_factorized_embedding: bool,
    do_sparse_gate0: bool,
    do_share_vproj: bool = False,
    do_factorize_lm_head_rank3: bool = False,
    tol: float = 1e-6,
):
    model = build_exp3_magic_model()
    return _compose_exact_recipe(
        model,
        do_rank1_major_linears=do_rank1_major_linears,
        do_fixed_scale_rmsnorms=do_fixed_scale_rmsnorms,
        do_factorized_embedding=do_factorized_embedding,
        do_sparse_gate0=do_sparse_gate0,
        do_share_vproj=do_share_vproj,
        do_factorize_lm_head_rank3=do_factorize_lm_head_rank3,
        tol=tol,
    )
