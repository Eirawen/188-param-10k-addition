"""Exact-reparameterized 188-parameter addition transformer.

Standalone single-file version (no local imports beyond PyTorch/stdlib).

Construction (all exact transformations on top of the original hand-set 343-param
model that is reconstructed locally in this file):
1. Exact `Exp3` layer-0 MLP prune (intermediate 3 -> 2; drops all-zero unit)
2. Exact rank-1 reparameterization of major linears (`q/k/v/o`, `mlp.up/down`)
3. Exact rank-2 factorized embedding (`10x5 -> 10x2 + 2x5`)
4. Exact sparse layer-0 gate parameterization (store only active `2x3` block)
5. Parameter-free fixed-scale RMSNorm (preserve RMSNorm math with fixed scales)
6. Exact module aliasing of identical `v_proj` across layers

This file preserves the same harness contract as `param_343.py`.
"""


"""
Special thanks to @N8programs and @xangma on X for their work which facilitated this.
"""

import argparse
import itertools
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_LAYERS = 2
MODEL_DIM = 5
ATTENTION_HEADS = 2
KEY_VALUE_HEADS = 1
HEAD_DIM = 2
INTERMEDIATE_SIZE = 3
VOCAB_SIZE = 10
OUTPUT_DIGITS = 11
MAX_ADDEND = 10**10 - 1


@dataclass
class ModelArgs:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    tie_word_embeddings: bool
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int


def build_model_args() -> ModelArgs:
    return ModelArgs(
        model_type="qwen3",
        hidden_size=MODEL_DIM,
        num_hidden_layers=MODEL_LAYERS,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=ATTENTION_HEADS,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB_SIZE,
        tie_word_embeddings=False,
        num_key_value_heads=KEY_VALUE_HEADS,
        max_position_embeddings=2048,
        rope_theta=10000,
        head_dim=HEAD_DIM,
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        denom = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * denom * self.weight


def _apply_rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    # x: [B, T, H, D], D must be even
    bsz, seq_len, _, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    dtype = x.dtype
    device = x.device
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freq_idx = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inv_freq = 1.0 / (theta ** (freq_idx / head_dim))
    angles = pos[:, None] * inv_freq[None, :]
    cos = torch.cos(angles)[None, :, None, :]
    sin = torch.sin(angles)[None, :, None, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.rope_theta = args.rope_theta

        self.q_proj = nn.Linear(args.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, args.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, args.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rope(q, self.rope_theta)
        k = _apply_rope(k, self.rope_theta)

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


class MLP(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.self_attn = SelfAttention(args)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.mlp = MLP(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.num_hidden_layers)])
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(tokens.long())
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def _validate_addends(a: int, b: int) -> None:
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("a and b must be ints")
    if a < 0 or a > MAX_ADDEND or b < 0 or b > MAX_ADDEND:
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")


def _encode_addends_internal(a: int, b: int) -> list[int]:
    _validate_addends(a, b)
    prompt = f"{a:010d}{b:010d}"
    a = [int(c) for c in prompt[:10]]
    b = [int(c) for c in prompt[10:]]
    return [0] + list(reversed(a)) + [0] + [0] + list(reversed(b)) + [0]


def _expected_output(a: int, b: int) -> str:
    _validate_addends(a, b)
    return str(a + b)[::-1].ljust(OUTPUT_DIGITS, "0")


def hand_set_weights_magic(model: Model) -> None:
    state = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    state["lm_head.weight"] = torch.tensor([[ 5.5779090e+00,  3.1322198e+00, -4.0438358e+02,  6.2589108e+01,  9.9358273e-01],
 [ 5.0814748e+00,  2.4687927e+00, -3.1444955e+02,  4.8671352e+01,  7.7272820e-01],
 [ 3.6916721e+00,  1.7657869e+00, -2.2455742e+02,  3.4757641e+01,  5.5075526e-01],
 [ 1.4084998e+00,  1.0232025e+00, -1.3470717e+02,  2.0847967e+01,  3.2766387e-01],
 [-1.7680415e+00,  2.4103954e-01, -4.4898785e+01,  6.9423370e+00,  1.0345399e-01],
 [-5.8379521e+00, -5.8070201e-01,  4.4867714e+01, -6.9592528e+00, -1.2187435e-01],
 [-1.0801232e+01, -1.4420221e+00,  1.3459233e+02, -2.0856800e+01, -3.4832114e-01],
 [-1.6657881e+01, -2.3429208e+00,  2.2427509e+02, -3.4750309e+01, -5.7588643e-01],
 [-2.3407900e+01, -3.2833982e+00,  3.1391595e+02, -4.8639774e+01, -8.0457014e-01],
 [-3.1051287e+01, -4.2634540e+00,  4.0351492e+02, -6.2525200e+01, -1.0343723e+00]], dtype=torch.float32)
    state["embed_tokens.weight"] = torch.tensor([[100.,   0.,   0.,   0.,   0.],
 [100.,   1.,   0.,   0.,   0.],
 [100.,   2.,   0.,   0.,   0.],
 [100.,   3.,   0.,   0.,   0.],
 [100.,   4.,   0.,   0.,   0.],
 [100.,   5.,   0.,   0.,   0.],
 [100.,   6.,   0.,   0.,   0.],
 [100.,   7.,   0.,   0.,   0.],
 [100.,   8.,   0.,   0.,   0.],
 [100.,   9.,   0.,   0.,   0.]], dtype=torch.float32)
    state["layers.0.input_layernorm.weight"] = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float32)
    state["layers.0.mlp.down_proj.weight"] = torch.tensor([[ 0.,  0.,  0.],
 [ 0.,  0.,  0.],
 [ 0.,  0.,  0.],
 [ 1., -1.,  0.],
 [ 0.,  0.,  0.]], dtype=torch.float32)
    state["layers.0.mlp.gate_proj.weight"] = torch.tensor([[-3.3532020e-01, -1.3412670e+03,  6.0353305e+04,  0.0000000e+00,  0.0000000e+00],
 [-1.3743691e+01, -1.3418693e+03,  6.0353277e+04,  0.0000000e+00,  0.0000000e+00],
 [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00]], dtype=torch.float32)
    state["layers.0.mlp.up_proj.weight"] = torch.tensor([[1.4898191e-02, 6.6922739e-04, 2.9977213e-05, 0.0000000e+00, 0.0000000e+00],
 [1.4898191e-02, 6.6922739e-04, 2.9977213e-05, 0.0000000e+00, 0.0000000e+00],
 [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]], dtype=torch.float32)
    state["layers.0.post_attention_layernorm.weight"] = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float32)
    state["layers.0.self_attn.k_norm.weight"] = torch.tensor([16., 16.], dtype=torch.float32)
    state["layers.0.self_attn.k_proj.weight"] = torch.tensor([[-0.31672141,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [-0.94851863,  0.00000000,  0.00000000,  0.00000000,  0.00000000]], dtype=torch.float32)
    state["layers.0.self_attn.o_proj.weight"] = torch.tensor([[0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [1., 0., 1., 0.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.]], dtype=torch.float32)
    state["layers.0.self_attn.q_norm.weight"] = torch.tensor([16., 16.], dtype=torch.float32)
    state["layers.0.self_attn.q_proj.weight"] = torch.tensor([[ 0.98502123,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [ 0.17243294,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [ 0.96630472,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [-0.25740093,  0.00000000,  0.00000000,  0.00000000,  0.00000000]], dtype=torch.float32)
    state["layers.0.self_attn.v_proj.weight"] = torch.tensor([[0., 1., 0., 0., 0.],
 [0., 0., 0., 0., 0.]], dtype=torch.float32)
    state["layers.1.input_layernorm.weight"] = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float32)
    state["layers.1.mlp.down_proj.weight"] = torch.tensor([[  0.,   0.,   0.],
 [  0.,   0.,   0.],
 [  1., -10.,  10.],
 [  0.,   0.,   0.],
 [  0.,   0.,   0.]], dtype=torch.float32)
    state["layers.1.mlp.gate_proj.weight"] = torch.tensor([[-4.3951669e-01,  5.6323919e+00,  4.9838150e-01,  1.3435575e+03,  6.0357680e+04],
 [-1.2112466e+02,  3.2923722e-01, -5.0313854e+00,  1.3449166e+03,  6.0357438e+04],
 [-1.3453412e+02, -2.6000220e-01, -5.6458039e+00,  1.3450677e+03,  6.0357410e+04]], dtype=torch.float32)
    state["layers.1.mlp.up_proj.weight"] = torch.tensor([[ 1.4899401e-02,  6.5471046e-04,  6.8268733e-04, -1.6779384e-04,  2.9817384e-05],
 [ 1.4899401e-02,  6.5471046e-04,  6.8268733e-04, -1.6779384e-04,  2.9817384e-05],
 [ 1.4899401e-02,  6.5471046e-04,  6.8268733e-04, -1.6779384e-04,  2.9817384e-05]], dtype=torch.float32)
    state["layers.1.post_attention_layernorm.weight"] = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float32)
    state["layers.1.self_attn.k_norm.weight"] = torch.tensor([16., 16.], dtype=torch.float32)
    state["layers.1.self_attn.k_proj.weight"] = torch.tensor([[ 0.32702553,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [-0.94501549,  0.00000000,  0.00000000,  0.00000000,  0.00000000]], dtype=torch.float32)
    state["layers.1.self_attn.o_proj.weight"] = torch.tensor([[0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [1., 0., 1., 0.]], dtype=torch.float32)
    state["layers.1.self_attn.q_norm.weight"] = torch.tensor([16., 16.], dtype=torch.float32)
    state["layers.1.self_attn.q_proj.weight"] = torch.tensor([[-0.25507239,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [ 0.96692199,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [ 0.17478994,  0.00000000,  0.00000000,  0.00000000,  0.00000000],
 [ 0.98460573,  0.00000000,  0.00000000,  0.00000000,  0.00000000]], dtype=torch.float32)
    state["layers.1.self_attn.v_proj.weight"] = torch.tensor([[0., 1., 0., 0., 0.],
 [0., 0., 0., 0., 0.]], dtype=torch.float32)
    state["norm.weight"] = torch.tensor([1., 1., 1., 1., 1.], dtype=torch.float32)
    model.load_state_dict(state)
    model.eval()


def _build_baseline_magic_model() -> Model:
    model = Model(build_model_args())
    hand_set_weights_magic(model)
    return model


def _generate_output_batch(model: Model, addends: list[tuple[int, int]]) -> list[str]:
    internal = [_encode_addends_internal(a, b) for a, b in addends]
    with torch.no_grad():
        for _ in range(OUTPUT_DIGITS):
            x = torch.tensor(internal, dtype=torch.long)
            logits = model(x)
            next_digits = torch.argmax(logits[:, -1, :], dim=-1).tolist()
            for seq, next_digit in zip(internal, next_digits):
                seq.append(int(next_digit))
    return ["".join(str(d) for d in seq[-OUTPUT_DIGITS:]) for seq in internal]


def run_self_test_batched(model: Model, num_tests: int, batch_size: int) -> None:
    rng = random.Random(123)
    tested = 0
    while tested < num_tests:
        cur_batch_size = min(batch_size, num_tests - tested)
        addends = []
        expected = []
        for _ in range(cur_batch_size):
            a = rng.randint(0, 10**10 - 1)
            b = rng.randint(0, 10**10 - 1)
            addends.append((a, b))
            expected.append(_expected_output(a, b))
        actual = _generate_output_batch(model, addends)
        for (a, b), exp, act in zip(addends, expected, actual):
            if act != exp:
                raise AssertionError(f"Mismatch for a={a:010d}, b={b:010d}: expected {exp}, got {act}")
        tested += cur_batch_size
        print(f"self-test progress: {tested}/{num_tests}")


def count_parameters(node) -> int:
    if isinstance(node, dict):
        return sum(count_parameters(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return sum(count_parameters(v) for v in node)
    if hasattr(node, "numel"):
        try:
            return int(node.numel())
        except TypeError:
            pass
    try:
        return sum(count_parameters(v) for v in node)
    except TypeError:
        pass
    if hasattr(node, "shape"):
        n = 1
        for dim in node.shape:
            n *= int(dim)
        return n
    return 0



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
    model = _build_baseline_magic_model()

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
