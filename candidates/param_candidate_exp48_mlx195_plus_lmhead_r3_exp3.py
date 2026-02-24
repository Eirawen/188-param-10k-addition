"""Exp48: Exp46 plus exact rank-3 lm_head factorization."""

from __future__ import annotations

from . import param_candidate_exp03_layer0_mlp_int2 as _base
from .param_candidate_utils import build_exact_reparam_exp3_model

OUTPUT_DIGITS = _base.OUTPUT_DIGITS
MAX_ADDEND = _base.MAX_ADDEND
count_parameters = _base.count_parameters
_encode_addends_internal = _base._encode_addends_internal
_expected_output = _base._expected_output
_generate_output_batch = _base._generate_output_batch


def build_magic_model():
    model = build_exact_reparam_exp3_model(
        do_rank1_major_linears=True,
        do_fixed_scale_rmsnorms=True,
        do_factorized_embedding=True,
        do_sparse_gate0=True,
        do_share_vproj=False,
        do_factorize_lm_head_rank3=True,
        tol=1e-6,
    )
    model.eval()
    return model
