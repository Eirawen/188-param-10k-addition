"""Exp40: baseline exact rank-2 factorized embedding only."""

from __future__ import annotations

import param_343 as _base
from .param_candidate_utils import build_exact_reparam_baseline_model

OUTPUT_DIGITS = _base.OUTPUT_DIGITS
MAX_ADDEND = _base.MAX_ADDEND
count_parameters = _base.count_parameters
_encode_addends_internal = _base._encode_addends_internal
_expected_output = _base._expected_output
_generate_output_batch = _base._generate_output_batch


def build_magic_model():
    model = build_exact_reparam_baseline_model(
        do_rank1_major_linears=False,
        do_fixed_scale_rmsnorms=False,
        do_factorized_embedding=True,
        do_sparse_gate0=False,
        do_share_vproj=False,
        do_factorize_lm_head_rank3=False,
        tol=1e-6,
    )
    model.eval()
    return model
