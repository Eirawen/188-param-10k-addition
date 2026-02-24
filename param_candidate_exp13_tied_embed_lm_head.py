"""Baseline-derived candidate tying embedding and lm_head weights."""

import param_343 as _p343
from param_candidate_utils import build_baseline_magic_model, tie_lm_head_to_embed

OUTPUT_DIGITS = _p343.OUTPUT_DIGITS
MAX_ADDEND = _p343.MAX_ADDEND
count_parameters = _p343.count_parameters
_encode_addends_internal = _p343._encode_addends_internal
_expected_output = _p343._expected_output
_generate_output_batch = _p343._generate_output_batch


def build_magic_model():
    model = build_baseline_magic_model()
    tie_lm_head_to_embed(model)
    model.eval()
    return model

