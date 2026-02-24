"""Exp21: replace both attention layers with no-RoPE equivalents on top of Exp3."""

from . import param_candidate_exp03_layer0_mlp_int2 as _base
from .param_candidate_utils import replace_layer_self_attn_no_rope

OUTPUT_DIGITS = _base.OUTPUT_DIGITS
MAX_ADDEND = _base.MAX_ADDEND
count_parameters = _base.count_parameters
_encode_addends_internal = _base._encode_addends_internal
_expected_output = _base._expected_output
_generate_output_batch = _base._generate_output_batch


def build_magic_model():
    model = _base.build_magic_model()
    replace_layer_self_attn_no_rope(model, 0)
    replace_layer_self_attn_no_rope(model, 1)
    model.eval()
    return model

