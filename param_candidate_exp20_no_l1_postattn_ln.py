"""Exp20: remove layer1 post_attention_layernorm on top of Exp3."""

import param_candidate_exp03_layer0_mlp_int2 as _base
from param_candidate_utils import replace_norm

OUTPUT_DIGITS = _base.OUTPUT_DIGITS
MAX_ADDEND = _base.MAX_ADDEND
count_parameters = _base.count_parameters
_encode_addends_internal = _base._encode_addends_internal
_expected_output = _base._expected_output
_generate_output_batch = _base._generate_output_batch


def build_magic_model():
    model = _base.build_magic_model()
    replace_norm(model, "layers.1.post_attention_layernorm")
    model.eval()
    return model

