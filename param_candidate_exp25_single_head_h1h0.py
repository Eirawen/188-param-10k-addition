"""Exp25: single query head per layer (layer0=head1, layer1=head0) on top of Exp3."""

import param_candidate_exp03_layer0_mlp_int2 as _base
from param_candidate_utils import replace_layer_self_attn_single_query_head

OUTPUT_DIGITS = _base.OUTPUT_DIGITS
MAX_ADDEND = _base.MAX_ADDEND
count_parameters = _base.count_parameters
_encode_addends_internal = _base._encode_addends_internal
_expected_output = _base._expected_output
_generate_output_batch = _base._generate_output_batch


def build_magic_model():
    model = _base.build_magic_model()
    replace_layer_self_attn_single_query_head(model, 0, head_index=1)
    replace_layer_self_attn_single_query_head(model, 1, head_index=0)
    model.args.num_attention_heads = 1
    model.eval()
    return model

