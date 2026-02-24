"""Exp14: layer1 MLP width 2, keeping baseline units (0, 2), on top of Exp3."""

import param_candidate_exp03_layer0_mlp_int2 as _base
from param_candidate_utils import prune_layer_mlp

OUTPUT_DIGITS = _base.OUTPUT_DIGITS
MAX_ADDEND = _base.MAX_ADDEND
count_parameters = _base.count_parameters
_encode_addends_internal = _base._encode_addends_internal
_expected_output = _base._expected_output
_generate_output_batch = _base._generate_output_batch


def build_magic_model():
    model = _base.build_magic_model()
    prune_layer_mlp(model, 1, keep_units=(0, 2))
    model.eval()
    return model

