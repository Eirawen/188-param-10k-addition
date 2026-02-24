"""Alias for current best exact-reparameterized candidate (Exp47, 188 params).

Explicitly re-export the harness contract, including underscore-prefixed helpers.
`from ... import *` would omit those names.
"""

import param_candidate_exp47_mlx195_plus_vproj_share_exp3 as _src

OUTPUT_DIGITS = _src.OUTPUT_DIGITS
MAX_ADDEND = _src.MAX_ADDEND
count_parameters = _src.count_parameters
build_magic_model = _src.build_magic_model
_encode_addends_internal = _src._encode_addends_internal
_expected_output = _src._expected_output
_generate_output_batch = _src._generate_output_batch
