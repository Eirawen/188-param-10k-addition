"""Baseline-equivalent clone used to validate candidate-module harness paths."""

import param_343 as _p343

# Public + harness-required re-exports.
ModelArgs = _p343.ModelArgs
RMSNorm = _p343.RMSNorm
SelfAttention = _p343.SelfAttention
MLP = _p343.MLP
DecoderLayer = _p343.DecoderLayer
Model = _p343.Model

MODEL_LAYERS = _p343.MODEL_LAYERS
MODEL_DIM = _p343.MODEL_DIM
ATTENTION_HEADS = _p343.ATTENTION_HEADS
KEY_VALUE_HEADS = _p343.KEY_VALUE_HEADS
HEAD_DIM = _p343.HEAD_DIM
INTERMEDIATE_SIZE = _p343.INTERMEDIATE_SIZE
VOCAB_SIZE = _p343.VOCAB_SIZE
OUTPUT_DIGITS = _p343.OUTPUT_DIGITS
MAX_ADDEND = _p343.MAX_ADDEND

build_model_args = _p343.build_model_args
hand_set_weights_magic = _p343.hand_set_weights_magic
build_magic_model = _p343.build_magic_model
count_parameters = _p343.count_parameters
_encode_addends_internal = _p343._encode_addends_internal
_expected_output = _p343._expected_output
_generate_output_batch = _p343._generate_output_batch
