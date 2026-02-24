Codex generated summary.

# 10-Digit Addition Transformer Compression Notes

This repo contains a hand-weighted tiny transformer for autoregressive 10-digit addition and a compressed standalone variant.

Current headline artifact:

- `param_343.py`: original standalone hand-weighted transformer (`343` params) (from @N8programs on X - [link to gist](https://gist.github.com/N8python/02e41d156ec615328cde2e1e5c0e9d53).
- `param_188.py`: standalone exact-compressed transformer (`188` params)

`param_188.py` is not a retrained model and does not change task semantics. It preserves the same autoregressive transformer behavior and was validated with the local harness.

## What Changed In `param_188.py`

`param_188.py` starts from the same underlying hand-set arithmetic circuit as `param_343.py`, but compresses it using exact transformations (plus one exact zero-unit prune already proven equivalent in earlier experiments).

Parameter path:

- `343` (`param_343.py`)
- `328`: exact layer-0 MLP width prune (`Exp3`) by dropping a structurally zero hidden unit
- `195`: exact MLX-style reparameterization stack applied on top of `Exp3` (`Exp46`)
- `188`: exact shared `v_proj` across layers (`Exp47`)

Exact changes in `param_188.py` relative to `param_343.py`:

- Inlines the baseline architecture and hand-set weights (standalone, no `param_343` import)
- Applies the exact `Exp3` layer-0 MLP prune:
  - layer 0 MLP intermediate size `3 -> 2`
  - removes an all-zero unit (no behavior change)
- Reparameterizes major linear layers as exact rank-1 factors:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - `mlp.up_proj`, `mlp.down_proj`
  - in both layers
- Reparameterizes the embedding as exact rank-2:
  - `10x5 -> (10x2) + (2x5)`
- Reparameterizes layer-0 gate as an exact sparse module:
  - stores only the active `2x3` block
  - inactive columns are hard-zero by construction
- Replaces learnable RMSNorm weight vectors with parameter-free fixed-scale RMSNorm:
  - RMSNorm math is retained
  - scales are fixed constants (`1.0` for main norms, `16.0` for q/k norms)
  - this is not an identity replacement
- Shares identical `v_proj` across layers:
  - same module / same `Parameter` objects
  - counted once by `named_parameters()`

## How `188` Differs From The `197` Exact Recipe

The MLX-style exact recipe reproduced locally is `197` params (`Exp42`) and consists of:

- rank-1 major linears
- rank-2 embedding
- sparse layer-0 gate
- fixed-scale RMSNorm (no learnable norm vectors)

`param_188.py` goes below that by adding two exact reductions:

- exact `Exp3` layer-0 MLP zero-unit prune lineage, which lowers the fully reparameterized count from `197 -> 195` (net `-2` versus the baseline-family exact recipe)
- exact `v_proj` sharing across layers (`-7` after rank-1 factorization)

Equivalent comparison points from the log:

- `Exp42` (baseline-family exact recipe): `197` params
- `Exp43` (`Exp42` + shared `v_proj`): `190` params
- `Exp46` (`Exp3` + exact recipe): `195` params
- `Exp47` (`Exp46` + shared `v_proj`): `188` params (this is the standalone `param_188.py`)

## Validation Summary (Local)

Harness:

- `test_param_343.py` (candidate/oracle validation mode)

Validated results for the `188` construction:

- `Exp47` candidate (`candidates.param_candidate_exp47_mlx195_plus_vproj_share_exp3`)
  - structured suite: pass (`101123` cases)
  - random exact-match:
    - `100k` seed `123456789`: `1.00000000`
    - `1M` seed `123456789`: `1.00000000`
    - `100k` seed `987654321`: `1.00000000`
- standalone `param_188.py`
  - equivalence vs `Exp47`: pass (`10k` random cases)
  - fresh random exact-match (`1M`, seed `123456789`): `1.00000000`

Note:

- The harness only writes `--failure-dump` JSON when there are mismatches. A clean pass produces stdout output but no failure JSON file.

## Parameter Counting Definition (Important)

Counts in this repo are reported using:

- `model.named_parameters()` (unique trainable tensors)
- `model.named_buffers()` (reported separately; `param_188.py` uses `0` buffers)

This matters because `param_188.py` uses exact cross-layer sharing for `v_proj`.

- The shared `v_proj` is counted once (legit weight tying / parameter sharing)
- This is a count of unique trainable tensors, not “untied per-layer slots”

## Key Experiment Lessons (Frontier Map)

The experiment log (`EXPERIMENT_LOG.md`) documents the full search. The main lessons:

- Naive destructive ablations hit real mechanistic boundaries quickly.
  - removing q/k norms failed
  - removing RoPE failed
  - single-layer variants failed
  - naive single-head slicing failed (often with timing/position shifts)
  - naive layer1 MLP width-2 pruning failed (carry-threshold instability)
- Exact structural simplifications can work when they respect the hand-built circuit.
  - `Exp3`: layer-0 MLP width `3 -> 2` is exact because the dropped unit is structurally zero
  - `Exp16`: final RMSNorm removal was an exact win in that lineage
- The biggest compression wins came from exact reparameterization, not ablation.
  - rank-1 linears alone (`Exp38`) reduced `343 -> 259` exactly
  - fixed-scale RMSNorms (`Exp39`) corrected an earlier misunderstanding (“remove norm params” is not the same as removing RMSNorm math)
  - embedding factorization (`Exp40`) and sparse gate (`Exp41`) were also exact
- Exact parameter sharing remains useful after low-rank compression.
  - `Exp43` and `Exp47` gained another `-7` from sharing identical factorized `v_proj`
- Numerical rank checks can be misleading if done only in float32.
  - `lm_head` appears rank-3 in float32, but is rank-5 in float64 exact arithmetic for this repo’s weights
  - attempted “exact rank-3 `lm_head` factorization” experiments (`Exp44/45/48/49`) correctly failed at build time

## Repo Organization

To keep the root cleaner:

- historical candidate modules were moved under `candidates/`
- `param_343.py` and `param_188.py` remain at repo root as standalone artifacts

Use candidate modules via package path, for example:

- `candidates.param_candidate_exp47_mlx195_plus_vproj_share_exp3`

## Reproducing A 1M Random Check For `param_188.py`

```bash
conda run -n param343-addition-wsl python test_param_343.py \
  --model-module param_188 \
  --oracle-module param_343 \
  --num-tests 0 \
  --random-accuracy-num-tests 1000000 \
  --min-random-accuracy 0.99 \
  --expected-param-count 188 \
  --batch-size 4096 \
  --golden-seed 123456789 \
  --random-accuracy-progress-every 102400
```

For live progress output with conda:

```bash
conda run --no-capture-output -n param343-addition-wsl python -u test_param_343.py ...
```
