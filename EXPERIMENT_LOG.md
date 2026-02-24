# Experiment Log: 10-Digit Addition Transformer Compression Frontier

Append-only lab notebook for compression experiments against `test_param_343.py`.

## Experiment 0 — Baseline Verification (Pending Environment)
- Timestamp (UTC): 2026-02-24T00:00:00Z
- Candidate module: `baseline`
- Hypothesis: The baseline `param_343` should pass parameter accounting, structured suite, and 100k random checks unchanged after harness extension.
- Exact architecture/config changes: None (baseline `param_343.py`).
- Parameter count (named_parameters): Pending (PyTorch unavailable in current shell environment)
- Buffer count (named_buffers): Pending
- Encoding changed: no
- Harness version/status: `test_param_343.py` patched for `--model-module` / `--oracle-module` and random accuracy mode; backward-compatibility not yet runtime-verified due missing torch.
- Validation commands:
  - `python test_param_343.py --model-module param_343 --oracle-module param_343 --structured-suite --num-tests 100000 --batch-size 4096 --expected-param-count 343`
- Structured-suite results: Pending (blocked on PyTorch install)
- Random results:
  - Seed `123456789`, cases `100000`: Pending (blocked on PyTorch install)
- Failure mode details: Environment blocker: `ModuleNotFoundError: No module named 'torch'`
- What I learned: Harness patch parses successfully and exposes required CLI for candidate/oracle separation and counted random accuracy.
- Next experiment planned: Complete baseline runtime verification after installing/activating the CPU PyTorch environment.

## Experiment Template
- Timestamp (UTC): <YYYY-MM-DDTHH:MM:SSZ>
- Candidate module: `<module>` (or `baseline`)
- Hypothesis: <one sentence>
- Exact architecture/config changes: <exact diff vs baseline or previous>
- Parameter count (named_parameters): <N>
- Buffer count (named_buffers): <B>
- Encoding changed: <no / yes + exact description + fairness note>
- Harness version/status: <patched harness commit state + key flags used>
- Validation commands:
  - `<cmd>`
  - `<cmd>`
- Structured-suite results: <pass/fail + first failing suite/case>
- Random results:
  - Seed `<seed>`, cases `<N>`: <accuracy or first-failure index>
  - Additional seeds: <if run>
- Failure mode details: <classification or N/A>
- What I learned: <specific insight>
- Next experiment planned: <ID + reason>

## Experiment 0 — Baseline Verification (Completed)
- Timestamp (UTC): 2026-02-24T20:39:00Z
- Candidate module: `baseline` (`param_343`)
- Hypothesis: The patched harness remains semantically backward-compatible and the baseline still passes all authoritative checks.
- Exact architecture/config changes: None.
- Parameter count (named_parameters): 343
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` with `--model-module`, `--oracle-module`, and random accuracy mode; baseline tested through both legacy and new codepaths.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --num-tests 1 --batch-size 1 --expected-param-count 343`
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_343 --oracle-module param_343 --structured-suite --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --batch-size 4096 --expected-param-count 343 --failure-dump artifacts/exp0_baseline.json`
- Structured-suite results: PASS (`smoke=10`, `power-boundaries=48`, `carry-chains=65`, `position-digit-sweep=1000`, `no-carry-random=50000`, `heavy-carry-random=50000`, total `101123`)
- Random results:
  - Seed `123456789`, cases `100000`: exact `100000`, mismatches `0`, accuracy `1.00000000` (counted accuracy mode)
- Failure mode details: N/A
- What I learned: The harness patch is runtime-safe on baseline; counted random accuracy mode and candidate/oracle module flags work without breaking legacy behavior.
- Next experiment planned: Baseline mechanism audit with explicit nonzero-density analysis and role hypotheses (Exp1).

## Experiment 1 — Baseline Mechanism Audit
- Timestamp (UTC): 2026-02-24T20:39:00Z
- Candidate module: `baseline` (`param_343`)
- Hypothesis: The hand-set baseline contains significant structural sparsity that can guide low-risk ablations (especially MLP width and normalization components).
- Exact architecture/config changes: None (analysis only).
- Parameter count (named_parameters): 343
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched harness; param breakdown sourced from harness output; sparsity sourced from one-off conda/PyTorch analysis command.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --num-tests 1 --batch-size 1 --expected-param-count 343`
  - `conda run -n param343-addition-wsl python -c "import param_343 as p; m=p.build_magic_model(); rows=[(name,int(t.numel()),int((t!=0).sum().item()),tuple(int(x) for x in t.shape)) for name,t in m.named_parameters()]; rows.sort(key=lambda x:(-x[1],x[0])); print('name\\tshape\\tnumel\\tnonzero\\tdensity'); [print(f'{name}\\t{shape}\\t{n}\\t{nz}\\t{(nz/n if n else 0):.3f}') for name,n,nz,shape in rows]"`
- Structured-suite results: N/A (reused from Exp0 baseline pass)
- Random results:
  - Seed `123456789`, cases `100000`: reused Exp0 result (`1.00000000`)
- Failure mode details: N/A
- What I learned: Largest tensors are `embed_tokens.weight` and `lm_head.weight` (50 each), while several blocks are extremely sparse (`layers.0.self_attn.v_proj.weight` 1/10 nonzero, each `o_proj` 2/20 nonzero, `layers.0.mlp.down_proj.weight` 2/15 nonzero). Layer 1 MLP is dense (15/15 for both `gate_proj` and `up_proj`), suggesting layer1 thresholding/combination is likely harder to compress than layer0.
- Next experiment planned: Clone candidate sanity through the new `--model-module` path (Exp2), then controlled MLP-width ablations.

## Experiment 2a — Clone Candidate Sanity (INVALID, export contract bug)
- Timestamp (UTC): 2026-02-24T20:39:00Z
- Candidate module: `param_candidate_exp02_clone`
- Hypothesis: A baseline-equivalent clone module should pass both candidate self-test mode and cross-module equivalence mode immediately.
- Exact architecture/config changes: None intended; clone initially implemented as `from param_343 import *`.
- Parameter count (named_parameters): 343 (candidate self-test path reached param accounting before failure)
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched harness candidate/oracle path exercised.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --compare-reference-module param_343 --compare-candidate-module param_candidate_exp02_clone --compare-num-tests 10000 --batch-size 4096`
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp02_clone --oracle-module param_343 --num-tests 10 --batch-size 10 --expected-param-count 343`
- Structured-suite results: N/A (failed before structured suite run)
- Random results:
  - Seed `123456789`, cases `10` / `10000`: INVALID (module missing `_generate_output_batch`)
- Failure mode details: Harness contract failure; `import *` does not re-export underscore-prefixed functions required by the harness (`_generate_output_batch`, `_encode_addends_internal`).
- What I learned: Candidate modules must explicitly export the harness-required underscore functions. A minimal clone file must forward these names explicitly.
- Next experiment planned: Fix clone module exports and rerun candidate-path + equivalence sanity checks (Exp2b).

## Experiment 2b — Clone Candidate Sanity (Completed)
- Timestamp (UTC): 2026-02-24T20:39:00Z
- Candidate module: `param_candidate_exp02_clone`
- Hypothesis: With explicit symbol forwarding, the clone module should be behaviorally identical to `param_343`.
- Exact architecture/config changes: None (baseline-equivalent clone with explicit re-exports).
- Parameter count (named_parameters): 343
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched harness; candidate/oracle and equivalence modes both exercised successfully.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --compare-reference-module param_343 --compare-candidate-module param_candidate_exp02_clone --compare-num-tests 10000 --batch-size 4096`
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp02_clone --oracle-module param_343 --num-tests 10 --batch-size 10 --expected-param-count 343`
- Structured-suite results: N/A (not run in this clone sanity pass)
- Random results:
  - Seed `123456789`, cases `10`: strict self-test PASS
  - Seed `123456789`, cases `10000`: equivalence PASS vs `param_343`
- Failure mode details: N/A
- What I learned: The patched harness now supports the intended workflow for arbitrary candidate modules while keeping baseline as the arithmetic oracle.
- Next experiment planned: Begin controlled compression ablations (Exp3 layer0 MLP width reduction to `intermediate_size=2`).

## Experiment 3 — Layer 0 MLP Width 3 -> 2 (WIN, New Best)
- Timestamp (UTC): 2026-02-24T20:47:42Z
- Candidate module: `param_candidate_exp03_layer0_mlp_int2` (alias: `param_candidate_0328_layer0_mlp_int2`)
- Hypothesis: Pruning layer0 MLP unit 2 is exact because baseline weights zero out that unit end-to-end (`gate/up` row 2 and `down` column 2).
- Exact architecture/config changes: Replaced only `layers[0].mlp` with a width-2 MLP; copied baseline units `(0,1)` exactly; all other weights/modules unchanged.
- Parameter count (named_parameters): 328
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched harness candidate/oracle path used throughout.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp03_layer0_mlp_int2 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 328 --batch-size 4096 --failure-dump artifacts/exp03_structured.json`
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp03_layer0_mlp_int2 --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 328 --batch-size 4096 --failure-dump artifacts/exp03_acc100k.json`
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp03_layer0_mlp_int2 --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 1000000 --min-random-accuracy 0.99 --expected-param-count 328 --batch-size 4096 --failure-dump artifacts/exp03_acc1m.json`
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp03_layer0_mlp_int2 --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 328 --batch-size 4096 --golden-seed 987654321 --failure-dump artifacts/exp03_acc100k_seed987654321.json`
- Structured-suite results: PASS (`101123` total cases)
- Random results:
  - Seed `123456789`, cases `100000`: exact `100000`, mismatches `0`, accuracy `1.00000000`
  - Seed `123456789`, cases `1000000`: exact `1000000`, mismatches `0`, accuracy `1.00000000`
  - Seed `987654321`, cases `100000`: exact `100000`, mismatches `0`, accuracy `1.00000000`
- Failure mode details: N/A
- What I learned: Layer0 MLP has a provably redundant intermediate unit that can be removed without any behavior change, reducing the baseline from `343` to `328` params.
- Next experiment planned: Test whether the same pruning works in layer1 (Exp4) and whether both MLPs can be width-2 (Exp5).

## Experiment 4 — Layer 1 MLP Width 3 -> 2 (FAIL)
- Timestamp (UTC): 2026-02-24T20:47:42Z
- Candidate module: `param_candidate_exp04_layer1_mlp_int2`
- Hypothesis: If layer1 MLP contains redundancy similar to layer0, pruning one unit may preserve addition behavior.
- Exact architecture/config changes: Replaced only `layers[1].mlp` with width-2 MLP using copied baseline units `(0,1)`.
- Parameter count (named_parameters): 328
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched harness candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp04_layer1_mlp_int2 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 328 --batch-size 4096 --failure-dump artifacts/exp04_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000009`, `b=0000000009` (expected `81000000000`, got `01000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Boundary carry failure / threshold instability in final carry-generation-decoding stage (`9+9` loses carry and produces `10` instead of `18`).
- What I learned: Layer1 MLP is not trivially redundant; unlike layer0, pruning a unit immediately destroys carry-threshold behavior.
- Next experiment planned: Confirm boundary by pruning both MLPs to width 2 (Exp5).

## Experiment 5 — Both MLPs Width 3 -> 2 (FAIL)
- Timestamp (UTC): 2026-02-24T20:47:42Z
- Candidate module: `param_candidate_exp05_both_mlp_int2`
- Hypothesis: Combining the safe layer0 pruning with the same layer1 pruning may still preserve behavior if layer1 unit selection is lucky.
- Exact architecture/config changes: Replaced both `layers[0].mlp` and `layers[1].mlp` with width-2 MLPs using copied baseline units `(0,1)`.
- Parameter count (named_parameters): 313
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched harness candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp05_both_mlp_int2 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 313 --batch-size 4096 --failure-dump artifacts/exp05_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000009`, `b=0000000009` (expected `81000000000`, got `01000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Same boundary carry failure as Exp4 (`9+9` carry lost).
- What I learned: The layer1 MLP bottleneck dominates this pruning attempt; the layer0 savings remain valid, but layer1 width-2 (with naive unit drop) is not viable.
- Next experiment planned: Test q/k norm removals (Exp6/Exp7).

## Experiment 6 — Remove q/k Norms in Layer 0 (FAIL)
- Timestamp (UTC): 2026-02-24T20:47:42Z
- Candidate module: `param_candidate_exp06_remove_qknorm_layer0`
- Hypothesis: Layer0 q/k norms may be removable with only modest degradation, yielding a low-cost param reduction (`-4`).
- Exact architecture/config changes: Replaced `layers[0].self_attn.q_norm` and `layers[0].self_attn.k_norm` with parameter-free identity modules; all projections/weights unchanged.
- Parameter count (named_parameters): 339
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched harness candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp06_remove_qknorm_layer0 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 339 --batch-size 4096 --failure-dump artifacts/exp06_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000009`, `b=0000000009` (expected `81000000000`, got `90100111100`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Routing collapse due to normalization removal (early output becomes globally corrupted, not just carry bit).
- What I learned: Layer0 q/k normalization is functionally significant for the hand-set attention routing; naive identity replacement is not a drop-in compression.
- Next experiment planned: Remove q/k norms in both layers to confirm failure boundary and parameter target `335` (Exp7).

## Experiment 7 — Remove q/k Norms in Both Layers (FAIL)
- Timestamp (UTC): 2026-02-24T20:47:42Z
- Candidate module: `param_candidate_exp07_remove_qknorm_both`
- Hypothesis: If q/k norms are mostly scale-setting, removing them globally might preserve some behavior while saving `8` params.
- Exact architecture/config changes: Replaced q/k norms with identity modules in both layers; all projections/weights unchanged.
- Parameter count (named_parameters): 335
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched harness candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp07_remove_qknorm_both --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 335 --batch-size 4096 --failure-dump artifacts/exp07_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000000`, `b=0000000001` (expected `10000000000`, got `00000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Severe routing/decoding collapse; even single-digit no-carry case fails.
- What I learned: q/k norms are essential in this mechanistic circuit as implemented; removing them without compensatory reweighting is not viable.
- Next experiment planned: Probe depth boundary with one-layer variants (Exp10/Exp11).

## Experiment 10 — Keep Only Layer 0 (FAIL)
- Timestamp (UTC): 2026-02-24T20:47:42Z
- Candidate module: `param_candidate_exp10_one_layer0_only`
- Hypothesis: A single layer may retain digit routing but lose carry thresholding/final decoding, revealing the role split across layers.
- Exact architecture/config changes: Removed layer 1; kept only baseline layer 0 (`ModuleList([layers[0]])`).
- Parameter count (named_parameters): 224
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched harness candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp10_one_layer0_only --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 224 --batch-size 4096 --failure-dump artifacts/exp10_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000000`, `b=0000000001` (expected `10000000000`, got `00000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Decoding/finalization failure; model cannot emit even the simplest nonzero sum without layer 1.
- What I learned: Layer0 alone is insufficient for final digit decoding / carry resolution in the current circuit split.
- Next experiment planned: Test the complementary one-layer variant (Exp11).

## Experiment 11 — Keep Only Layer 1 (FAIL)
- Timestamp (UTC): 2026-02-24T20:47:42Z
- Candidate module: `param_candidate_exp11_one_layer1_only`
- Hypothesis: If layer1 is primarily the decoder/carry-threshold stage, it may preserve some local behavior but fail where layer0 routing is required.
- Exact architecture/config changes: Removed layer 0; kept only baseline layer 1 (`ModuleList([layers[1]])`).
- Parameter count (named_parameters): 224
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched harness candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp11_one_layer1_only --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 224 --batch-size 4096 --failure-dump artifacts/exp11_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000009`, `b=0000000009` (expected `81000000000`, got `80000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Boundary carry failure (ones digit sum partly decoded, carry propagation/carry generation missing).
- What I learned: Layer1 alone preserves some additive structure but not the routing/precomputation needed for correct carry handling.
- Next experiment planned: Test tied embeddings/output head for a large count drop (Exp13).

## Experiment 13 — Tie Embedding and LM Head (FAIL)
- Timestamp (UTC): 2026-02-24T20:47:42Z
- Candidate module: `param_candidate_exp13_tied_embed_lm_head`
- Hypothesis: Weight tying may be too restrictive, but the parameter drop (`343 -> 293`) is large enough to justify a direct test.
- Exact architecture/config changes: Tied `model.lm_head.weight` to `model.embed_tokens.weight` by sharing the same `Parameter` object.
- Parameter count (named_parameters): 293
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched harness candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp13_tied_embed_lm_head --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 293 --batch-size 4096 --failure-dump artifacts/exp13_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000000`, `b=0000000001` (expected `10000000000`, got `00000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Decoding confusion / output head under-resolved after forced tying.
- What I learned: The baseline embedding basis and output decoding basis are not interchangeable; naive tying immediately destroys even simple no-carry decoding.
- Next experiment planned: Implement and test single-head and/or no-RoPE variants to continue frontier mapping below the current 328-param winner.

## Experiment 14 — Layer1 MLP Width 3 -> 2 (Keep Units 0,2) (FAIL)
- Timestamp (UTC): 2026-02-24T21:06:41Z
- Candidate module: `param_candidate_exp14_l1_mlp_int2_keep02`
- Hypothesis: Layer1 MLP width-2 may work if the dropped unit in earlier attempts was the wrong one; keeping units `(0,2)` could preserve carry thresholding.
- Exact architecture/config changes: Start from `Exp3` (`param_candidate_exp03_layer0_mlp_int2`, 328 params), then prune `layers[1].mlp` to width 2 keeping baseline units `(0,2)`.
- Parameter count (named_parameters): 313
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path; baseline remains oracle.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp14_l1_mlp_int2_keep02 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 313 --batch-size 4096 --failure-dump artifacts/exp14_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000009`, `b=0000000009` (expected `81000000000`, got `91000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Boundary carry-threshold instability / over-carry at `9+9` (produces `19` instead of `18`).
- What I learned: Layer1 MLP width reduction still breaks the carry threshold even with a different unit pair; unit `(1)` is not the only essential contributor.
- Next experiment planned: Try the remaining layer1 width-2 unit pair `(1,2)` (Exp15).

## Experiment 15 — Layer1 MLP Width 3 -> 2 (Keep Units 1,2) (FAIL)
- Timestamp (UTC): 2026-02-24T21:06:41Z
- Candidate module: `param_candidate_exp15_l1_mlp_int2_keep12`
- Hypothesis: Keeping units `(1,2)` might preserve layer1 thresholding if unit `0` is redundant.
- Exact architecture/config changes: Start from `Exp3`, then prune `layers[1].mlp` to width 2 keeping baseline units `(1,2)`.
- Parameter count (named_parameters): 313
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp15_l1_mlp_int2_keep12 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 313 --batch-size 4096 --failure-dump artifacts/exp15_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000000`, `b=0000000001` (expected `10000000000`, got `00000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Decode / activation collapse in simplest no-carry case.
- What I learned: All three tested layer1 width-2 pair choices (`(0,1)`, `(0,2)`, `(1,2)`) fail under deterministic unit dropping; layer1 likely truly needs 3 units without retuning.
- Next experiment planned: Probe non-attention RMSNorm redundancy on top of the 328-param winner (Exp16).

## Experiment 16 — Remove Final RMSNorm (WIN, New Best)
- Timestamp (UTC): 2026-02-24T21:06:41Z
- Candidate module: `param_candidate_exp16_no_final_norm`
- Hypothesis: The final model RMSNorm may be redundant and its effect absorbable by the existing `lm_head` in this hand-set circuit.
- Exact architecture/config changes: Start from `Exp3` (layer0 MLP width-2 exact prune), then replace top-level `model.norm` with parameter-free identity.
- Parameter count (named_parameters): 323
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp16_no_final_norm --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 323 --batch-size 4096 --failure-dump artifacts/exp16_structured.json`
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp16_no_final_norm --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 323 --batch-size 4096 --failure-dump artifacts/exp16_acc100k.json`
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp16_no_final_norm --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 1000000 --min-random-accuracy 0.99 --expected-param-count 323 --batch-size 4096 --failure-dump artifacts/exp16_acc1m.json`
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp16_no_final_norm --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 323 --batch-size 4096 --golden-seed 987654321 --failure-dump artifacts/exp16_acc100k_seed987654321.json`
- Structured-suite results: PASS (`101123` total cases)
- Random results:
  - Seed `123456789`, cases `100000`: exact `100000`, mismatches `0`, accuracy `1.00000000`
  - Seed `123456789`, cases `1000000`: exact `1000000`, mismatches `0`, accuracy `1.00000000`
  - Seed `987654321`, cases `100000`: exact `100000`, mismatches `0`, accuracy `1.00000000`
- Failure mode details: N/A
- What I learned: Final RMSNorm is redundant in the baseline mechanism when starting from the proven `Exp3` prune; this yields a new best of `323` params with strong validation.
- Next experiment planned: Sweep remaining per-layer RMSNorm ablations to test if any additional `-5` reductions stack (Exp17–Exp20).

## Experiment 17 — Remove Layer0 Input RMSNorm (FAIL)
- Timestamp (UTC): 2026-02-24T21:06:41Z
- Candidate module: `param_candidate_exp17_no_l0_input_ln`
- Hypothesis: Layer0 input RMSNorm may be redundant relative to q/k norms and removable for another `-5` params.
- Exact architecture/config changes: Start from `Exp3`, replace `layers[0].input_layernorm` with identity.
- Parameter count (named_parameters): 323
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp17_no_l0_input_ln --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 323 --batch-size 4096 --failure-dump artifacts/exp17_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000000`, `b=0000000001` (expected `10000000000`, got `12000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Routing/scale calibration drift (simple no-carry case emits extra spurious digit).
- What I learned: Layer0 input RMSNorm is not redundant in this hand-set circuit; removing it shifts downstream scale enough to corrupt decoding.
- Next experiment planned: Test layer0 post-attention RMSNorm removal (Exp18).

## Experiment 18 — Remove Layer0 Post-Attention RMSNorm (FAIL)
- Timestamp (UTC): 2026-02-24T21:06:41Z
- Candidate module: `param_candidate_exp18_no_l0_postattn_ln`
- Hypothesis: Layer0 post-attention RMSNorm may be redundant after the exact layer0 MLP prune.
- Exact architecture/config changes: Start from `Exp3`, replace `layers[0].post_attention_layernorm` with identity.
- Parameter count (named_parameters): 323
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp18_no_l0_postattn_ln --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 323 --batch-size 4096 --failure-dump artifacts/exp18_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000001`, `b=0000000000` (expected `10000000000`, got `00000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Early decode collapse / lost activation for simple no-carry case.
- What I learned: The layer0 post-attention norm is functionally important to feed the layer0 MLP thresholding path even after structural pruning.
- Next experiment planned: Test layer1 input RMSNorm removal (Exp19).

## Experiment 19 — Remove Layer1 Input RMSNorm (FAIL)
- Timestamp (UTC): 2026-02-24T21:06:41Z
- Candidate module: `param_candidate_exp19_no_l1_input_ln`
- Hypothesis: Layer1 pre-attention RMSNorm may be removable while preserving routing, enabling another `-5`.
- Exact architecture/config changes: Start from `Exp3`, replace `layers[1].input_layernorm` with identity.
- Parameter count (named_parameters): 323
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp19_no_l1_input_ln --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 323 --batch-size 4096 --failure-dump artifacts/exp19_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000000`, `b=0000000001` (expected `10000000000`, got `90000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Late-stage routing/decoding collapse (digit class shifts dramatically in trivial case).
- What I learned: Layer1 pre-attention RMSNorm is also essential; unlike the final norm, this normalization cannot be removed without re-tuning.
- Next experiment planned: Test layer1 post-attention RMSNorm removal (Exp20).

## Experiment 20 — Remove Layer1 Post-Attention RMSNorm (FAIL)
- Timestamp (UTC): 2026-02-24T21:06:41Z
- Candidate module: `param_candidate_exp20_no_l1_postattn_ln`
- Hypothesis: Layer1 post-attention RMSNorm may be replaceable if the layer1 MLP tolerates raw attention scale.
- Exact architecture/config changes: Start from `Exp3`, replace `layers[1].post_attention_layernorm` with identity.
- Parameter count (named_parameters): 323
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp20_no_l1_postattn_ln --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 323 --batch-size 4096 --failure-dump artifacts/exp20_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000000`, `b=0000000001` (expected `10000000000`, got `99999999999`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Severe threshold saturation / layer1 MLP instability (outputs all `9`s).
- What I learned: Layer1 post-attention normalization is a critical scale stabilizer for the carry-threshold MLP.
- Next experiment planned: Probe RoPE dependence cleanly on top of the 328-param winner (Exp21).

## Experiment 21 — No-RoPE on 328-Param Winner (FAIL)
- Timestamp (UTC): 2026-02-24T21:06:41Z
- Candidate module: `param_candidate_exp21_no_rope_328`
- Hypothesis: RoPE may be unnecessary given fixed encoding, but this direct ablation will reveal whether positional routing depends on it.
- Exact architecture/config changes: Start from `Exp3`, replace both layers’ self-attention with weight-compatible no-RoPE variants (same q/k/v/o projections and q/k norms, RoPE application removed).
- Parameter count (named_parameters): 328
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp21_no_rope_328 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 328 --batch-size 4096 --failure-dump artifacts/exp21_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000000`, `b=0000000001` (expected `10000000000`, got `00000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Positional routing collapse without RoPE.
- What I learned: RoPE is essential to the current mechanistic routing under the existing encoding; removing it without encoding changes is not viable.
- Next experiment planned: Probe single-head compression via deterministic head slicing (Exp22–Exp25).

## Experiment 22 — Single-Head (Layer0=head0, Layer1=head0) (FAIL)
- Timestamp (UTC): 2026-02-24T21:06:41Z
- Candidate module: `param_candidate_exp22_single_head_h0h0`
- Hypothesis: One query head per layer may be sufficient if one baseline head is redundant.
- Exact architecture/config changes: Start from `Exp3`, replace each layer’s self-attention with a 1-query-head module keeping query head 0 (`q_proj` rows `[0:2]`, `o_proj` cols `[0:2]`), preserve `k_proj/v_proj/q_norm/k_norm`.
- Parameter count (named_parameters): 288
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp22_single_head_h0h0 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 288 --batch-size 4096 --failure-dump artifacts/exp22_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000000`, `b=0000000001` (expected `10000000000`, got `00000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Decode/routing collapse from dropping one query head in both layers.
- What I learned: The pair of retained head-0 slices is insufficient to preserve even simple no-carry decoding.
- Next experiment planned: Try the all-head1 single-head variant (Exp23).

## Experiment 23 — Single-Head (Layer0=head1, Layer1=head1) (FAIL)
- Timestamp (UTC): 2026-02-24T21:06:41Z
- Candidate module: `param_candidate_exp23_single_head_h1h1`
- Hypothesis: The useful per-layer head might be head 1 rather than head 0.
- Exact architecture/config changes: Start from `Exp3`, replace each layer’s self-attention with a 1-query-head module keeping query head 1 (`q_proj` rows `[2:4]`, `o_proj` cols `[2:4]`).
- Parameter count (named_parameters): 288
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp23_single_head_h1h1 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 288 --batch-size 4096 --failure-dump artifacts/exp23_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000001`, `b=0000000000` (expected `10000000000`, got `00000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Decode/routing collapse from dropping one query head in both layers (head1-only also insufficient).
- What I learned: Neither single head alone can support the full baseline mechanism when used in both layers without retuning.
- Next experiment planned: Try mixed per-layer head selections to test head specialization by layer (Exp24/Exp25).

## Experiment 24 — Single-Head (Layer0=head0, Layer1=head1) (FAIL)
- Timestamp (UTC): 2026-02-24T21:06:41Z
- Candidate module: `param_candidate_exp24_single_head_h0h1`
- Hypothesis: Heads may specialize differently by layer, so a mixed choice could preserve the role split.
- Exact architecture/config changes: Start from `Exp3`, keep layer0 query head 0 and layer1 query head 1 via 1-query-head sliced attention replacements.
- Parameter count (named_parameters): 288
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp24_single_head_h0h1 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 288 --batch-size 4096 --failure-dump artifacts/exp24_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000001`, `b=0000000000` (expected `10000000000`, got `01000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Positional / temporal misalignment (correct digit appears shifted one step).
- What I learned: Mixed head selection preserves a fragment of behavior but loses alignment, suggesting cross-head cooperation matters for timing/position routing.
- Next experiment planned: Test the opposite mixed head selection (Exp25).

## Experiment 25 — Single-Head (Layer0=head1, Layer1=head0) (FAIL)
- Timestamp (UTC): 2026-02-24T21:06:41Z
- Candidate module: `param_candidate_exp25_single_head_h1h0`
- Hypothesis: The opposite mixed head pairing may better align carry routing and decode roles.
- Exact architecture/config changes: Start from `Exp3`, keep layer0 query head 1 and layer1 query head 0 via 1-query-head sliced attention replacements.
- Parameter count (named_parameters): 288
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path.
- Validation commands:
  - `conda run -n param343-addition-wsl python test_param_343.py --model-module param_candidate_exp25_single_head_h1h0 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 288 --batch-size 4096 --failure-dump artifacts/exp25_structured.json`
- Structured-suite results: FAIL in `smoke` on `a=0000000000`, `b=0000000001` (expected `10000000000`, got `01000000000`)
- Random results:
  - Seed `123456789`, cases `0`: not run (structured fast-fail)
- Failure mode details: Positional / temporal misalignment (digit emitted one step late).
- What I learned: Single-query-head compression via pure head slicing reaches `288` params but fails immediately; head interactions appear essential without re-optimizing weights or changing encoding.
- Next experiment planned: Shift to the next batch focused on encoding changes and/or local search to rescue sub-323 architectures (especially single-head and layer1-width2 variants).


## Pivot — Low-Rank Exact Reparameterization Reproduction (MLX Report)
- Timestamp (UTC): 2026-02-24T21:36:26Z
- External frontier information: Reported MLX reproduction achieves 197 parameters via exact reparameterization (rank-1 linears, rank-2 embedding, sparse layer0 gate, fixed-scale RMSNorm without learnable norm vectors), with full addition tests passing. This is treated as a legitimate hypothesis, not accepted locally until our PyTorch harness reproduces it.
- Fairness note: The planned transformations are exact reparameterizations / exact sharing within the transformer forward path. No harness edits, no task-semantic changes, no lookup tables, no oracle leakage, and parameter accounting remains via `named_parameters()` in `test_param_343.py`.
- Parameter delta math (baseline 343 target): `343 - 84 (rank1 major linears) - 20 (embed rank2) - 9 (sparse gate0) - 33 (learnable norm vectors removed, fixed-scale RMSNorm retained) = 197`.
- Parameter delta math (Exp3 lineage 328 target): `328 - 76 (rank1 major linears) - 20 (embed rank2) - 4 (sparse gate0 on width-2 layer0 gate) - 33 (learnable norm vectors removed, fixed-scale RMSNorm retained) = 195`.
- Local preflight assumptions to verify before experiments: targeted major linears are exact rank-1, embedding is exact rank-2, layer0 gate has the expected sparse pattern, and `v_proj` matches across layers for exact sharing.
- Reproduction plan for this batch (`Exp38`–`Exp49`):
  - Baseline-family exact component checks (`Exp38`–`Exp41`)
  - Full MLX-style exact recipe reproduction (`Exp42`, target 197)
  - Exact extensions below 197 via `v_proj` sharing and exact rank-3 `lm_head` factorization (`Exp43`–`Exp45`)
  - Stack the same recipe on the proven `Exp3` layer0-width2 lineage (`Exp46`–`Exp49`, target as low as 183 with additional exact extensions)
- Validation protocol: equivalence-first (10k random) for exact reparameterizations, then structured suite, then full random accuracy ladder (100k / 1M / extra seed) on milestone and new-best candidates.

## Experiment 38 — Rank-1 Major Linears Only (Baseline) (PASS)
- Timestamp (UTC): 2026-02-24T22:06:41Z
- Candidate module: `param_candidate_exp38_rank1_major_linears_baseline`
- Hypothesis: Exact rank-1 reparameterization of q/k/v/o and MLP up/down in both layers preserves behavior while cutting parameters substantially.
- Exact architecture/config changes: Base `param_343`; replace all `q_proj`, `k_proj`, `v_proj`, `o_proj`, `mlp.up_proj`, and `mlp.down_proj` in both layers with `Rank1LinearExact(u,v)` modules; leave embedding, gate, norms, and `lm_head` unchanged.
- Explicit sharing mode: none
- Parameter count (named_parameters): 259
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path (unchanged in this batch); validations executed via `python run_pivot_exp38_49.py` orchestrating harness CLI calls.
- Decomposition residual checks: rank1 major linears max residual `0`; lm_head rank check `rank32=3`, `rank64=5`
- Equivalence result (10k): PASS (10,000 random cases, seed `123456789`)
- Validation commands:
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --compare-reference-module param_343 --compare-candidate-module param_candidate_exp38_rank1_major_linears_baseline --compare-num-tests 10000 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp38_equiv.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp38_rank1_major_linears_baseline --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 259 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp38_structured.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp38_rank1_major_linears_baseline --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 259 --batch-size 4096 --golden-seed 123456789 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp38_acc100k.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp38_rank1_major_linears_baseline --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 1000000 --min-random-accuracy 0.99 --expected-param-count 259 --batch-size 4096 --golden-seed 123456789 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp38_acc1m.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp38_rank1_major_linears_baseline --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 259 --batch-size 4096 --golden-seed 987654321 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp38_acc100k_seed987654321.json`
- Structured-suite results: PASS (101123 total cases)
- Random results:
  - Seed `123456789`, cases `100000`: accuracy `1.00000000` (exact `100000`, mismatches `0`)
  - Seed `123456789`, cases `1000000`: accuracy `1.00000000` (exact `1000000`, mismatches `0`)
  - Seed `987654321`, cases `100000`: accuracy `1.00000000` (exact `100000`, mismatches `0`)
- Failure mode details: N/A
- What I learned: Exact rank-1 structure in the major linears is real and highly compressive: this alone cuts the model from 343 to 259 with no behavior change.
- Next experiment planned: Exp39

## Experiment 39 — Fixed-Scale RMSNorms Only (Baseline) (PASS)
- Timestamp (UTC): 2026-02-24T22:06:42Z
- Candidate module: `param_candidate_exp39_fixedscale_rmsnorms_baseline`
- Hypothesis: Removing learnable RMSNorm vectors while keeping exact RMSNorm math with fixed scales (1.0 / 16.0) preserves behavior, correcting the earlier mistaken identity-ablation interpretation.
- Exact architecture/config changes: Base `param_343`; replace all layer norms/final norm with parameter-free `FixedScaleRMSNorm(scale=1.0)` and q/k norms with `FixedScaleRMSNorm(scale=16.0)`; no other changes.
- Explicit sharing mode: none
- Parameter count (named_parameters): 310
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path (unchanged in this batch); validations executed via `python run_pivot_exp38_49.py` orchestrating harness CLI calls.
- Decomposition residual checks: fixed-scale RMSNorm replacements `9` modules; lm_head rank check `rank32=3`, `rank64=5`
- Equivalence result (10k): PASS (10,000 random cases, seed `123456789`)
- Validation commands:
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --compare-reference-module param_343 --compare-candidate-module param_candidate_exp39_fixedscale_rmsnorms_baseline --compare-num-tests 10000 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp39_equiv.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp39_fixedscale_rmsnorms_baseline --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 310 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp39_structured.json`
- Structured-suite results: PASS (101123 total cases)
- Random results:
  - Seed `123456789`, cases `0`: not run (not a milestone/new-best candidate after structured pass or build failed)
- Failure mode details: N/A
- What I learned: Prior norm-ablation failures were specifically about removing RMSNorm math; replacing learnable norm vectors with fixed scales preserves exact behavior and yields a valid 33-parameter reduction.
- Next experiment planned: Exp40

## Experiment 40 — Factorized Embedding Rank-2 Only (Baseline) (PASS)
- Timestamp (UTC): 2026-02-24T22:06:43Z
- Candidate module: `param_candidate_exp40_factored_embedding_baseline`
- Hypothesis: The baseline `10x5` embedding is exactly rank-2 and can be replaced by an exact factorized embedding without changing function.
- Exact architecture/config changes: Base `param_343`; replace `embed_tokens.weight (10x5)` with exact rank-2 factorization `token_factors (10x2)` and `mix (2x5)`; no other changes.
- Explicit sharing mode: none
- Parameter count (named_parameters): 323
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path (unchanged in this batch); validations executed via `python run_pivot_exp38_49.py` orchestrating harness CLI calls.
- Decomposition residual checks: embedding rank-2 residual `0`; lm_head rank check `rank32=3`, `rank64=5`
- Equivalence result (10k): PASS (10,000 random cases, seed `123456789`)
- Validation commands:
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --compare-reference-module param_343 --compare-candidate-module param_candidate_exp40_factored_embedding_baseline --compare-num-tests 10000 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp40_equiv.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp40_factored_embedding_baseline --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 323 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp40_structured.json`
- Structured-suite results: PASS (101123 total cases)
- Random results:
  - Seed `123456789`, cases `0`: not run (not a milestone/new-best candidate after structured pass or build failed)
- Failure mode details: N/A
- What I learned: The embedding table is exactly rank-2 and can be factorized losslessly, confirming another large exact-reparameterization lever.
- Next experiment planned: Exp41

## Experiment 41 — Sparse Layer0 Gate Only (Baseline) (PASS)
- Timestamp (UTC): 2026-02-24T22:06:44Z
- Candidate module: `param_candidate_exp41_sparse_gate0_baseline`
- Hypothesis: Layer0 `gate_proj` stores only 6 active coefficients and can be reparameterized sparsely with exact reconstruction and zero-padded inactive output channel.
- Exact architecture/config changes: Base `param_343`; replace `layers.0.mlp.gate_proj (3x5)` with `SparseGate0ExactBaseline` storing only active `w23 (2x3)` and hard-padding the third output channel to zero.
- Explicit sharing mode: none
- Parameter count (named_parameters): 334
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path (unchanged in this batch); validations executed via `python run_pivot_exp38_49.py` orchestrating harness CLI calls.
- Decomposition residual checks: sparse gate0 kind `baseline`, stored shape `(2, 3)`; lm_head rank check `rank32=3`, `rank64=5`
- Equivalence result (10k): PASS (10,000 random cases, seed `123456789`)
- Validation commands:
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --compare-reference-module param_343 --compare-candidate-module param_candidate_exp41_sparse_gate0_baseline --compare-num-tests 10000 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp41_equiv.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp41_sparse_gate0_baseline --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 334 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp41_structured.json`
- Structured-suite results: PASS (101123 total cases)
- Random results:
  - Seed `123456789`, cases `0`: not run (not a milestone/new-best candidate after structured pass or build failed)
- Failure mode details: N/A
- What I learned: Layer0 gate sparsity is structurally exact and can be encoded directly as a sparse submodule with zero-padding, preserving behavior.
- Next experiment planned: Exp42

## Experiment 42 — Full MLX-Style Exact Recipe Reproduction (Baseline) (PASS)
- Timestamp (UTC): 2026-02-24T22:06:45Z
- Candidate module: `param_candidate_exp42_mlx197_repro_baseline`
- Hypothesis: Composing rank-1 major linears, fixed-scale RMSNorms, rank-2 embedding, and sparse layer0 gate reproduces the reported ~197 exact frontier in PyTorch.
- Exact architecture/config changes: Base `param_343`; compose `Exp38 + Exp39 + Exp40 + Exp41` exact reparameterizations (no `v_proj` sharing, no `lm_head` factorization).
- Explicit sharing mode: none
- Parameter count (named_parameters): 197
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path (unchanged in this batch); validations executed via `python run_pivot_exp38_49.py` orchestrating harness CLI calls.
- Decomposition residual checks: rank1 major linears max residual `0`; embedding rank-2 residual `0`; fixed-scale RMSNorm replacements `9` modules; sparse gate0 kind `baseline`, stored shape `(2, 3)`; lm_head rank check `rank32=3`, `rank64=5`
- Equivalence result (10k): PASS (10,000 random cases, seed `123456789`)
- Validation commands:
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --compare-reference-module param_343 --compare-candidate-module param_candidate_exp42_mlx197_repro_baseline --compare-num-tests 10000 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp42_equiv.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp42_mlx197_repro_baseline --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 197 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp42_structured.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp42_mlx197_repro_baseline --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 197 --batch-size 4096 --golden-seed 123456789 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp42_acc100k.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp42_mlx197_repro_baseline --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 1000000 --min-random-accuracy 0.99 --expected-param-count 197 --batch-size 4096 --golden-seed 123456789 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp42_acc1m.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp42_mlx197_repro_baseline --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 197 --batch-size 4096 --golden-seed 987654321 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp42_acc100k_seed987654321.json`
- Structured-suite results: PASS (101123 total cases)
- Random results:
  - Seed `123456789`, cases `100000`: accuracy `1.00000000` (exact `100000`, mismatches `0`)
  - Seed `123456789`, cases `1000000`: accuracy `1.00000000` (exact `1000000`, mismatches `0`)
  - Seed `987654321`, cases `100000`: accuracy `1.00000000` (exact `100000`, mismatches `0`)
- Failure mode details: N/A
- What I learned: The external MLX-style 197 result reproduces cleanly in the PyTorch harness under strict equivalence + structured + large random validation.
- Next experiment planned: Exp43

## Experiment 43 — Full Exact Recipe + Shared v_proj (Baseline Family) (PASS)
- Timestamp (UTC): 2026-02-24T22:06:46Z
- Candidate module: `param_candidate_exp43_mlx197_plus_vproj_share_baseline`
- Hypothesis: On top of the 197 exact recipe, exact module aliasing of identical factorized `v_proj` across layers should reduce count further with no behavior change.
- Exact architecture/config changes: Base `Exp42`; alias `layers[1].self_attn.v_proj = layers[0].self_attn.v_proj` after rank-1 reparameterization (same module instance / shared Parameters).
- Explicit sharing mode: module aliasing (`v_proj` across layers)
- Parameter count (named_parameters): 190
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path (unchanged in this batch); validations executed via `python run_pivot_exp38_49.py` orchestrating harness CLI calls.
- Decomposition residual checks: rank1 major linears max residual `0`; embedding rank-2 residual `0`; fixed-scale RMSNorm replacements `9` modules; sparse gate0 kind `baseline`, stored shape `(2, 3)`; shared_vproj identity checks `{'module_shared': True, 'pre_equal': True, 'weight_param_shared': True}`; lm_head rank check `rank32=3`, `rank64=5`
- Equivalence result (10k): PASS (10,000 random cases, seed `123456789`)
- Validation commands:
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --compare-reference-module param_343 --compare-candidate-module param_candidate_exp43_mlx197_plus_vproj_share_baseline --compare-num-tests 10000 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp43_equiv.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp43_mlx197_plus_vproj_share_baseline --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 190 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp43_structured.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp43_mlx197_plus_vproj_share_baseline --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 190 --batch-size 4096 --golden-seed 123456789 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp43_acc100k.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp43_mlx197_plus_vproj_share_baseline --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 1000000 --min-random-accuracy 0.99 --expected-param-count 190 --batch-size 4096 --golden-seed 123456789 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp43_acc1m.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp43_mlx197_plus_vproj_share_baseline --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 190 --batch-size 4096 --golden-seed 987654321 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp43_acc100k_seed987654321.json`
- Structured-suite results: PASS (101123 total cases)
- Random results:
  - Seed `123456789`, cases `100000`: accuracy `1.00000000` (exact `100000`, mismatches `0`)
  - Seed `123456789`, cases `1000000`: accuracy `1.00000000` (exact `1000000`, mismatches `0`)
  - Seed `987654321`, cases `100000`: accuracy `1.00000000` (exact `100000`, mismatches `0`)
- Failure mode details: N/A
- What I learned: Combining the MLX recipe with exact `v_proj` module aliasing is a genuine additional win; exact sharing remains valuable after low-rank reparameterization.
- Next experiment planned: Exp44

## Experiment 44 — Full Exact Recipe + lm_head Rank-3 Factorization (Baseline Family) (BUILD FAIL)
- Timestamp (UTC): 2026-02-24T22:06:47Z
- Candidate module: `param_candidate_exp44_mlx197_plus_lmhead_r3_baseline`
- Hypothesis: If `lm_head` were exactly rank-3, exact rank-3 factorization would push the baseline-family exact recipe below 197 without retuning.
- Exact architecture/config changes: Base `Exp42`; attempt exact `FactorizedLinearExact(rank=3)` replacement for `lm_head`.
- Explicit sharing mode: none
- Parameter count (named_parameters): N/A (build failed before accounting; target `192`)
- Buffer count (named_buffers): N/A
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path (unchanged in this batch); validations executed via `python run_pivot_exp38_49.py` orchestrating harness CLI calls.
- Decomposition residual checks: N/A (build failed before metadata collection)
- Equivalence result (10k): not run (candidate build failed before equivalence)
- Validation commands:
  - ``python run_pivot_exp38_49.py` (candidate build/meta precheck failed before harness equivalence/structured commands were issued)`
- Structured-suite results: not run (candidate build failed before harness validation)
- Random results:
  - Seed `123456789`, cases `0`: not run (not a milestone/new-best candidate after structured pass or build failed)
- Failure mode details: Exactness assumption invalid: `lm_head` is rank-5 in float64 exact arithmetic, so exact rank-3 factorization is impossible in this batch.
- What I learned: The planned exact `lm_head` rank-3 factorization is invalid for this repo weights: `lm_head` is rank-5 in float64 exact arithmetic despite appearing rank-3 in float32 numerical rank checks.
- Next experiment planned: Exp45

## Experiment 45 — Full Exact Recipe + Shared v_proj + lm_head Rank-3 (Baseline Family) (BUILD FAIL)
- Timestamp (UTC): 2026-02-24T22:06:48Z
- Candidate module: `param_candidate_exp45_mlx197_plus_vprojshare_lmheadr3_baseline`
- Hypothesis: Composing exact `v_proj` sharing with exact rank-3 `lm_head` factorization could push the baseline-family exact frontier lower than `Exp43` if `lm_head` rank-3 exactness holds.
- Exact architecture/config changes: Base `Exp42`; add exact `v_proj` module aliasing plus attempted exact `lm_head` rank-3 factorization.
- Explicit sharing mode: module aliasing (`v_proj` across layers) + attempted exact factorization
- Parameter count (named_parameters): N/A (build failed before accounting; target `185`)
- Buffer count (named_buffers): N/A
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path (unchanged in this batch); validations executed via `python run_pivot_exp38_49.py` orchestrating harness CLI calls.
- Decomposition residual checks: N/A (build failed before metadata collection)
- Equivalence result (10k): not run (candidate build failed before equivalence)
- Validation commands:
  - ``python run_pivot_exp38_49.py` (candidate build/meta precheck failed before harness equivalence/structured commands were issued)`
- Structured-suite results: not run (candidate build failed before harness validation)
- Random results:
  - Seed `123456789`, cases `0`: not run (not a milestone/new-best candidate after structured pass or build failed)
- Failure mode details: Exactness assumption invalid: `lm_head` is rank-5 in float64 exact arithmetic, so exact rank-3 factorization is impossible in this batch.
- What I learned: The planned exact `lm_head` rank-3 factorization is invalid for this repo weights: `lm_head` is rank-5 in float64 exact arithmetic despite appearing rank-3 in float32 numerical rank checks.
- Next experiment planned: Exp46

## Experiment 46 — Full MLX-Style Exact Recipe on Exp3 Lineage (PASS)
- Timestamp (UTC): 2026-02-24T22:06:49Z
- Candidate module: `param_candidate_exp46_mlx197_recipe_on_exp3`
- Hypothesis: Stacking the MLX-style exact reparameterization recipe onto the proven `Exp3` layer0-width2 prune should yield an exact ~195-param model.
- Exact architecture/config changes: Base `param_candidate_exp03_layer0_mlp_int2`; apply rank-1 major linears, fixed-scale RMSNorms, rank-2 embedding, and `SparseGate0ExactExp3` for the width-2 layer0 gate.
- Explicit sharing mode: none
- Parameter count (named_parameters): 195
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path (unchanged in this batch); validations executed via `python run_pivot_exp38_49.py` orchestrating harness CLI calls.
- Decomposition residual checks: rank1 major linears max residual `0`; embedding rank-2 residual `0`; fixed-scale RMSNorm replacements `9` modules; sparse gate0 kind `exp3`, stored shape `(2, 3)`; lm_head rank check `rank32=3`, `rank64=5`
- Equivalence result (10k): PASS (10,000 random cases, seed `123456789`)
- Validation commands:
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --compare-reference-module param_candidate_exp03_layer0_mlp_int2 --compare-candidate-module param_candidate_exp46_mlx197_recipe_on_exp3 --compare-num-tests 10000 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp46_equiv.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp46_mlx197_recipe_on_exp3 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 195 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp46_structured.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp46_mlx197_recipe_on_exp3 --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 195 --batch-size 4096 --golden-seed 123456789 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp46_acc100k.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp46_mlx197_recipe_on_exp3 --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 1000000 --min-random-accuracy 0.99 --expected-param-count 195 --batch-size 4096 --golden-seed 123456789 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp46_acc1m.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp46_mlx197_recipe_on_exp3 --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 195 --batch-size 4096 --golden-seed 987654321 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp46_acc100k_seed987654321.json`
- Structured-suite results: PASS (101123 total cases)
- Random results:
  - Seed `123456789`, cases `100000`: accuracy `1.00000000` (exact `100000`, mismatches `0`)
  - Seed `123456789`, cases `1000000`: accuracy `1.00000000` (exact `1000000`, mismatches `0`)
  - Seed `987654321`, cases `100000`: accuracy `1.00000000` (exact `100000`, mismatches `0`)
- Failure mode details: N/A
- What I learned: The MLX-style exact recipe stacks cleanly on the proven Exp3 layer0-width2 prune, reproducing an exact 195-param model.
- Next experiment planned: Exp47

## Experiment 47 — Exp3 Exact Recipe + Shared v_proj (PASS)
- Timestamp (UTC): 2026-02-24T22:06:50Z
- Candidate module: `param_candidate_exp47_mlx195_plus_vproj_share_exp3`
- Hypothesis: Combining the `Exp46` exact recipe with exact `v_proj` sharing across layers should beat the 197 milestone without retuning.
- Exact architecture/config changes: Base `Exp46`; alias factorized `v_proj` modules across layers (same module instance / shared Parameters).
- Explicit sharing mode: module aliasing (`v_proj` across layers)
- Parameter count (named_parameters): 188
- Buffer count (named_buffers): 0
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path (unchanged in this batch); validations executed via `python run_pivot_exp38_49.py` orchestrating harness CLI calls.
- Decomposition residual checks: rank1 major linears max residual `0`; embedding rank-2 residual `0`; fixed-scale RMSNorm replacements `9` modules; sparse gate0 kind `exp3`, stored shape `(2, 3)`; shared_vproj identity checks `{'module_shared': True, 'pre_equal': True, 'weight_param_shared': True}`; lm_head rank check `rank32=3`, `rank64=5`
- Equivalence result (10k): PASS (10,000 random cases, seed `123456789`)
- Validation commands:
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --compare-reference-module param_candidate_exp03_layer0_mlp_int2 --compare-candidate-module param_candidate_exp47_mlx195_plus_vproj_share_exp3 --compare-num-tests 10000 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp47_equiv.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp47_mlx195_plus_vproj_share_exp3 --oracle-module param_343 --structured-suite --num-tests 0 --expected-param-count 188 --batch-size 4096 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp47_structured.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp47_mlx195_plus_vproj_share_exp3 --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 188 --batch-size 4096 --golden-seed 123456789 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp47_acc100k.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp47_mlx195_plus_vproj_share_exp3 --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 1000000 --min-random-accuracy 0.99 --expected-param-count 188 --batch-size 4096 --golden-seed 123456789 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp47_acc1m.json`
  - `conda run -n param343-addition-wsl python /home/khaled/03f119dcfc4b8351855c00b1c5224b58/test_param_343.py --model-module param_candidate_exp47_mlx195_plus_vproj_share_exp3 --oracle-module param_343 --num-tests 0 --random-accuracy-num-tests 100000 --min-random-accuracy 0.99 --expected-param-count 188 --batch-size 4096 --golden-seed 987654321 --failure-dump /home/khaled/03f119dcfc4b8351855c00b1c5224b58/artifacts/exp47_acc100k_seed987654321.json`
- Structured-suite results: PASS (101123 total cases)
- Random results:
  - Seed `123456789`, cases `100000`: accuracy `1.00000000` (exact `100000`, mismatches `0`)
  - Seed `123456789`, cases `1000000`: accuracy `1.00000000` (exact `1000000`, mismatches `0`)
  - Seed `987654321`, cases `100000`: accuracy `1.00000000` (exact `100000`, mismatches `0`)
- Failure mode details: N/A
- What I learned: Stacking `v_proj` sharing on the Exp3 exact recipe gives the strongest exact no-retune result in this batch (188 params), beating the 197 reproduction and our previous 323 frontier by a large margin.
- Next experiment planned: Exp48

## Experiment 48 — Exp3 Exact Recipe + lm_head Rank-3 Factorization (BUILD FAIL)
- Timestamp (UTC): 2026-02-24T22:06:51Z
- Candidate module: `param_candidate_exp48_mlx195_plus_lmhead_r3_exp3`
- Hypothesis: If `lm_head` were exactly rank-3 on the Exp3 lineage, exact factorization would reduce the count further without affecting behavior.
- Exact architecture/config changes: Base `Exp46`; attempt exact `FactorizedLinearExact(rank=3)` replacement for `lm_head`.
- Explicit sharing mode: none
- Parameter count (named_parameters): N/A (build failed before accounting; target `190`)
- Buffer count (named_buffers): N/A
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path (unchanged in this batch); validations executed via `python run_pivot_exp38_49.py` orchestrating harness CLI calls.
- Decomposition residual checks: N/A (build failed before metadata collection)
- Equivalence result (10k): not run (candidate build failed before equivalence)
- Validation commands:
  - ``python run_pivot_exp38_49.py` (candidate build/meta precheck failed before harness equivalence/structured commands were issued)`
- Structured-suite results: not run (candidate build failed before harness validation)
- Random results:
  - Seed `123456789`, cases `0`: not run (not a milestone/new-best candidate after structured pass or build failed)
- Failure mode details: Exactness assumption invalid: `lm_head` is rank-5 in float64 exact arithmetic, so exact rank-3 factorization is impossible in this batch.
- What I learned: The planned exact `lm_head` rank-3 factorization is invalid for this repo weights: `lm_head` is rank-5 in float64 exact arithmetic despite appearing rank-3 in float32 numerical rank checks.
- Next experiment planned: Exp49

## Experiment 49 — Exp3 Exact Recipe + Shared v_proj + lm_head Rank-3 (BUILD FAIL)
- Timestamp (UTC): 2026-02-24T22:06:52Z
- Candidate module: `param_candidate_exp49_mlx195_plus_vprojshare_lmheadr3_exp3`
- Hypothesis: Composing `Exp47` with exact rank-3 `lm_head` factorization would provide the strongest no-retune exact point if `lm_head` rank-3 exactness held.
- Exact architecture/config changes: Base `Exp46`; add exact `v_proj` aliasing plus attempted exact `lm_head` rank-3 factorization.
- Explicit sharing mode: module aliasing (`v_proj` across layers) + attempted exact factorization
- Parameter count (named_parameters): N/A (build failed before accounting; target `183`)
- Buffer count (named_buffers): N/A
- Encoding changed: no
- Harness version/status: Patched `test_param_343.py` candidate/oracle path (unchanged in this batch); validations executed via `python run_pivot_exp38_49.py` orchestrating harness CLI calls.
- Decomposition residual checks: N/A (build failed before metadata collection)
- Equivalence result (10k): not run (candidate build failed before equivalence)
- Validation commands:
  - ``python run_pivot_exp38_49.py` (candidate build/meta precheck failed before harness equivalence/structured commands were issued)`
- Structured-suite results: not run (candidate build failed before harness validation)
- Random results:
  - Seed `123456789`, cases `0`: not run (not a milestone/new-best candidate after structured pass or build failed)
- Failure mode details: Exactness assumption invalid: `lm_head` is rank-5 in float64 exact arithmetic, so exact rank-3 factorization is impossible in this batch.
- What I learned: The planned exact `lm_head` rank-3 factorization is invalid for this repo weights: `lm_head` is rank-5 in float64 exact arithmetic despite appearing rank-3 in float32 numerical rank checks.
- Next experiment planned: Next batch: exact lm_head alternatives (rank-5 exact factorization or approximate+retune) and additional exact sharing opportunities
