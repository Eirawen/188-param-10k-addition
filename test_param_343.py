import argparse
import importlib
import json
import random
from pathlib import Path
from typing import Iterable, Iterator

MAX_ADDEND = 10**10 - 1
OUTPUT_DIGITS = 11
DEFAULT_GOLDEN_SEED = 123456789
DEFAULT_EXPECTED_PARAM_COUNT = 343


def _chunked(it: Iterable[tuple[int, int]], batch_size: int) -> Iterator[list[tuple[int, int]]]:
    batch: list[tuple[int, int]] = []
    for item in it:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_param_343():
    return importlib.import_module("param_343")


def _load_module(name: str):
    return importlib.import_module(name)


def write_addition_dataset(
    output_path: str,
    count: int,
    seed: int,
    fmt: str = "csv",
    progress_every: int = 100_000,
) -> None:
    if count < 0:
        raise ValueError("dataset count must be >= 0")
    if progress_every <= 0:
        raise ValueError("progress_every must be > 0")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    with path.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        if fmt == "csv":
            f.write("a,b,sum,reversed_sum_11\n")
        elif fmt == "tsv":
            f.write("a\tb\tsum\treversed_sum_11\n")
        elif fmt != "jsonl":
            raise ValueError(f"unsupported dataset format: {fmt}")

        for i in range(1, count + 1):
            a = rng.randint(0, MAX_ADDEND)
            b = rng.randint(0, MAX_ADDEND)
            s = a + b
            rev = str(s)[::-1].ljust(OUTPUT_DIGITS, "0")
            if fmt == "csv":
                f.write(f"{a:010d},{b:010d},{s},{rev}\n")
            elif fmt == "tsv":
                f.write(f"{a:010d}\t{b:010d}\t{s}\t{rev}\n")
            else:
                f.write(
                    json.dumps(
                        {"a": f"{a:010d}", "b": f"{b:010d}", "sum": s, "reversed_sum_11": rev},
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            if i % progress_every == 0 or i == count:
                print(f"dataset progress: {i}/{count}")


def _dump_failure(path: str | None, payload: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote failure dump: {out}")


def _build_failure_payload(
    candidate_mod,
    model,
    *,
    suite: str,
    case_index: int,
    seed: int | None,
    a: int,
    b: int,
    expected: str,
    got: str,
    include_argmax_trace: bool,
) -> dict:
    payload = {
        "suite": suite,
        "case_index": case_index,
        "seed": seed,
        "a": f"{a:010d}",
        "b": f"{b:010d}",
        "expected": expected,
        "got": got,
        "input_tokens": candidate_mod._encode_addends_internal(a, b),
    }
    if include_argmax_trace:
        payload["argmax_trace"] = _compute_argmax_trace(candidate_mod, model, a, b)
    return payload


def _compute_argmax_trace(candidate_mod, model, a: int, b: int) -> list[dict]:
    # Optional debug trace for deterministic failure reproduction.
    torch = importlib.import_module("torch")
    seq = list(candidate_mod._encode_addends_internal(a, b))
    trace: list[dict] = []
    model.eval()
    with torch.no_grad():
        for step in range(candidate_mod.OUTPUT_DIGITS):
            x = torch.tensor([seq], dtype=torch.long)
            logits = model(x)
            last = logits[0, -1, :]
            argmax = int(torch.argmax(last).item())
            topk = min(3, int(last.numel()))
            vals, idx = torch.topk(last, k=topk)
            trace.append(
                {
                    "step": step,
                    "input_len": len(seq),
                    "argmax": argmax,
                    "topk": [
                        {"digit": int(d), "logit": float(v)}
                        for d, v in zip(idx.tolist(), vals.tolist())
                    ],
                }
            )
            seq.append(argmax)
    return trace


def _iter_random_examples(count: int, seed: int) -> Iterator[tuple[int, int]]:
    rng = random.Random(seed)
    for _ in range(count):
        yield rng.randint(0, MAX_ADDEND), rng.randint(0, MAX_ADDEND)


def _named_parameter_breakdown(model) -> list[tuple[str, int, tuple[int, ...]]]:
    if not hasattr(model, "named_parameters"):
        return []
    rows = []
    for name, param in model.named_parameters():
        rows.append((name, int(param.numel()), tuple(int(x) for x in param.shape)))
    rows.sort(key=lambda x: (-x[1], x[0]))
    return rows


def _named_buffer_breakdown(model) -> list[tuple[str, int, tuple[int, ...]]]:
    if not hasattr(model, "named_buffers"):
        return []
    rows = []
    for name, buf in model.named_buffers():
        rows.append((name, int(buf.numel()), tuple(int(x) for x in buf.shape)))
    rows.sort(key=lambda x: (-x[1], x[0]))
    return rows


def enforce_param_accounting(
    model,
    expected_param_count: int | None = DEFAULT_EXPECTED_PARAM_COUNT,
    breakdown_limit: int = 20,
) -> int:
    rows = _named_parameter_breakdown(model)
    if rows:
        total = sum(n for _, n, _ in rows)
        print(f"parameter count (named_parameters): {total}")
        print(f"parameter breakdown (top {min(breakdown_limit, len(rows))} by size):")
        for name, n, shape in rows[:breakdown_limit]:
            print(f"  {name}: {n} {shape}")
    else:
        total = None

    buffers = _named_buffer_breakdown(model)
    if buffers:
        buffer_total = sum(n for _, n, _ in buffers)
        print(f"buffer count (named_buffers): {buffer_total}")
        for name, n, shape in buffers[: min(10, len(buffers))]:
            print(f"  [buffer] {name}: {n} {shape}")
    else:
        print("buffer count (named_buffers): 0")

    if expected_param_count is not None:
        if total is None:
            raise AssertionError("Cannot enforce param count: model has no named_parameters()")
        if total != expected_param_count:
            raise AssertionError(
                f"Parameter count mismatch: expected {expected_param_count}, got {total}"
            )
    return total if total is not None else -1


def _single_case_examples() -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, 1),
        (1, 0),
        (9, 9),
        (10, 90),
        (99, 1),
        (9999999999, 0),
        (9999999999, 1),
        (5000000000, 5000000000),
        (1234567890, 9876543210),
    ]


def _power_boundary_examples() -> Iterator[tuple[int, int]]:
    for k in range(1, 11):
        p = 10**k
        if p - 1 <= MAX_ADDEND:
            yield (p - 1, 1)
        if p <= MAX_ADDEND:
            yield (p, 0)
            yield (p, p)
        if p - 2 >= 0:
            yield (p - 2, 2)
        if p - 5 >= 0:
            yield (p - 5, 5)


def _carry_chain_examples() -> Iterator[tuple[int, int]]:
    for chain_len in range(1, 11):
        a = 10**chain_len - 1
        b = 1
        yield (a, b)
        for shift in range(0, 10 - chain_len + 1):
            aa = a * (10**shift)
            bb = 10**shift
            if aa <= MAX_ADDEND and bb <= MAX_ADDEND:
                yield (aa, bb)


def _position_digit_sweep_examples() -> Iterator[tuple[int, int]]:
    for pos in range(10):
        place = 10**pos
        for da in range(10):
            for db in range(10):
                yield (da * place, db * place)


def _no_carry_random_examples(count: int, seed: int) -> Iterator[tuple[int, int]]:
    rng = random.Random(seed)
    for _ in range(count):
        a_digits = []
        b_digits = []
        for _ in range(10):
            da = rng.randint(0, 9)
            db = rng.randint(0, 9 - da)
            a_digits.append(str(da))
            b_digits.append(str(db))
        yield int("".join(a_digits)), int("".join(b_digits))


def _heavy_carry_random_examples(count: int, seed: int) -> Iterator[tuple[int, int]]:
    rng = random.Random(seed)
    for _ in range(count):
        a_digits = []
        b_digits = []
        for _ in range(10):
            da = rng.randint(5, 9)
            db = rng.randint(max(0, 10 - da), 9)
            a_digits.append(str(da))
            b_digits.append(str(db))
        yield int("".join(a_digits)), int("".join(b_digits))


def _low_digits_exhaustive_examples(num_digits: int) -> Iterator[tuple[int, int]]:
    if num_digits < 1 or num_digits > 4:
        raise ValueError("num_digits must be in [1, 4]")
    n = 10**num_digits
    for a in range(n):
        for b in range(n):
            yield a, b


def _run_named_case_stream(
    oracle_mod,
    candidate_mod,
    name: str,
    model,
    cases: Iterable[tuple[int, int]],
    batch_size: int,
    *,
    failure_dump_path: str | None,
    include_argmax_trace: bool,
    seed: int | None,
    case_index_start: int = 0,
) -> int:
    total = 0
    for batch in _chunked(cases, batch_size):
        expected = [oracle_mod._expected_output(a, b) for a, b in batch]
        actual = candidate_mod._generate_output_batch(model, batch)
        for i, ((a, b), exp, act) in enumerate(zip(batch, expected, actual)):
            if act != exp:
                _dump_failure(
                    failure_dump_path,
                    _build_failure_payload(
                        candidate_mod,
                        model,
                        suite=name,
                        case_index=case_index_start + total + i,
                        seed=seed,
                        a=a,
                        b=b,
                        expected=exp,
                        got=act,
                        include_argmax_trace=include_argmax_trace,
                    ),
                )
                raise AssertionError(
                    f"[{name}] mismatch for a={a:010d}, b={b:010d}: expected {exp}, got {act}"
                )
        total += len(batch)
    print(f"structured suite: {name} passed ({total} cases)")
    return total


def run_structured_test_suite(
    oracle_mod,
    candidate_mod,
    model,
    batch_size: int,
    exhaustive_low_digits: int,
    *,
    failure_dump_path: str | None,
    include_argmax_trace: bool,
    seed: int | None,
) -> None:
    total = 0
    total += _run_named_case_stream(
        oracle_mod,
        candidate_mod,
        "smoke",
        model,
        _single_case_examples(),
        batch_size,
        failure_dump_path=failure_dump_path,
        include_argmax_trace=include_argmax_trace,
        seed=seed,
        case_index_start=total,
    )
    total += _run_named_case_stream(
        oracle_mod,
        candidate_mod,
        "power-boundaries",
        model,
        _power_boundary_examples(),
        batch_size,
        failure_dump_path=failure_dump_path,
        include_argmax_trace=include_argmax_trace,
        seed=seed,
        case_index_start=total,
    )
    total += _run_named_case_stream(
        oracle_mod,
        candidate_mod,
        "carry-chains",
        model,
        _carry_chain_examples(),
        batch_size,
        failure_dump_path=failure_dump_path,
        include_argmax_trace=include_argmax_trace,
        seed=seed,
        case_index_start=total,
    )
    total += _run_named_case_stream(
        oracle_mod,
        candidate_mod,
        "position-digit-sweep",
        model,
        _position_digit_sweep_examples(),
        batch_size,
        failure_dump_path=failure_dump_path,
        include_argmax_trace=include_argmax_trace,
        seed=seed,
        case_index_start=total,
    )
    total += _run_named_case_stream(
        oracle_mod,
        candidate_mod,
        "no-carry-random",
        model,
        _no_carry_random_examples(50_000, 101),
        batch_size,
        failure_dump_path=failure_dump_path,
        include_argmax_trace=include_argmax_trace,
        seed=101,
        case_index_start=total,
    )
    total += _run_named_case_stream(
        oracle_mod,
        candidate_mod,
        "heavy-carry-random",
        model,
        _heavy_carry_random_examples(50_000, 202),
        batch_size,
        failure_dump_path=failure_dump_path,
        include_argmax_trace=include_argmax_trace,
        seed=202,
        case_index_start=total,
    )
    if exhaustive_low_digits > 0:
        total += _run_named_case_stream(
            oracle_mod,
            candidate_mod,
            f"exhaustive-low-{exhaustive_low_digits}-digits",
            model,
            _low_digits_exhaustive_examples(exhaustive_low_digits),
            batch_size,
            failure_dump_path=failure_dump_path,
            include_argmax_trace=include_argmax_trace,
            seed=seed,
            case_index_start=total,
        )
    print(f"structured suite passed ({total} total cases)")


def run_random_test_batched(
    oracle_mod,
    candidate_mod,
    model,
    *,
    num_tests: int,
    batch_size: int,
    seed: int,
    failure_dump_path: str | None,
    include_argmax_trace: bool,
) -> None:
    tested = 0
    for batch in _chunked(_iter_random_examples(num_tests, seed), batch_size):
        expected = [oracle_mod._expected_output(a, b) for a, b in batch]
        actual = candidate_mod._generate_output_batch(model, batch)
        for i, ((a, b), exp, act) in enumerate(zip(batch, expected, actual)):
            if act != exp:
                _dump_failure(
                    failure_dump_path,
                    _build_failure_payload(
                        candidate_mod,
                        model,
                        suite="random-self-test",
                        case_index=tested + i,
                        seed=seed,
                        a=a,
                        b=b,
                        expected=exp,
                        got=act,
                        include_argmax_trace=include_argmax_trace,
                    ),
                )
                raise AssertionError(
                    f"Random mismatch at case {tested + i} (seed={seed}) for "
                    f"a={a:010d}, b={b:010d}: expected {exp}, got {act}"
                )
        tested += len(batch)
        print(f"self-test progress: {tested}/{num_tests}")


def run_random_accuracy_batched(
    oracle_mod,
    candidate_mod,
    model,
    *,
    num_tests: int,
    batch_size: int,
    seed: int,
    failure_dump_path: str | None,
    include_argmax_trace: bool,
    min_accuracy: float | None,
    max_recorded_failures: int,
    progress_every: int,
) -> float:
    tested = 0
    mismatches = 0
    recorded_failures: list[dict] = []
    next_progress = progress_every if progress_every > 0 else None

    for batch in _chunked(_iter_random_examples(num_tests, seed), batch_size):
        expected = [oracle_mod._expected_output(a, b) for a, b in batch]
        actual = candidate_mod._generate_output_batch(model, batch)
        for i, ((a, b), exp, act) in enumerate(zip(batch, expected, actual)):
            if act != exp:
                mismatches += 1
                if len(recorded_failures) < max_recorded_failures:
                    failure = {
                        "case_index": tested + i,
                        "a": f"{a:010d}",
                        "b": f"{b:010d}",
                        "expected": exp,
                        "got": act,
                        "input_tokens": candidate_mod._encode_addends_internal(a, b),
                    }
                    if include_argmax_trace:
                        failure["argmax_trace"] = _compute_argmax_trace(candidate_mod, model, a, b)
                    recorded_failures.append(failure)
        tested += len(batch)
        if progress_every <= 0:
            print(f"accuracy progress: {tested}/{num_tests}")
        else:
            while next_progress is not None and tested >= next_progress:
                print(f"accuracy progress: {tested}/{num_tests}")
                next_progress += progress_every

    exact = tested - mismatches
    accuracy = (exact / tested) if tested else 1.0
    print(
        f"random accuracy: exact={exact}, mismatches={mismatches}, total={tested}, "
        f"accuracy={accuracy:.8f}, seed={seed}"
    )
    if recorded_failures:
        print(f"recorded failures shown: {len(recorded_failures)} (max {max_recorded_failures})")
        for row in recorded_failures[: min(3, len(recorded_failures))]:
            print(
                "  mismatch case "
                f"{row['case_index']} a={row['a']} b={row['b']} expected={row['expected']} got={row['got']}"
            )
        if failure_dump_path:
            _dump_failure(
                failure_dump_path,
                {
                    "suite": "random-accuracy",
                    "seed": seed,
                    "num_tests": tested,
                    "exact": exact,
                    "mismatches": mismatches,
                    "accuracy": accuracy,
                    "recorded_failures": recorded_failures,
                },
            )
    if min_accuracy is not None and accuracy < min_accuracy:
        raise AssertionError(
            f"Random accuracy below threshold: required {min_accuracy:.8f}, got {accuracy:.8f}"
        )
    return accuracy


def compare_modules_equivalence(
    *,
    candidate_module_name: str,
    reference_module_name: str,
    num_tests: int,
    batch_size: int,
    seed: int,
    failure_dump_path: str | None,
    include_argmax_trace: bool,
) -> None:
    cand = _load_module(candidate_module_name)
    ref = _load_module(reference_module_name)
    cand_model = cand.build_magic_model()
    ref_model = ref.build_magic_model()
    tested = 0
    for batch in _chunked(_iter_random_examples(num_tests, seed), batch_size):
        cand_out = cand._generate_output_batch(cand_model, batch)
        ref_out = ref._generate_output_batch(ref_model, batch)
        for i, ((a, b), got_cand, got_ref) in enumerate(zip(batch, cand_out, ref_out)):
            if got_cand != got_ref:
                payload = {
                    "suite": "cross-module-equivalence",
                    "candidate_module": candidate_module_name,
                    "reference_module": reference_module_name,
                    "case_index": tested + i,
                    "seed": seed,
                    "a": f"{a:010d}",
                    "b": f"{b:010d}",
                    "candidate_output": got_cand,
                    "reference_output": got_ref,
                    "expected_arith_output": cand._expected_output(a, b),
                    "input_tokens": cand._encode_addends_internal(a, b),
                }
                if include_argmax_trace:
                    payload["candidate_argmax_trace"] = _compute_argmax_trace(cand, cand_model, a, b)
                    payload["reference_argmax_trace"] = _compute_argmax_trace(ref, ref_model, a, b)
                _dump_failure(failure_dump_path, payload)
                raise AssertionError(
                    "Cross-module mismatch at case "
                    f"{tested + i} (seed={seed}) for a={a:010d}, b={b:010d}: "
                    f"candidate={got_cand}, reference={got_ref}"
                )
        tested += len(batch)
        print(f"equivalence progress: {tested}/{num_tests}")
    print(
        f"equivalence passed ({num_tests} cases, seed {seed}): "
        f"{candidate_module_name} == {reference_module_name}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tests", type=int, default=8192)
    parser.add_argument(
        "--random-accuracy-num-tests",
        type=int,
        default=0,
        help="Counted random exact-match evaluation (continues past mismatches).",
    )
    parser.add_argument(
        "--min-random-accuracy",
        type=float,
        default=None,
        help="Optional minimum exact-match accuracy threshold for --random-accuracy-num-tests.",
    )
    parser.add_argument(
        "--max-recorded-failures",
        type=int,
        default=10,
        help="Max mismatches to retain in memory / failure dump for random accuracy mode.",
    )
    parser.add_argument(
        "--random-accuracy-progress-every",
        type=int,
        default=100_000,
        help="Progress interval for random accuracy mode (<=0 prints once per batch).",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--golden-seed", type=int, default=DEFAULT_GOLDEN_SEED)
    parser.add_argument("--structured-suite", action="store_true")
    parser.add_argument("--exhaustive-low-digits", type=int, default=0)
    parser.add_argument("--write-dataset", type=str, default=None)
    parser.add_argument("--dataset-size", type=int, default=0)
    parser.add_argument("--dataset-seed", type=int, default=DEFAULT_GOLDEN_SEED)
    parser.add_argument("--dataset-format", type=str, default="csv", choices=("csv", "tsv", "jsonl"))
    parser.add_argument("--dataset-progress-every", type=int, default=100_000)
    parser.add_argument("--skip-model-tests", action="store_true")
    parser.add_argument("--failure-dump", type=str, default="artifacts/last_failure.json")
    parser.add_argument("--include-argmax-trace", action="store_true")
    parser.add_argument("--expected-param-count", type=int, default=DEFAULT_EXPECTED_PARAM_COUNT)
    parser.add_argument("--skip-param-check", action="store_true")
    parser.add_argument(
        "--model-module",
        type=str,
        default="param_343",
        help="Module name used to build the model and generate outputs.",
    )
    parser.add_argument(
        "--oracle-module",
        type=str,
        default="param_343",
        help="Module name used for arithmetic expected outputs (anti-cheating oracle).",
    )
    parser.add_argument("--compare-reference-module", type=str, default=None)
    parser.add_argument(
        "--compare-candidate-module",
        type=str,
        default="param_343",
        help="Module name for candidate implementation in equivalence mode.",
    )
    parser.add_argument(
        "--compare-num-tests",
        type=int,
        default=100_000,
        help="Random cases for equivalence mode (same --golden-seed stream).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_tests < 0:
        raise ValueError("--num-tests must be >= 0")
    if args.random_accuracy_num_tests < 0:
        raise ValueError("--random-accuracy-num-tests must be >= 0")
    if args.compare_num_tests < 0:
        raise ValueError("--compare-num-tests must be >= 0")
    if args.exhaustive_low_digits < 0 or args.exhaustive_low_digits > 4:
        raise ValueError("--exhaustive-low-digits must be in [0, 4]")
    if args.dataset_size < 0:
        raise ValueError("--dataset-size must be >= 0")
    if args.max_recorded_failures < 0:
        raise ValueError("--max-recorded-failures must be >= 0")
    if args.min_random_accuracy is not None and not (0.0 <= args.min_random_accuracy <= 1.0):
        raise ValueError("--min-random-accuracy must be in [0, 1]")

    if args.write_dataset:
        if args.dataset_size <= 0:
            raise ValueError("--dataset-size must be > 0 when --write-dataset is set")
        write_addition_dataset(
            args.write_dataset,
            args.dataset_size,
            args.dataset_seed,
            fmt=args.dataset_format,
            progress_every=args.dataset_progress_every,
        )
        if args.skip_model_tests:
            return

    if args.compare_reference_module:
        if args.compare_num_tests == 0:
            print("equivalence mode skipped (0 cases)")
        else:
            compare_modules_equivalence(
                candidate_module_name=args.compare_candidate_module,
                reference_module_name=args.compare_reference_module,
                num_tests=args.compare_num_tests,
                batch_size=args.batch_size,
                seed=args.golden_seed,
                failure_dump_path=args.failure_dump,
                include_argmax_trace=args.include_argmax_trace,
            )
        return

    oracle_mod = _load_module(args.oracle_module)
    candidate_mod = _load_module(args.model_module)
    model = candidate_mod.build_magic_model()
    if args.skip_param_check:
        if hasattr(candidate_mod, "count_parameters"):
            print(f"parameter count (module helper): {candidate_mod.count_parameters(model.parameters())}")
        else:
            print("candidate module has no count_parameters(); falling back to named_parameters accounting")
            enforce_param_accounting(model, expected_param_count=None)
    else:
        enforce_param_accounting(model, expected_param_count=args.expected_param_count)
    if args.structured_suite:
        run_structured_test_suite(
            oracle_mod,
            candidate_mod,
            model,
            args.batch_size,
            args.exhaustive_low_digits,
            failure_dump_path=args.failure_dump,
            include_argmax_trace=args.include_argmax_trace,
            seed=args.golden_seed,
        )
    if args.num_tests > 0:
        run_random_test_batched(
            oracle_mod,
            candidate_mod,
            model,
            num_tests=args.num_tests,
            batch_size=args.batch_size,
            seed=args.golden_seed,
            failure_dump_path=args.failure_dump,
            include_argmax_trace=args.include_argmax_trace,
        )
        print(
            f"self-test passed ({args.num_tests} random cases, batch size {args.batch_size}, "
            f"seed {args.golden_seed})"
        )
    if args.random_accuracy_num_tests > 0:
        run_random_accuracy_batched(
            oracle_mod,
            candidate_mod,
            model,
            num_tests=args.random_accuracy_num_tests,
            batch_size=args.batch_size,
            seed=args.golden_seed,
            failure_dump_path=args.failure_dump,
            include_argmax_trace=args.include_argmax_trace,
            min_accuracy=args.min_random_accuracy,
            max_recorded_failures=args.max_recorded_failures,
            progress_every=args.random_accuracy_progress_every,
        )


if __name__ == "__main__":
    main()
