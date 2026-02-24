from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

CONDA_ENV = "param343-addition-wsl"
PYTHON = ["conda", "run", "-n", CONDA_ENV, "python"]
HARNESS = str(ROOT / "test_param_343.py")

CURRENT_BEST_BEFORE_BATCH = 323
SEED_DEFAULT = 123456789
SEED_EXTRA = 987654321
BATCH_SIZE = 4096

@dataclass
class Exp:
    exp_id: int
    module: str
    family: str  # 'baseline' or 'exp3'
    expected_params: int
    source_compare: str
    milestone: bool = False

EXPS = [
    Exp(38, "param_candidate_exp38_rank1_major_linears_baseline", "baseline", 259, "param_343"),
    Exp(39, "param_candidate_exp39_fixedscale_rmsnorms_baseline", "baseline", 310, "param_343"),
    Exp(40, "param_candidate_exp40_factored_embedding_baseline", "baseline", 323, "param_343"),
    Exp(41, "param_candidate_exp41_sparse_gate0_baseline", "baseline", 334, "param_343"),
    Exp(42, "param_candidate_exp42_mlx197_repro_baseline", "baseline", 197, "param_343", milestone=True),
    Exp(43, "param_candidate_exp43_mlx197_plus_vproj_share_baseline", "baseline", 190, "param_343"),
    Exp(44, "param_candidate_exp44_mlx197_plus_lmhead_r3_baseline", "baseline", 192, "param_343"),
    Exp(45, "param_candidate_exp45_mlx197_plus_vprojshare_lmheadr3_baseline", "baseline", 185, "param_343", milestone=True),
    Exp(46, "param_candidate_exp46_mlx197_recipe_on_exp3", "exp3", 195, "param_candidate_exp03_layer0_mlp_int2", milestone=True),
    Exp(47, "param_candidate_exp47_mlx195_plus_vproj_share_exp3", "exp3", 188, "param_candidate_exp03_layer0_mlp_int2"),
    Exp(48, "param_candidate_exp48_mlx195_plus_lmhead_r3_exp3", "exp3", 190, "param_candidate_exp03_layer0_mlp_int2"),
    Exp(49, "param_candidate_exp49_mlx195_plus_vprojshare_lmheadr3_exp3", "exp3", 183, "param_candidate_exp03_layer0_mlp_int2", milestone=True),
]


def run_cmd(args: list[str], timeout: int | None = None) -> dict:
    t0 = time.time()
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    dt = time.time() - t0
    return {
        "cmd": args,
        "returncode": p.returncode,
        "stdout": p.stdout,
        "stderr": p.stderr,
        "duration_sec": dt,
    }


def parse_build_meta(out: str) -> dict:
    try:
        return json.loads(out)
    except Exception:
        return {"parse_error": True, "raw": out}


def parse_equiv(out: str) -> dict:
    d: dict[str, object] = {}
    if "equivalence passed" in out:
        d["status"] = "pass"
        m = re.search(r"equivalence passed \((\d+) cases, seed (\d+)\)", out)
        if m:
            d["cases"] = int(m.group(1))
            d["seed"] = int(m.group(2))
    elif "equivalence mismatch" in out or "equivalence failed" in out:
        d["status"] = "fail"
    else:
        d["status"] = "unknown"
    return d


def parse_structured(out: str) -> dict:
    d: dict[str, object] = {}
    mpass = re.search(r"structured suite passed \((\d+) total cases\)", out)
    if mpass:
        d["status"] = "pass"
        d["total_cases"] = int(mpass.group(1))
        return d
    mfail = re.search(r"structured suite FAILED in ([^\n]+)", out)
    if mfail:
        d["status"] = "fail"
        d["where"] = mfail.group(1).strip()
    elif "structured suite" in out and "fail" in out.lower():
        d["status"] = "fail"
    else:
        d["status"] = "unknown"
    # try to capture first failing case line
    for line in out.splitlines():
        if "mismatch:" in line or ("expected" in line and "got" in line and "a=" in line):
            d.setdefault("detail", line.strip())
    return d


def parse_param_counts(out: str) -> dict:
    d: dict[str, object] = {}
    mp = re.search(r"parameter count \(named_parameters\): (\d+)", out)
    mb = re.search(r"buffer count \(named_buffers\): (\d+)", out)
    if mp:
        d["param_count"] = int(mp.group(1))
    if mb:
        d["buffer_count"] = int(mb.group(1))
    return d


def parse_random_accuracy(out: str) -> dict:
    d: dict[str, object] = {}
    m = re.search(r"random accuracy: exact=(\d+), mismatches=(\d+), total=(\d+), accuracy=([0-9.]+), seed=(\d+)", out)
    if m:
        d.update(
            status="ok",
            exact=int(m.group(1)),
            mismatches=int(m.group(2)),
            total=int(m.group(3)),
            accuracy=float(m.group(4)),
            seed=int(m.group(5)),
        )
    else:
        d["status"] = "unknown"
    return d


def short_cmd(args: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def build_meta_cmd(module: str) -> list[str]:
    code = (
        "import importlib,json,torch,traceback; "
        f"mod=importlib.import_module('{module}'); "
        "m=mod.build_magic_model(); "
        "meta=getattr(m,'_exact_reparam_metadata',{}); "
        "out={'param_count':sum(p.numel() for p in m.parameters()),'buffer_count':sum(b.numel() for b in m.buffers())," \
        "'meta':meta,'lm_head_rank32':int(torch.linalg.matrix_rank(m.lm_head.weight.detach().float()).item())," \
        "'lm_head_rank64':int(torch.linalg.matrix_rank(m.lm_head.weight.detach().double()).item())}; "
        "print(json.dumps(out, sort_keys=True))"
    )
    return PYTHON + ["-c", code]


def main() -> int:
    results: dict[str, object] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conda_env": CONDA_ENV,
        "experiments": [],
        "notes": [
            "Exact lm_head rank-3 factorization is expected to fail if lm_head rank64 != 3 (observed rank64=5)."
        ],
    }
    best_so_far = CURRENT_BEST_BEFORE_BATCH

    for exp in EXPS:
        rec: dict[str, object] = {
            "exp_id": exp.exp_id,
            "module": exp.module,
            "family": exp.family,
            "expected_params": exp.expected_params,
            "source_compare": exp.source_compare,
            "milestone": exp.milestone,
            "commands": [],
        }
        print(f"=== Exp{exp.exp_id} {exp.module} ===", flush=True)

        # Build + metadata
        cmd = build_meta_cmd(exp.module)
        r = run_cmd(cmd, timeout=180)
        rec["commands"].append({"kind": "build_meta", **r, "cmd_str": short_cmd(cmd)})
        if r["returncode"] != 0:
            rec["status"] = "build_fail"
            rec["failure_class"] = "implementation/assumption failure during exact build"
            rec["build_error"] = (r["stderr"] + "\n" + r["stdout"]).strip()
            results["experiments"].append(rec)
            continue
        meta = parse_build_meta(r["stdout"].strip().splitlines()[-1])
        rec["build_meta"] = meta
        if meta.get("param_count") != exp.expected_params:
            rec["status"] = "build_fail"
            rec["failure_class"] = "parameter-count mismatch"
            results["experiments"].append(rec)
            continue

        # Equivalence 10k
        eq_args = PYTHON + [HARNESS,
            "--compare-reference-module", exp.source_compare,
            "--compare-candidate-module", exp.module,
            "--compare-num-tests", "10000",
            "--batch-size", str(BATCH_SIZE),
            "--failure-dump", str(ART / f"exp{exp.exp_id}_equiv.json"),
        ]
        r = run_cmd(eq_args, timeout=1200)
        rec["commands"].append({"kind": "equivalence", **r, "cmd_str": short_cmd(eq_args)})
        rec["equivalence"] = {**parse_equiv(r["stdout"]), **parse_param_counts(r["stdout"])}
        if r["returncode"] != 0 or rec["equivalence"].get("status") != "pass":
            rec["status"] = "equiv_fail"
            rec["failure_class"] = "implementation bug / non-exact transformation"
            results["experiments"].append(rec)
            continue

        # Structured suite
        st_args = PYTHON + [HARNESS,
            "--model-module", exp.module,
            "--oracle-module", "param_343",
            "--structured-suite",
            "--num-tests", "0",
            "--expected-param-count", str(exp.expected_params),
            "--batch-size", str(BATCH_SIZE),
            "--failure-dump", str(ART / f"exp{exp.exp_id}_structured.json"),
        ]
        r = run_cmd(st_args, timeout=1800)
        rec["commands"].append({"kind": "structured", **r, "cmd_str": short_cmd(st_args)})
        rec["structured"] = {**parse_structured(r["stdout"]), **parse_param_counts(r["stdout"])}
        if r["returncode"] != 0 or rec["structured"].get("status") != "pass":
            rec["status"] = "structured_fail"
            rec["failure_class"] = "behavioral failure on structured suite"
            results["experiments"].append(rec)
            continue

        rec["status"] = "structured_pass"

        # Decide random ladder
        should_full_validate = False
        if exp.milestone:
            should_full_validate = True
        if exp.expected_params < best_so_far:
            should_full_validate = True
        # planned exact-lmhead experiments may build-fail before here anyway

        if should_full_validate:
            ra100k_args = PYTHON + [HARNESS,
                "--model-module", exp.module,
                "--oracle-module", "param_343",
                "--num-tests", "0",
                "--random-accuracy-num-tests", "100000",
                "--min-random-accuracy", "0.99",
                "--expected-param-count", str(exp.expected_params),
                "--batch-size", str(BATCH_SIZE),
                "--golden-seed", str(SEED_DEFAULT),
                "--failure-dump", str(ART / f"exp{exp.exp_id}_acc100k.json"),
            ]
            r = run_cmd(ra100k_args, timeout=3600)
            rec["commands"].append({"kind": "random_100k", **r, "cmd_str": short_cmd(ra100k_args)})
            rec["random_100k"] = {**parse_random_accuracy(r["stdout"]), **parse_param_counts(r["stdout"])}
            if r["returncode"] == 0 and rec["random_100k"].get("status") == "ok" and rec["random_100k"].get("accuracy", 0.0) >= 0.99:
                ra1m_args = PYTHON + [HARNESS,
                    "--model-module", exp.module,
                    "--oracle-module", "param_343",
                    "--num-tests", "0",
                    "--random-accuracy-num-tests", "1000000",
                    "--min-random-accuracy", "0.99",
                    "--expected-param-count", str(exp.expected_params),
                    "--batch-size", str(BATCH_SIZE),
                    "--golden-seed", str(SEED_DEFAULT),
                    "--failure-dump", str(ART / f"exp{exp.exp_id}_acc1m.json"),
                ]
                r = run_cmd(ra1m_args, timeout=14400)
                rec["commands"].append({"kind": "random_1m", **r, "cmd_str": short_cmd(ra1m_args)})
                rec["random_1m"] = {**parse_random_accuracy(r["stdout"]), **parse_param_counts(r["stdout"])}
                if r["returncode"] == 0 and rec["random_1m"].get("status") == "ok" and rec["random_1m"].get("accuracy", 0.0) >= 0.99:
                    raextra_args = PYTHON + [HARNESS,
                        "--model-module", exp.module,
                        "--oracle-module", "param_343",
                        "--num-tests", "0",
                        "--random-accuracy-num-tests", "100000",
                        "--min-random-accuracy", "0.99",
                        "--expected-param-count", str(exp.expected_params),
                        "--batch-size", str(BATCH_SIZE),
                        "--golden-seed", str(SEED_EXTRA),
                        "--failure-dump", str(ART / f"exp{exp.exp_id}_acc100k_seed{SEED_EXTRA}.json"),
                    ]
                    r = run_cmd(raextra_args, timeout=3600)
                    rec["commands"].append({"kind": "random_100k_extra", **r, "cmd_str": short_cmd(raextra_args)})
                    rec["random_100k_extra"] = {**parse_random_accuracy(r["stdout"]), **parse_param_counts(r["stdout"])}
            if exp.expected_params < best_so_far and rec.get("random_100k", {}).get("status") == "ok" and rec.get("random_100k", {}).get("accuracy", 0.0) >= 0.99:
                best_so_far = exp.expected_params
                rec["new_best_after_validation"] = best_so_far

        results["experiments"].append(rec)
        (ART / "pivot_exp38_49_results.partial.json").write_text(json.dumps(results, indent=2))

    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (ART / "pivot_exp38_49_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({
        "written": str(ART / "pivot_exp38_49_results.json"),
        "num_experiments": len(results["experiments"]),
        "best_after_batch": best_so_far,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
