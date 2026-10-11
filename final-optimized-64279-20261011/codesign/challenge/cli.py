"""Public command entry point for candidate two-case challenge runs."""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None

from .baseline import generate_m1_p1
from .hardware import Hardware, baseline_hardware, cost_model
from .release import CANDIDATE_BASELINE_SHA256, FROZEN_BASELINE_SHA256
from .runner import (
    REPORT_VERSION,
    TIMING_MODEL,
    check_case,
    digest_bytes,
    estimate_case,
    evaluate_submission,
    provenance,
    seed_digest,
    write_report,
)
from .score import CASES


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_repo_root()))
    except ValueError:
        return str(resolved)


def _resolve_artifact(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = _repo_root()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Relative artifact path escapes repository")
    return resolved


def runtime_environment() -> dict:
    """Record the numerical runtime and reference host, outside semantic hashes."""
    cpu_model = platform.processor()
    if sys.platform == "darwin":
        model = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            check=False,
            text=True,
        )
        if model.returncode == 0 and model.stdout.strip():
            cpu_model = model.stdout.strip()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "clock_hz": cost_model()["clock_hz"],
    }


def runtime_observation(start_ns: int) -> dict:
    """Wall time and process high-water RSS are measurements, not replay invariants."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else 0
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    return {
        "elapsed_seconds": (time.perf_counter_ns() - start_ns) / 1e9,
        "peak_process_rss_bytes": rss_bytes,
    }


def stable_report(report: dict) -> dict:
    """Keep exact numerical fields while treating host details as observations."""
    if not isinstance(report.get("runtime_observation"), dict) or set(
        report["runtime_observation"]
    ) != {"elapsed_seconds", "peak_process_rss_bytes"}:
        raise ValueError("Missing runtime observation")
    observation = report["runtime_observation"]
    if (
        type(observation["elapsed_seconds"]) not in (float, int)
        or not math.isfinite(observation["elapsed_seconds"])
        or observation["elapsed_seconds"] < 0
        or type(observation["peak_process_rss_bytes"]) is not int
        or observation["peak_process_rss_bytes"] < 0
    ):
        raise ValueError("Invalid runtime observation")
    stable = {
        name: value
        for name, value in report.items()
        if name not in {"runtime_observation", "presentation_sha256"}
    }
    environment = stable.get("runtime_environment")
    if not isinstance(environment, dict) or not {"python", "numpy", "clock_hz"} <= set(environment):
        raise ValueError("Missing numerical runtime environment")
    stable["runtime_environment"] = {
        name: environment[name] for name in ("python", "numpy", "clock_hz")
    }
    return stable


def _programs(paths: dict[str, Path]) -> dict[str, str]:
    return {case: paths[case].read_text(encoding="utf-8") for case in CASES}


def _hardware(path: Path) -> Hardware:
    return Hardware.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _baseline_programs() -> dict[str, str]:
    # Imported when used so this command can coexist with an in-progress D1 compiler.
    from .baseline import generate_m1_d1

    return {"M1_P1": generate_m1_p1()[0], "M2_D1": generate_m1_d1()[0]}


def _baseline_manifest(
    hardware: Hardware, programs: dict[str, str], seed: int, status: str, started: int
) -> dict:
    checks = {case: check_case(case, seed, hardware, programs[case]) for case in CASES}
    if not all(check["passed"] for check in checks.values()):
        raise ValueError("Generated baseline failed its functional check")
    metrics = {case: estimate_case(case, hardware, programs[case]) for case in CASES}
    return {
        "version": REPORT_VERSION,
        "timing_model": TIMING_MODEL,
        "status": status,
        "baseline_cycles": {case: metrics[case]["cycles"] for case in CASES},
        "baseline_metrics": metrics,
        "functional_checks": checks,
        "functional_seed_sha256": seed_digest(seed),
        "provenance": provenance(hardware, programs),
        "clock_hz": cost_model()["clock_hz"],
        "runtime_environment": runtime_environment(),
        "runtime_observation": runtime_observation(started),
    }


def generate_baseline(output_dir: Path, seed: int = 7) -> dict:
    """Generate, functionally check, and time a candidate baseline."""
    started = time.perf_counter_ns()
    seed_digest(seed)
    if output_dir.exists():
        raise FileExistsError(f"Baseline output already exists: {output_dir}")
    hardware = baseline_hardware()
    programs = _baseline_programs()
    manifest = _baseline_manifest(hardware, programs, seed, "candidate", started)
    output_dir.mkdir(parents=True, exist_ok=False)
    for case in CASES:
        (output_dir / f"{case}.asm").write_text(programs[case], encoding="utf-8")
    (output_dir / "hardware.json").write_text(
        json.dumps(hardware.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "baseline_manifest.json", manifest)
    return manifest


def _load_baseline(path: Path, required_status: str | None = None) -> dict:
    raw = path.read_bytes()
    digest = digest_bytes(raw)
    trusted = {
        "candidate": CANDIDATE_BASELINE_SHA256,
        "frozen": FROZEN_BASELINE_SHA256,
    }
    manifest = json.loads(raw)
    status = manifest.get("status")
    if (
        type(status) is not str
        or status not in trusted
        or trusted[status] is None
        or digest != trusted[status]
    ):
        raise ValueError("Untrusted baseline digest")
    if required_status is not None and status != required_status:
        raise ValueError("Baseline status differs from required release")
    if manifest.get("version") != REPORT_VERSION or manifest.get("timing_model") != TIMING_MODEL:
        raise ValueError("Incompatible baseline manifest")
    current = provenance(baseline_hardware(), _baseline_programs())
    # The frozen file pins the numerical baseline and its inputs. A scorer-only
    # performance fix changes its implementation hash without changing those
    # inputs or the baseline cycles used by the score formula.
    input_keys = ("hardware_sha256", "program_sha256", "cost_sha256", "workload_sha256")
    if any(manifest["provenance"].get(key) != current[key] for key in input_keys):
        raise ValueError("Baseline cost, workload, hardware or program hash changed")
    if set(manifest.get("baseline_cycles", {})) != set(CASES) or any(
        type(value) is not int or value <= 0 for value in manifest["baseline_cycles"].values()
    ):
        raise ValueError("Invalid baseline cycles")
    if manifest.get("clock_hz") != cost_model()["clock_hz"] or any(
        manifest.get("baseline_metrics", {}).get(case, {}).get("cycles")
        != manifest["baseline_cycles"][case]
        for case in CASES
    ):
        raise ValueError("Baseline metrics or clock differ")
    checks = manifest.get("functional_checks", {})
    if set(checks) != set(CASES) or any(
        checks[case].get("seed_sha256") != manifest.get("functional_seed_sha256")
        or checks[case].get("passed") is not True
        for case in CASES
    ):
        raise ValueError("Baseline functional evidence differs")
    if sys.version_info[:2] != (3, 12) or int(np.__version__.split(".")[0]) != 2:
        raise ValueError("The current starter requires Python 3.12 and NumPy 2.x")
    stable_report(manifest)
    return manifest


def _evaluate(
    mode: str,
    hardware_path: Path,
    paths: dict[str, Path],
    seeds: list[int],
    baseline_path: Path | None,
) -> dict:
    started = time.perf_counter_ns()
    hardware = _hardware(hardware_path)
    programs = _programs(paths)
    baseline = _load_baseline(baseline_path) if baseline_path is not None else None
    report = evaluate_submission(mode, hardware, programs, seeds, baseline)
    package = Path(__file__).resolve().parent
    report["trust_policy_sha256"] = {
        name: digest_bytes((package / name).read_bytes())
        for name in ("cli.py", "official.py", "release.py")
    }
    report["presentation_sha256"] = digest_bytes((package / "report_view.py").read_bytes())
    report["artifact_paths"] = {
        "hardware": _artifact_ref(hardware_path),
        "programs": {case: _artifact_ref(paths[case]) for case in CASES},
        "baseline": _artifact_ref(baseline_path) if baseline_path is not None else None,
    }
    report["baseline_manifest_sha256"] = (
        digest_bytes(baseline_path.read_bytes()) if baseline_path is not None else None
    )
    report["baseline_status"] = baseline["status"] if baseline is not None else None
    report["runtime_environment"] = runtime_environment()
    report["runtime_observation"] = runtime_observation(started)
    return report


def verify_report(report_path: Path, seeds: list[int]) -> dict:
    """Recompute a saved report; never replace it or its baseline."""
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    if saved.get("version") != REPORT_VERSION:
        raise ValueError("Unsupported report version")
    if saved.get("seed_sha256") != [seed_digest(seed) for seed in seeds]:
        raise ValueError("Verification seed list differs")
    artifacts = saved["artifact_paths"]
    paths = {case: _resolve_artifact(artifacts["programs"][case]) for case in CASES}
    baseline_path = _resolve_artifact(artifacts["baseline"]) if artifacts["baseline"] else None
    current = _evaluate(
        saved["mode"], _resolve_artifact(artifacts["hardware"]), paths, seeds, baseline_path
    )
    # Legacy reports saved absolute paths even for files inside this checkout.
    # The resolved files and provenance were checked by _evaluate; preserve the
    # original path representation when comparing that historical report.
    current["artifact_paths"] = artifacts
    if stable_report(current) != stable_report(saved):
        raise ValueError("Report reproduction differs from saved result")
    return {
        "verified": True,
        "report_sha256": digest_bytes(report_path.read_bytes()),
        "saved_runtime_observation": saved["runtime_observation"],
        "replay_runtime_observation": current["runtime_observation"],
    }


def verify_baseline(manifest_path: Path, seeds: list[int]) -> dict:
    """Recheck a pinned baseline and its sibling artifacts without writing to them."""
    if len(seeds) != 1:
        raise ValueError("Baseline verification requires its original seed")
    started = time.perf_counter_ns()
    saved = _load_baseline(manifest_path)
    if saved.get("functional_seed_sha256") != seed_digest(seeds[0]):
        raise ValueError("Baseline verification seed differs")
    directory = manifest_path.parent
    hardware = _hardware(directory / "hardware.json")
    programs = _programs({case: directory / f"{case}.asm" for case in CASES})
    if provenance(hardware, programs) != saved["provenance"]:
        raise ValueError("Baseline artifact hashes differ")
    current = _baseline_manifest(hardware, programs, seeds[0], saved["status"], started)
    if stable_report(current) != stable_report(saved):
        raise ValueError("Baseline reproduction differs from saved result")
    return {
        "verified": True,
        "manifest_sha256": digest_bytes(manifest_path.read_bytes()),
        "saved_runtime_observation": saved["runtime_observation"],
        "replay_runtime_observation": current["runtime_observation"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-case programmable Transformer challenge")
    parser.add_argument(
        "--verify", type=Path, metavar="REPORT", help="Recompute an existing report read-only"
    )
    parser.add_argument("--seed", type=int, action="append", dest="verify_seeds")
    commands = parser.add_subparsers(dest="command")
    generate = commands.add_parser("generate-baseline")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--seed", type=int, action="append")
    for name in ("estimate", "check", "grade"):
        command = commands.add_parser(name)
        command.add_argument("--hardware", type=Path, required=True)
        command.add_argument("--program-p1", type=Path, required=True)
        command.add_argument("--program-d1", type=Path, required=True)
        command.add_argument("--report", type=Path, required=True)
        command.add_argument(
            "--seed", type=int, action="append", help="Public or externally supplied test seed"
        )
        if name == "grade":
            command.add_argument("--baseline", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.verify is not None:
        saved = json.loads(args.verify.read_text(encoding="utf-8"))
        verify = verify_baseline if "baseline_cycles" in saved else verify_report
        result = verify(args.verify, args.verify_seeds or [])
    elif args.command == "generate-baseline":
        result = generate_baseline(args.out, (args.seed or [7])[0])
    elif args.command in {"estimate", "check", "grade"}:
        if args.report.exists():
            raise FileExistsError(f"Report already exists: {args.report}")
        seeds = args.seed or ([] if args.command == "estimate" else [7])
        paths = {"M1_P1": args.program_p1, "M2_D1": args.program_d1}
        result = _evaluate(
            args.command,
            args.hardware,
            paths,
            seeds,
            getattr(args, "baseline", None),
        )
        write_report(args.report, result)
    else:
        parser.error("Choose a command or --verify")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    if args.command in {"estimate", "check", "grade"}:
        if any("timing" not in case for case in result["cases"].values()):
            return 1
        if args.command in {"check", "grade"} and not all(
            case["functional_passed"] for case in result["cases"].values()
        ):
            return 1
    return 0
