#!/usr/bin/env python3
"""Snapshot and grade five version-pinned static programs with bounded parallel Rust workers."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


def load(path):
    if path.stat().st_size > 1024**2:
        raise ValueError(f"report too large: {path}")
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("programs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--cpu-budget", type=int, default=4)
    parser.add_argument("--total-memory-gib", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**64 or args.timeout <= 0:
        parser.error("seed must fit u64 and timeout must be positive")
    if args.cpu_budget < args.jobs or args.total_memory_gib < 4 * args.jobs:
        parser.error("jobs exceed CPU or memory budget")
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [str(binary), "contract", "-", str(args.seed), str(output / "contract")],
        check=True,
        timeout=10,
    )
    contract = load(output / "contract/result.json")
    cases = contract["evaluation"]["cases"]
    file_limit = contract["evaluation"].get("limits", {}).get("program_file_bytes", 1024**3)
    if not isinstance(file_limit, int) or not 1 <= file_limit <= 4 * 1024**3:
        raise ValueError("invalid evaluator file limit")
    if not all(case["reference_cycles"] > 0 for case in cases.values()):
        raise ValueError("reference cycles have not been frozen")
    programs = output / "programs"
    configs = output / "configs"
    programs.mkdir()
    configs.mkdir()
    hardware, hashes = None, {}
    for name, case in cases.items():
        source = args.programs / (name + ".jsonl")
        if source.stat().st_size > file_limit:
            raise ValueError("program exceeds evaluator file limit")
        target = programs / source.name
        with source.open("rb") as reader, target.open("xb") as writer:
            remaining = file_limit
            while chunk := reader.read(min(1024**2, remaining + 1)):
                remaining -= len(chunk)
                if remaining < 0:
                    raise ValueError("program grew beyond evaluator file limit")
                writer.write(chunk)
        if target.stat().st_size > file_limit:
            raise ValueError("program grew beyond evaluator file limit")
        with target.open("rb") as reader:
            line = reader.readline(8 * 1024**2 + 1)
            if len(line) > 8 * 1024**2:
                raise ValueError("header too large")
            header = json.loads(line)
            reader.seek(0)
            hashes[name] = hashlib.file_digest(reader, "sha256").hexdigest()
        if any(header.get(key) != case[key] for key in ("shape", "batch", "decode")):
            raise ValueError(f"fixed workload mismatch: {name}")
        if (
            header.get("contract") != contract["contract"]
            or header.get("model") != contract["model"]
        ):
            raise ValueError("wrong contract/model")
        if hardware is not None and header["hardware"] != hardware:
            raise ValueError("all five programs must use identical hardware")
        hardware = header["hardware"]
        (configs / (name + ".json")).write_text(json.dumps({"program": str(target)}) + "\n")
    scheduler = Path(__file__).with_name("evaluate_explore.py")
    code = subprocess.run(
        [
            sys.executable,
            str(scheduler),
            str(configs),
            str(output / "workers"),
            "--binary",
            str(binary),
            "--mode",
            "evaluate",
            "--seed",
            str(args.seed),
            "--jobs",
            str(args.jobs),
            "--cpu-budget",
            str(args.cpu_budget),
            "--memory-gib",
            "4",
            "--total-memory-gib",
            str(args.total_memory_gib),
            "--timeout",
            str(args.timeout),
        ],
        check=False,
    ).returncode
    summary = load(output / "workers/summary.json")
    valid = code == 0 and summary["all_checked"] and summary["all_feasible"]
    values, log_score = {}, 0.0
    if valid:
        for name, case in cases.items():
            result = load(output / "workers" / name / "evaluation/result.json")
            evaluation = result["evaluation"]
            header = evaluation["header"]
            if (
                result["source_sha256"] != contract["source_sha256"]
                or evaluation["program_sha256"] != hashes[name]
                or header["hardware"] != hardware
                or any(header[key] != case[key] for key in ("shape", "batch", "decode"))
            ):
                raise ValueError("program/evaluator changed after admission")
            cycles = evaluation["report"]["stats"]["cycles"]
            if cycles <= 0:
                raise ValueError("zero-cycle result")
            log_score += case["weight"] * math.log(case["reference_cycles"] / cycles)
            values[name] = {"cycles": cycles, "program_sha256": hashes[name]}
    receipt = {
        "status": "complete" if valid else "failed",
        "contract": contract["contract"],
        "model": contract["model"],
        "source_sha256": contract["source_sha256"],
        "seed": args.seed,
        "score": 1000 * math.exp(log_score) if valid else None,
        "cases": values,
        "hardware": hardware,
        "host_seconds": summary["host_seconds"],
        "note": "Local reproducible result, not a signed server receipt. Organizer reruns the same program hashes with private seeds.",
    }
    (output / "grade.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
