#!/usr/bin/env python3
"""Bounded process scheduler for trusted vNext research cases (not student submissions)."""

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def worker(args):
    config, output = args.config, args.output
    output.mkdir(parents=True, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(
        resource.RLIMIT_CPU, (args.timeout * args.threads, args.timeout * args.threads + 1)
    )
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (args.memory_gib * 1024**3,) * 2)
    source = json.loads(config.read_text())
    source["reference_threads"] = args.threads
    write(output / "config.json", source)
    command = [
        str(args.binary),
        args.mode,
        str(source["program"] if args.mode == "evaluate" else output / "config.json"),
        str(args.seed),
        str(output / "evaluation"),
    ]
    start = time.monotonic()
    error, code, result = None, None, None
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
                error = "wall timeout"
        except (OSError, subprocess.SubprocessError) as exc:
            error = str(exc)
    result_path = output / "evaluation" / "result.json"
    try:
        if result_path.stat().st_size > 1024**2:
            raise ValueError("result exceeds 1 MiB")
        result = json.loads(result_path.read_text())
        if not isinstance(result, dict):
            raise ValueError("result is not an object")
    except (OSError, ValueError) as exc:
        error = error or str(exc)
        result = None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    complete = code == 0 and not error and result is not None and result.get("status") == "complete"
    evaluation = (result or {}).get("evaluation") or {}
    checked = (
        complete and args.mode in {"check", "evaluate"} and evaluation.get("comparison") is not None
    )
    receipt = {
        "status": "complete" if complete else "failed",
        "exit_code": code,
        "error": error,
        "host_seconds": time.monotonic() - start,
        "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "checked": checked,
        "feasible": checked and evaluation.get("report", {}).get("power_pass") is True,
        "limits": {
            "wall_seconds": args.timeout,
            "reserved_gib": args.memory_gib,
            "threads": args.threads,
            "address_space_enforced": sys.platform.startswith("linux"),
        },
        "command": command,
    }
    write(output / "runner.json", receipt)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="?")
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu-budget", type=int, default=4)
    parser.add_argument("--memory-gib", type=int, default=4, help="reservation/limit per worker")
    parser.add_argument("--total-memory-gib", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["check", "estimate", "evaluate"], default="check")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (
        1 <= args.jobs <= 32
        and 1 <= args.threads <= 8
        and 1 <= args.cpu_budget <= 256
        and 1 <= args.memory_gib <= 16
        and 1 <= args.total_memory_gib <= 256
        and 1 <= args.timeout <= 1800
        and 0 <= args.seed < 2**64
    ):
        parser.error("invalid worker budget or seed")
    args.binary = args.binary.resolve(strict=True)
    if args.worker:
        return worker(args)
    if args.configs is None:
        parser.error("configs directory required")
    slots = min(
        args.jobs, args.cpu_budget // args.threads, args.total_memory_gib // args.memory_gib
    )
    if slots < 1:
        parser.error("budget cannot reserve one worker")
    configs = sorted(args.configs.resolve().glob("*.json"))
    if not configs or len(configs) > 64:
        parser.error("require 1..64 configuration files")
    for config in configs:
        if config.stat().st_size > 1024**2:
            parser.error("config exceeds 1 MiB")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    script = Path(__file__).resolve()
    write(
        output / "manifest.json",
        {
            "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            "scheduler_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "seed": args.seed,
            "slots": slots,
            "threads": args.threads,
            "reserved_memory_gib": slots * args.memory_gib,
            "cpu_budget": args.cpu_budget,
            "configs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in configs},
            "official_score": None,
        },
    )
    pending, running, receipts = list(configs), {}, {}
    start, peak_jobs = time.monotonic(), 0
    try:
        while pending or running:
            while pending and len(running) < slots:
                config = pending.pop(0)
                dest = output / config.stem
                command = [
                    sys.executable,
                    str(script),
                    str(dest),
                    "--worker",
                    "--config",
                    str(config),
                    "--binary",
                    str(args.binary),
                    "--threads",
                    str(args.threads),
                    "--memory-gib",
                    str(args.memory_gib),
                    "--timeout",
                    str(args.timeout),
                    "--seed",
                    str(args.seed),
                    "--mode",
                    args.mode,
                ]
                process = subprocess.Popen(command, start_new_session=True)
                running[config.stem] = (process, time.monotonic(), dest)
                peak_jobs = max(peak_jobs, len(running))
            for name, (process, started, dest) in list(running.items()):
                if process.poll() is None and time.monotonic() - started > args.timeout + 10:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                if process.poll() is None:
                    continue
                try:
                    receipt = json.loads((dest / "runner.json").read_text())
                    if not isinstance(receipt, dict):
                        raise ValueError("worker receipt is not an object")
                except (OSError, ValueError) as exc:
                    receipt = {
                        "status": "failed",
                        "error": str(exc),
                        "exit_code": process.returncode,
                    }
                if process.returncode != 0:
                    receipt["status"] = "failed"
                receipts[name] = receipt
                print(
                    f"{name}: {receipt['status']}, {receipt.get('host_seconds', 0):.2f}s",
                    flush=True,
                )
                del running[name]
                write(output / "progress.json", receipts)
            if running:
                time.sleep(0.05)
    finally:
        for process, _, _ in running.values():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    complete = len(receipts) == len(configs) and all(
        r["status"] == "complete" for r in receipts.values()
    )
    summary = {
        "status": "complete" if complete else "failed",
        "host_seconds": time.monotonic() - start,
        "peak_jobs": peak_jobs,
        "all_checked": all(r.get("checked", False) for r in receipts.values()),
        "all_feasible": all(r.get("feasible", False) for r in receipts.values()),
        "jobs": receipts,
        "official_score": None,
    }
    write(output / "summary.json", summary)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
