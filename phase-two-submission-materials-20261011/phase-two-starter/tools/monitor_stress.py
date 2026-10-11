#!/usr/bin/env python3
"""Linux process-tree RSS sampler for a bounded trusted evaluation command."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def tree_rss(pid):
    pending, seen, rss = [pid], set(), 0
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        try:
            lines = Path(f"/proc/{item}/status").read_text().splitlines()
            rss += next(
                (int(line.split()[1]) * 1024 for line in lines if line.startswith("VmRSS:")),
                0,
            )
            pending.extend(map(int, Path(f"/proc/{item}/task/{item}/children").read_text().split()))
        except (OSError, ValueError):
            pass  # The process may have exited between procfs reads.
    return rss, len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not Path("/proc/self/status").exists():
        parser.error("Linux and a command are required")
    with args.receipt.open("x") as output:
        start = time.monotonic()
        process = subprocess.Popen(command, start_new_session=True)
        peak, count, samples, timed_out = 0, 0, 0, False
        try:
            while process.poll() is None:
                rss, children = tree_rss(process.pid)
                peak, count = max(peak, rss), max(count, children)
                samples += 1
                if time.monotonic() - start > 1800:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
        json.dump(
            {
                "command": command,
                "wall_seconds": time.monotonic() - start,
                "exit_code": code,
                "timed_out": timed_out,
                "sampled_aggregate_rss_bytes": peak,
                "max_processes": count,
                "samples": samples,
                "note": "Sum of RSS, shared pages counted per process; 100 ms sampling may miss peaks. Not PSS or a hard memory limit.",
            },
            output,
            indent=2,
        )
        output.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
