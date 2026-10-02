"""Summarize expanded HBM line reads for one barrier-delimited P1 segment."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import prod
from pathlib import Path

from codesign.challenge.abi import build_layout
from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.perf import _lines
from codesign.challenge.workload import MODELS

from .rough_model import profile_program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", type=int)
    parser.add_argument("--program", type=Path, default=Path("programs/M1_P1.asm"))
    parser.add_argument("--hardware", type=Path, default=Path("hardware.json"))
    args = parser.parse_args()

    text = args.program.read_text()
    begins = [line for line in text.splitlines() if line.startswith("WG.BEGIN ")]
    ends = [line for line in text.splitlines() if line.startswith("WG.END ")]
    pieces = text.split("BARRIER ")
    lines = pieces[args.segment].splitlines()
    if args.segment:
        lines = lines[1:]
    lines = [
        line
        for line in lines
        if line and not line.startswith(("WG.BEGIN ", "WG.END ", "STEP.COMMIT "))
    ]
    mini = "\n".join(begins + lines + ends) + "\n"
    profile = profile_program(
        Hardware.from_dict(json.loads(args.hardware.read_text())), mini
    )

    layout = build_layout(MODELS["M1"], "P1")
    spans = []
    for name, item in layout.symbols.items():
        lo = item.address // 4
        spans.append((lo, lo + prod(item.shape), name))

    def owner(word: int) -> str:
        for lo, hi, name in spans:
            if lo <= word < hi:
                return name
        return "scratch"

    requests = Counter()
    uniques: dict[str, set[int]] = defaultdict(set)
    by_worker = Counter()
    for ins in iter_parse(mini):
        if ins.op != "LD" or ins.args["src"].get("space") != "HBM":
            continue
        event = ins.args.get("event", "unknown")
        worker = event.split("_e", 1)[0]
        for line in _lines(ins.args["src"]):
            name = owner(line * 16)
            requests[name] += 1
            uniques[name].add(line)
            by_worker[worker] += 1

    print(f"segment={args.segment} total_requests={sum(requests.values())} "
          f"unique_lines={len(set().union(*uniques.values()))}")
    maxima = {
        key: max(item.get(key, 0) for item in profile.sm_demands.values())
        for key in ("issue", "dma", "noc_in", "noc_out", "tc", "vec", "sfu", "rf_read", "rf_write")
    }
    print("static", maxima, "shared", profile.shared_demands)
    for name, count in requests.most_common():
        print(f"{name:28s} requests={count:7d} unique={len(uniques[name]):6d} "
              f"dup={count / max(1, len(uniques[name])):6.2f}x")
    print("workers", dict(sorted(by_worker.items())))


if __name__ == "__main__":
    main()
