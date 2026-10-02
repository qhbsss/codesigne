"""Measure an ideal coordinated cache warm-up before one P1 stage.

This diagnostic reads each distinct HBM cache line used by the selected stage
exactly once, distributed over its active workers, then places a barrier before
the unmodified stage.  It quantifies the opportunity from avoiding concurrent
unmerged cache misses without changing the numerical program.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from codesign.challenge.abi import build_layout
from codesign.challenge.hardware import Hardware
from codesign.challenge.isa import iter_parse
from codesign.challenge.perf import _lines
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.runner import required_hbm_words
from codesign.challenge.workload import MODELS


def _operand(space: str, offset: int, count: int, wg: str | None, lane: int = 0):
    return {"space": space, "offset": offset, "count": count, "wg": wg, "lane": lane}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", type=int)
    parser.add_argument("--peer", type=int, help="Second prompt-batch segment to run in lockstep")
    parser.add_argument("--program", type=Path, default=Path("programs/M1_P1.asm"))
    parser.add_argument("--hardware", type=Path, default=Path("hardware.json"))
    args = parser.parse_args()

    text = args.program.read_text()
    begins = [line for line in text.splitlines() if line.startswith("WG.BEGIN ")]
    ends = [line for line in text.splitlines() if line.startswith("WG.END ")]
    pieces = text.split("BARRIER ")

    def segment_lines(index: int) -> list[str]:
        selected = pieces[index].splitlines()
        if index:
            selected = selected[1:]
        return [
            line for line in selected
            if line and not line.startswith(("WG.BEGIN ", "WG.END ", "STEP.COMMIT "))
        ]

    stage_lines = segment_lines(args.segment)
    if args.peer is not None:
        stage_lines += segment_lines(args.peer)
    stage_text = "\n".join(begins + stage_lines + ends) + "\n"

    cache_lines: set[int] = set()
    workers: set[str] = set()
    for ins in iter_parse(stage_text):
        event = ins.args.get("event")
        if isinstance(event, str) and "_e" in event:
            workers.add(event.split("_e", 1)[0])
        if ins.op == "LD" and ins.args["src"].get("space") == "HBM":
            cache_lines.update(_lines(ins.args["src"]))
    ordered_workers = sorted(workers)
    if not ordered_workers:
        raise ValueError("Stage has no active workers")

    # Convert unique lines into contiguous runs capped by the 32 KiB SH quota.
    runs: list[tuple[int, int]] = []
    ordered = sorted(cache_lines)
    start = previous = ordered[0]
    for line in ordered[1:] + [None]:
        if line is not None and line == previous + 1 and line - start < 512:
            previous = line
            continue
        runs.append((start, previous + 1))
        if line is not None:
            start = previous = line

    prefetch = []
    for index, (lo, hi) in enumerate(runs):
        worker = ordered_workers[index % len(ordered_workers)]
        count = (hi - lo) * 16
        args_json = {
            "src": _operand("HBM", lo * 16, count, None),
            "dst": _operand("SH", 0, count, worker),
            "event": f"pf_s{args.segment}_{index}",
        }
        prefetch.append("LD " + json.dumps(args_json, separators=(",", ":")))
    prefetch.append(
        "BARRIER " + json.dumps({"wgs": ordered_workers, "events": []}, separators=(",", ":"))
    )

    program = "\n".join(begins + prefetch + stage_lines + ends) + "\n"
    hardware = Hardware.from_dict(json.loads(args.hardware.read_text()))
    layout = build_layout(MODELS["M1"], "P1")
    started = perf_counter()
    timing = asdict(
        estimate_pipeline(
            hardware,
            iter_parse(program),
            required_hbm_words(program, layout),
            None,
        )
    )
    keys = (
        "cycles", "peak_window_power_w", "hbm_read_bytes", "hbm_write_bytes",
        "cache_hits", "cache_misses", "instruction_count",
    )
    print(json.dumps({
        "segment": args.segment,
        "peer": args.peer,
        "workers": ordered_workers,
        "unique_prefetch_lines": len(cache_lines),
        "prefetch_instructions": len(runs),
        "elapsed_seconds": perf_counter() - started,
        **{key: timing[key] for key in keys},
    }, indent=2))


if __name__ == "__main__":
    main()
