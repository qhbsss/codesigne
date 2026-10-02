"""Run one accepted P1 barrier segment through the exact public scheduler."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from codesign.challenge.hardware import Hardware
from codesign.challenge.abi import build_layout
from codesign.challenge.isa import iter_parse
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.runner import required_hbm_words
from codesign.challenge.workload import MODELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segment", type=int)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--hardware", type=Path)
    args = parser.parse_args()
    root = Path("checkpoints/accepted-28885")
    hardware_path = args.hardware or (root / "hardware.json")
    hardware = Hardware.from_dict(json.loads(hardware_path.read_text()))
    program_path = args.program or (root / "programs/M1_P1.asm")
    text = program_path.read_text()
    begins = [line for line in text.splitlines() if line.startswith("WG.BEGIN ")]
    ends = [line for line in text.splitlines() if line.startswith("WG.END ")]
    pieces = text.split("BARRIER ")
    piece = pieces[args.segment]
    lines = piece.splitlines()
    if args.segment:
        lines = lines[1:]
    lines = [
        line
        for line in lines
        if line and not line.startswith(("WG.BEGIN ", "WG.END ", "STEP.COMMIT "))
    ]
    program = "\n".join(begins + lines + ends) + "\n"
    started = perf_counter()
    layout = build_layout(MODELS["M1"], "P1")
    timing = asdict(
        estimate_pipeline(
            hardware,
            iter_parse(program),
            required_hbm_words(program, layout),
            None,
        )
    )
    print(
        json.dumps(
            {
                "segment": args.segment,
                "stage": ("ln1", "qkv_attn", "wo", "ln2", "w1", "w2")[args.segment % 6],
                "elapsed_seconds": perf_counter() - started,
                **{key: timing[key] for key in (
                    "cycles", "peak_window_power_w", "hbm_read_bytes",
                    "hbm_write_bytes", "cache_hits", "cache_misses",
                    "instruction_count",
                )},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
