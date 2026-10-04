"""Run the public exact timing model for one case without timing the other case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from codesign.challenge.hardware import Hardware
from codesign.challenge.runner import estimate_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("M1_P1", "M2_D1"), required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    hardware = Hardware.from_dict(json.loads(args.hardware.read_text(encoding="utf-8")))
    program = args.program.read_text(encoding="utf-8")
    started = perf_counter()
    timing = estimate_case(args.case, hardware, program)
    report = {
        "kind": "single-case-exact-timing-v1",
        "case": args.case,
        "elapsed_seconds": perf_counter() - started,
        "timing": timing,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case": args.case,
                "elapsed_seconds": report["elapsed_seconds"],
                "cycles": timing["cycles"],
                "peak_window_power_w": timing["peak_window_power_w"],
                "area_mm2": timing["area_mm2"],
                "instruction_count": timing["instruction_count"],
                "hbm_read_bytes": timing["hbm_read_bytes"],
                "hbm_write_bytes": timing["hbm_write_bytes"],
                "cache_hits": timing["cache_hits"],
                "cache_misses": timing["cache_misses"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
