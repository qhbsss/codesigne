"""Run race and numerical checks without invoking the timing estimator."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from codesign.challenge.hardware import Hardware
from codesign.challenge.hbm_race import validate_hbm_races
from codesign.challenge.micro import MicroMachine
from codesign.challenge.runner import check_case


CASES = ("M1_P1", "M2_D1")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", type=Path, default=Path("hardware.json"))
    parser.add_argument("--program-p1", type=Path, default=Path("programs/M1_P1.asm"))
    parser.add_argument("--program-d1", type=Path, default=Path("programs/M2_D1.asm"))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--case", choices=CASES, action="append", dest="selected_cases")
    parser.add_argument("--race-only", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    original_write = MicroMachine._write

    def diagnostic_write(machine, desc, values):
        try:
            return original_write(machine, desc, values)
        except ValueError as exc:
            array = np.asarray(values).ravel()
            finite = np.isfinite(array)
            caller = inspect.currentframe().f_back
            operation_args = caller.f_locals.get("args", {}) if caller is not None else {}
            raise ValueError(
                f"{exc}; destination={desc}; value_count={array.size}; "
                f"finite_count={int(finite.sum())}; event={operation_args.get('event')}; "
                f"kind={operation_args.get('kind')}"
            ) from exc

    MicroMachine._write = diagnostic_write
    hardware = Hardware.from_dict(json.loads(args.hardware.read_text(encoding="utf-8")))
    programs = {
        "M1_P1": args.program_p1.read_text(encoding="utf-8"),
        "M2_D1": args.program_d1.read_text(encoding="utf-8"),
    }
    seeds = args.seed or [7]
    report = {"kind": "functional-only-v1", "race_only": args.race_only, "cases": {}}
    passed = True
    for case in args.selected_cases or CASES:
        started = perf_counter()
        item = {"hbm_races_validated": False, "checks": []}
        try:
            validate_hbm_races(programs[case])
            item["hbm_races_validated"] = True
            if not args.race_only:
                for seed in seeds:
                    item["checks"].append(
                        check_case(
                            case,
                            seed,
                            hardware,
                            programs[case],
                            hbm_races_validated=True,
                        )
                    )
        except (ValueError, FloatingPointError, IndexError, KeyError, TypeError) as exc:
            item["error"] = str(exc)
        item["elapsed_seconds"] = perf_counter() - started
        item["passed"] = bool(
            item["hbm_races_validated"]
            and (args.race_only or (item["checks"] and all(check["passed"] for check in item["checks"])))
        )
        report["cases"][case] = item
        passed &= item["passed"]
    report["passed"] = passed
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.report:
        if args.report.exists():
            raise FileExistsError(f"Report already exists: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
