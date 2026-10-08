"""Fast, calibrated screening model for challenge candidates.

This module deliberately does not call ``estimate_pipeline``.  It expands the
literal ISA once, totals service demand by SM/resource, and calibrates the
result against a previously generated, unedited grade report.  The central
estimate is useful for ranking candidates; the conservative estimate adds an
explicit uncertainty margin and is the only value used by the go/no-go gate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from hashlib import sha256
from math import ceil, sqrt
from pathlib import Path

from codesign.challenge.hardware import Hardware, cost_model
from codesign.challenge.isa import Instruction, iter_parse
from codesign.challenge.service import (
    mma_service,
    reduction_cycles,
    sfu_cycles,
    vector_cycles,
)


CASES = ("M1_P1", "M2_D1")


def _digest_text(text: str) -> str:
    return sha256(text.encode()).hexdigest()


def _load_hardware(path: Path) -> Hardware:
    return Hardware.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _ports(hardware: Hardware) -> tuple[int, int]:
    return {"2R1W": (2, 1), "4R2W": (4, 2), "8R4W": (8, 4)}[hardware.rf_ports]


def _rf_cycles(hardware: Hardware, read_bytes: int, write_bytes: int) -> tuple[int, int]:
    reads, writes = _ports(hardware)
    return ceil(read_bytes / (16 * reads)), ceil(write_bytes / (16 * writes))


@lru_cache(maxsize=None)
def _cached_line_ids(
    offset: int, count: int, shape: tuple[int, ...], strides: tuple[int, ...]
) -> tuple[int, ...]:
    """Return touched 64-byte line numbers without enumerating every word."""
    if len(shape) != len(strides) or len(shape) not in (1, 2):
        raise ValueError("Invalid HBM view rank")

    result: set[int] = set()

    def add_run(start: int, words: int) -> None:
        result.update(range(start // 16, (start + words - 1) // 16 + 1))

    if len(shape) == 1:
        if strides == [1]:
            add_run(offset, shape[0])
        else:
            for index in range(shape[0]):
                result.add((offset + index * strides[0]) // 16)
        return tuple(sorted(result))

    rows, cols = shape
    row_stride, col_stride = strides
    if col_stride == 1:
        for row in range(rows):
            add_run(offset + row * row_stride, cols)
    elif row_stride == 1:
        for col in range(cols):
            add_run(offset + col * col_stride, rows)
    else:
        for row in range(rows):
            for col in range(cols):
                result.add((offset + row * row_stride + col * col_stride) // 16)
    return tuple(sorted(result))


def _line_ids(view: dict) -> tuple[int, ...]:
    count = view["count"]
    return _cached_line_ids(
        view["offset"],
        count,
        tuple(view.get("shape", [count])),
        tuple(view.get("strides", [1])),
    )


def _wg_for(ins: Instruction) -> str | None:
    args = ins.args
    if ins.op in {"WG.BEGIN", "WG.END"}:
        return args["wg"]
    if ins.op in {"WAIT"}:
        return args.get("wg")
    if ins.op in {"BARRIER", "STEP.COMMIT"}:
        return None
    if ins.op in {"LD", "ST"}:
        src, dst = args["src"], args["dst"]
        return src.get("wg") if src.get("space") != "HBM" else dst.get("wg")
    if ins.op == "MMA.ACC":
        return args["acc"]["wg"]
    return args["dst"]["wg"]


@dataclass
class StaticProfile:
    instruction_count: int
    opcode_counts: dict[str, int]
    raw_cycles: int
    raw_bottleneck: str
    sm_demands: dict[str, dict[str, int]]
    shared_demands: dict[str, int]
    hbm_read_line_requests: int
    hbm_write_line_requests: int
    dynamic_energy_pj: float


def _with_capacity_bottleneck(
    profile: StaticProfile, hardware: Hardware, hbm_request_scale: float = 1.0
) -> StaticProfile:
    """Normalize aggregate demand by the capacity of replicated/byte resources."""
    candidates: list[tuple[int, str]] = []
    sm_line_capacity = max(1, hardware.sm_noc_bytes_per_cycle // 64)
    for sm, demand in profile.sm_demands.items():
        for resource, cycles in demand.items():
            if resource == "dma":
                capacity = hardware.dma_engines
            elif resource in {"noc_in", "noc_out"}:
                capacity = sm_line_capacity
            else:
                capacity = 1
            candidates.append((ceil(cycles / capacity), f"sm{sm}:{resource}"))
    noc_capacity = max(1, hardware.noc_bytes_per_cycle // 64)
    noc_slices = profile.shared_demands.get(
        "noc_multicast_line_slices", profile.shared_demands["noc_line_slices"]
    )
    candidates.append(
        (
            ceil(noc_slices / noc_capacity),
            "shared:noc_bandwidth",
        )
    )
    for resource, cycles in profile.shared_demands.items():
        if resource.startswith("hbm_channel_"):
            candidates.append((ceil(cycles * hbm_request_scale), f"shared:{resource}"))
    raw_cycles, bottleneck = max(candidates)
    return replace(profile, raw_cycles=raw_cycles, raw_bottleneck=bottleneck)


def profile_program(hardware: Hardware, program: str) -> StaticProfile:
    hardware.validate()
    prices = cost_model()["energy_pj"]
    rf_price = prices["rf_byte"][hardware.rf_ports]
    noc_price = prices["noc_byte_fixed"] + prices["noc_byte_per_width"] * hardware.noc_bytes_per_cycle
    groups: dict[str, int] = {}
    sm_demands: dict[int, Counter] = defaultdict(Counter)
    shared = Counter()
    opcodes = Counter()
    dynamic_energy = 0.0
    read_lines = 0
    write_lines = 0
    read_ordinals: Counter = Counter()
    multicast_reads: set[tuple[int, int]] = set()

    for ins in iter_parse(program):
        op, args = ins.op, ins.args
        opcodes[op] += 1
        if op == "WG.BEGIN":
            groups[args["wg"]] = args["sm"]
            sm_demands[args["sm"]]["issue"] += 1
            continue
        if op == "WG.END":
            sm_demands[groups[args["wg"]]]["issue"] += 1
            continue
        if op in {"WAIT", "BARRIER", "STEP.COMMIT"}:
            shared["synchronization_points"] += 1
            continue

        wg = _wg_for(ins)
        if wg is None or wg not in groups:
            raise ValueError(f"Cannot assign {op} on line {ins.line} to a workgroup")
        sm = groups[wg]
        demand = sm_demands[sm]
        demand["issue"] += 1

        if op == "MMA.ACC":
            service = mma_service(hardware, args["m"], args["n"], args["k"])
            read_bytes, write_bytes = service.rf_read_bytes, service.rf_write_bytes
            demand["tc"] += service.compute_cycles
            dynamic_energy += service.arithmetic_pj
        elif op == "VEC":
            count = args["dst"]["count"]
            read_bytes = 4 * sum(item["count"] for item in args["src"] if "space" in item)
            write_bytes = 4 * count
            demand["vec"] += vector_cycles(count, hardware.vector_lanes)
            dynamic_energy += count * prices["vector_fma" if args["kind"] == "fma" else "vector_other"]
        elif op == "REDUCE":
            count = args["src"]["count"]
            read_bytes, write_bytes = 4 * count, 4
            resource = "reduce" if hardware.reduction_units else "vec"
            demand[resource] += reduction_cycles(count, hardware.vector_lanes, hardware.reduction_units)
            dynamic_energy += max(0, count - 1) * prices[
                "reduction_dedicated_merge" if hardware.reduction_units else "reduction_vector_merge"
            ]
        elif op == "SFU":
            count = args["src"]["count"]
            read_bytes = write_bytes = 4 * count
            demand["sfu"] += sfu_cycles(count, hardware.sfu_lanes)
            dynamic_energy += count * prices["sfu"]
        elif op in {"LD", "ST"}:
            src, dst = args["src"], args["dst"]
            payload = 4 * src["count"]
            read_bytes = payload if src["space"] == "RF" else 0
            write_bytes = payload if dst["space"] == "RF" else 0
            if "HBM" in {src["space"], dst["space"]}:
                hbm_view = src if src["space"] == "HBM" else dst
                lines = _line_ids(hbm_view)
                line_count = len(lines)
                demand["dma"] += line_count
                direction = "noc_in" if src["space"] == "HBM" else "noc_out"
                demand[direction] += line_count
                shared["noc_line_slices"] += line_count
                for line in lines:
                    channel = (line * 64 // 256) % hardware.hbm_channels
                    shared[f"hbm_channel_{channel}_gross"] += 2
                if src["space"] == "HBM":
                    read_lines += line_count
                    ordinal = read_ordinals[wg]
                    read_ordinals[wg] += 1
                    multicast_reads.update((ordinal, line) for line in lines)
                else:
                    write_lines += line_count
                dynamic_energy += line_count * (64 * noc_price + prices["dma_slice"])
                if dst["space"] == "SH":
                    rate = 16
                    demand["shared_write"] += ceil(
                        payload / max(1, rate * hardware.shared_banks)
                    )
                    dynamic_energy += payload * prices["shared_byte"][hardware.shared_ports]
            elif "SH" in {src["space"], dst["space"]}:
                # Shared traffic is uncommon in the current candidates.  This
                # is a conservative aggregate bank demand for screening.
                rate = 16 if dst["space"] == "SH" or hardware.shared_ports == "1R1W" else 32
                resource = "shared_write" if dst["space"] == "SH" else "shared_read"
                demand[resource] += ceil(payload / max(1, rate * hardware.shared_banks))
                dynamic_energy += payload * prices["shared_byte"][hardware.shared_ports]
        else:
            raise ValueError(f"Unsupported screening opcode: {op}")

        rf_read_cycles, rf_write_cycles = _rf_cycles(hardware, read_bytes, write_bytes)
        demand["rf_read"] += rf_read_cycles
        demand["rf_write"] += rf_write_cycles
        dynamic_energy += (read_bytes + write_bytes) * rf_price

    shared["noc_multicast_line_slices"] = (
        len(multicast_reads) + write_lines
        if hardware.multicast
        else shared["noc_line_slices"]
    )
    # Gross HBM energy intentionally assumes every read request misses.  A
    # calibrated cache miss ratio is applied later for the power proxy.
    profile = StaticProfile(
        instruction_count=sum(opcodes.values()),
        opcode_counts=dict(sorted(opcodes.items())),
        raw_cycles=1,
        raw_bottleneck="uninitialized",
        sm_demands={str(sm): dict(sorted(values.items())) for sm, values in sorted(sm_demands.items())},
        shared_demands=dict(sorted(shared.items())),
        hbm_read_line_requests=read_lines,
        hbm_write_line_requests=write_lines,
        dynamic_energy_pj=dynamic_energy,
    )
    return _with_capacity_bottleneck(profile, hardware)


def _score(cycles: dict[str, int], baseline_cycles: dict[str, int]) -> float:
    return 1000.0 * sqrt(
        (baseline_cycles["M1_P1"] / cycles["M1_P1"])
        * (baseline_cycles["M2_D1"] / cycles["M2_D1"])
    )


def _calibration_case(report: dict, case: str) -> dict:
    timing = report["cases"][case]["timing"]
    return {
        "cycles": timing["cycles"],
        "peak_window_power_w": timing["peak_window_power_w"],
        "dynamic_energy_pj": timing["dynamic_energy_pj"],
        "hbm_read_bytes": timing["hbm_read_bytes"],
        "cache_hits": timing.get("cache_hits", 0),
        "cache_misses": timing.get("cache_misses", 0),
    }


def _profile_from_dict(value: dict) -> StaticProfile:
    return StaticProfile(**value)


def build_report(args: argparse.Namespace) -> dict:
    candidate_hardware = _load_hardware(args.hardware)
    calibration_hardware = _load_hardware(args.calibration_hardware)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    baseline_cycles = baseline["baseline_cycles"]
    safety = 1.0 + args.uncertainty

    candidate_programs = {
        "M1_P1": args.program_p1.read_text(encoding="utf-8"),
        "M2_D1": args.program_d1.read_text(encoding="utf-8"),
    }
    calibration_programs = {
        "M1_P1": args.calibration_program_p1.read_text(encoding="utf-8"),
        "M2_D1": args.calibration_program_d1.read_text(encoding="utf-8"),
    }

    provenance_ok = True
    expected = calibration.get("provenance", {}).get("program_sha256", {})
    for case in CASES:
        provenance_ok &= expected.get(case) == _digest_text(calibration_programs[case])

    cached_reference = None
    if args.calibration_static_report:
        cached = json.loads(args.calibration_static_report.read_text(encoding="utf-8"))
        cached_provenance = cached.get("static_provenance", {})
        if (
            cached_provenance.get("hardware") == calibration_hardware.to_dict()
            and cached_provenance.get("program_sha256")
            == {case: _digest_text(calibration_programs[case]) for case in CASES}
        ):
            cached_reference = {
                case: _profile_from_dict(cached["cases"][case]["profile"]) for case in CASES
            }
        else:
            raise ValueError("Calibration static report does not match calibration inputs")

    reference_profiles = cached_reference or {
        case: profile_program(calibration_hardware, calibration_programs[case]) for case in CASES
    }
    reference_profiles = {
        case: _with_capacity_bottleneck(profile, calibration_hardware)
        for case, profile in reference_profiles.items()
    }
    cached_candidate = None
    if args.candidate_static_report:
        cached = json.loads(args.candidate_static_report.read_text(encoding="utf-8"))
        cached_provenance = cached.get("static_provenance", {})
        if (
            cached_provenance.get("hardware") == candidate_hardware.to_dict()
            and cached_provenance.get("program_sha256")
            == {case: _digest_text(candidate_programs[case]) for case in CASES}
        ):
            cached_candidate = {
                case: _profile_from_dict(cached["cases"][case]["profile"]) for case in CASES
            }
        else:
            raise ValueError("Candidate static report does not match candidate inputs")
    candidate_profiles = cached_candidate or {}
    for case in CASES:
        if case in candidate_profiles:
            continue
        if candidate_hardware == calibration_hardware and candidate_programs[case] == calibration_programs[case]:
            candidate_profiles[case] = reference_profiles[case]
        else:
            candidate_profiles[case] = profile_program(candidate_hardware, candidate_programs[case])
    candidate_profiles = {
        case: _with_capacity_bottleneck(profile, candidate_hardware)
        for case, profile in candidate_profiles.items()
    }

    cases = {}
    central_cycles: dict[str, int] = {}
    conservative_cycles: dict[str, int] = {}
    peak_estimates = {}
    for case in CASES:
        exact = _calibration_case(calibration, case)
        accesses = exact["cache_hits"] + exact["cache_misses"]
        reference_miss_rate = exact["cache_misses"] / accesses if accesses else 1.0
        if candidate_hardware.cache_mib:
            cache_ratio = calibration_hardware.cache_mib / candidate_hardware.cache_mib
            miss_rate = min(1.0, reference_miss_rate * sqrt(cache_ratio))
        else:
            miss_rate = 1.0
        # HBM writes are a small fraction of these inference programs.  Use a
        # weighted request scale so cache-hit reads do not become a false HBM
        # bottleneck while writes remain fully charged.
        reference_total_lines = (
            reference_profiles[case].hbm_read_line_requests
            + reference_profiles[case].hbm_write_line_requests
        )
        candidate_total_lines = (
            candidate_profiles[case].hbm_read_line_requests
            + candidate_profiles[case].hbm_write_line_requests
        )
        reference_hbm_scale = (
            (
                reference_profiles[case].hbm_read_line_requests * reference_miss_rate
                + reference_profiles[case].hbm_write_line_requests
            )
            / reference_total_lines
            if reference_total_lines
            else 1.0
        )
        candidate_hbm_scale = (
            (
                candidate_profiles[case].hbm_read_line_requests * miss_rate
                + candidate_profiles[case].hbm_write_line_requests
            )
            / candidate_total_lines
            if candidate_total_lines
            else 1.0
        )
        reference = _with_capacity_bottleneck(
            reference_profiles[case], calibration_hardware, reference_hbm_scale
        )
        candidate = _with_capacity_bottleneck(
            candidate_profiles[case], candidate_hardware, candidate_hbm_scale
        )
        factor = exact["cycles"] / reference.raw_cycles
        central = max(1, ceil(candidate.raw_cycles * factor))
        conservative = max(1, ceil(central * safety))
        central_cycles[case] = central
        conservative_cycles[case] = conservative

        read_misses = ceil(candidate.hbm_read_line_requests * miss_rate)
        prices = cost_model()["energy_pj"]
        estimated_dynamic = (
            candidate.dynamic_energy_pj
            + read_misses * 64 * prices["hbm_byte"]
            + candidate.hbm_write_line_requests * 64 * prices["hbm_byte"]
        )
        reference_density = exact["dynamic_energy_pj"] / exact["cycles"]
        candidate_density = estimated_dynamic / central
        density_ratio = candidate_density / reference_density if reference_density else 1.0
        ref_static = calibration_hardware.static_power_w()
        candidate_static = candidate_hardware.static_power_w()
        peak = candidate_static + max(0.0, exact["peak_window_power_w"] - ref_static) * sqrt(
            max(0.0, density_ratio)
        )
        peak_estimates[case] = peak
        cases[case] = {
            "central_cycles": central,
            "conservative_cycles": conservative,
            "calibration_factor": factor,
            "estimated_peak_power_w": peak,
            "estimated_hbm_read_misses": read_misses,
            "profile": asdict(candidate),
            "calibration_profile": {
                "raw_cycles": reference.raw_cycles,
                "raw_bottleneck": reference.raw_bottleneck,
                "exact_cycles": exact["cycles"],
            },
        }

    area = candidate_hardware.area_mm2()
    central_score = _score(central_cycles, baseline_cycles)
    conservative_score = _score(conservative_cycles, baseline_cycles)
    gate_checks = {
        "calibration_provenance_matches": provenance_ok,
        "conservative_score_at_least_target": conservative_score >= args.target_score,
        "area_at_most_gate": area <= args.area_gate,
        "estimated_peak_power_at_most_gate": max(peak_estimates.values()) <= args.power_gate,
    }
    return {
        "kind": "calibrated-static-screen-v1",
        "warning": "Screening model only; official estimate/grade remains authoritative.",
        "settings": {
            "target_score": args.target_score,
            "uncertainty": args.uncertainty,
            "area_gate_mm2": args.area_gate,
            "power_gate_w": args.power_gate,
        },
        "area_mm2": area,
        "static_provenance": {
            "hardware": candidate_hardware.to_dict(),
            "program_sha256": {case: _digest_text(candidate_programs[case]) for case in CASES},
        },
        "central_score": central_score,
        "conservative_score": conservative_score,
        "gate_checks": gate_checks,
        "passes_all_gates": all(gate_checks.values()),
        "cases": cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", type=Path, default=Path("hardware.json"))
    parser.add_argument("--program-p1", type=Path, default=Path("programs/M1_P1.asm"))
    parser.add_argument("--program-d1", type=Path, default=Path("programs/M2_D1.asm"))
    parser.add_argument("--baseline", type=Path, default=Path("baseline_manifest.json"))
    parser.add_argument("--calibration-report", type=Path, default=Path("local-grade.json"))
    parser.add_argument("--calibration-hardware", type=Path, default=Path("hardware.json"))
    parser.add_argument("--calibration-program-p1", type=Path, default=Path("programs/M1_P1.asm"))
    parser.add_argument("--calibration-program-d1", type=Path, default=Path("programs/M2_D1.asm"))
    parser.add_argument(
        "--calibration-static-report",
        type=Path,
        help="Prior rough report whose static profiles match the calibration inputs",
    )
    parser.add_argument(
        "--candidate-static-report",
        type=Path,
        help="Prior rough report whose static profiles match the candidate inputs",
    )
    parser.add_argument("--uncertainty", type=float, default=0.25)
    parser.add_argument("--target-score", type=float, default=95_000.0)
    parser.add_argument("--area-gate", type=float, default=23.8)
    parser.add_argument("--power-gate", type=float, default=18.0)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 0 <= args.uncertainty <= 1:
        raise ValueError("uncertainty must be between zero and one")
    report = build_report(args)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.report:
        if args.report.exists():
            raise FileExistsError(f"Report already exists: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passes_all_gates"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
