"""Frozen two-case challenge score arithmetic, separate from simulation."""

from dataclasses import dataclass
from math import exp, isfinite, log

from .hardware import cost_model
from .workload import SCENARIOS, WORKLOAD

CASES = tuple(WORKLOAD["active_cases"])


@dataclass(frozen=True)
class ScenarioMetric:
    cycles: int
    peak_power_w: float
    functional_passed: bool


def _finite_nonnegative(value, name: str) -> None:
    if type(value) not in (int, float):
        raise ValueError(f"Invalid {name}")
    try:
        valid = isfinite(value)
    except OverflowError:
        valid = False
    if not valid or value < 0:
        raise ValueError(f"Invalid {name}")


def _validate_metric(metric: ScenarioMetric, baseline: int) -> None:
    if (
        not isinstance(metric, ScenarioMetric)
        or type(metric.cycles) is not int
        or not 0 < metric.cycles <= 2**63 - 1
        or type(baseline) is not int
        or not 0 < baseline <= 2**63 - 1
        or type(metric.functional_passed) is not bool
    ):
        raise ValueError("Invalid scenario metric or baseline")
    _finite_nonnegative(metric.peak_power_w, "peak power")


def evaluate_score(
    area_mm2: float,
    measured: dict[str, ScenarioMetric],
    baseline_cycles: dict[str, int],
) -> dict:
    """Score P1 and D1 equally if every correctness and budget gate passes."""
    _finite_nonnegative(area_mm2, "area")
    if set(measured) != set(CASES) or set(baseline_cycles) != set(CASES):
        raise ValueError("Exactly the active case metrics and baselines required")
    cost = cost_model()
    clock = cost["clock_hz"]
    area_passed = area_mm2 <= cost["area_limit_mm2"]
    diagnostics = {}
    eligible = area_passed
    log_ratio_sum = 0.0
    for case in CASES:
        metric, baseline = measured[case], baseline_cycles[case]
        _validate_metric(metric, baseline)
        scenario = case.split("_")[1]
        batch, context, generated = SCENARIOS[scenario]
        tokens = batch * (context if scenario.startswith("P") else generated)
        latency_divisor = generated if scenario.startswith("D") else 1
        latency_passed = metric.cycles <= 2 * baseline
        power_passed = metric.peak_power_w <= cost["power_limit_w"]
        case_passed = metric.functional_passed and latency_passed and power_passed
        diagnostics[case] = {
            "cycles": metric.cycles,
            "throughput_tokens_per_second": tokens * clock / metric.cycles,
            "first_token_or_average_step_seconds": metric.cycles / (clock * latency_divisor),
            "peak_power_w": metric.peak_power_w,
            "functional_passed": metric.functional_passed,
            "latency_gate_passed": latency_passed,
            "power_gate_passed": power_passed,
            "throughput_ratio_to_baseline": baseline / metric.cycles,
        }
        eligible &= case_passed
        log_ratio_sum += log(baseline) - log(metric.cycles)
    return {
        "eligible": bool(eligible),
        "score": 1000 * exp(log_ratio_sum / len(CASES)) if eligible else None,
        "area_mm2": area_mm2,
        "area_gate_passed": area_passed,
        "scenarios": diagnostics,
    }


def evaluate_single_score(
    area_mm2: float,
    metric: ScenarioMetric,
    baseline_cycles: int,
) -> dict:
    """Historical M1/P1 trial score retained for old experiment reports."""
    _finite_nonnegative(area_mm2, "area")
    _validate_metric(metric, baseline_cycles)
    cost = cost_model()
    eligible = (
        metric.functional_passed
        and area_mm2 <= cost["area_limit_mm2"]
        and metric.peak_power_w <= cost["power_limit_w"]
        and metric.cycles <= 2 * baseline_cycles
    )
    return {
        "case": "M1_P1",
        "eligible": eligible,
        "score": 1000 * baseline_cycles / metric.cycles if eligible else None,
        "cycles": metric.cycles,
        "baseline_cycles": baseline_cycles,
        "first_token_seconds": metric.cycles / cost["clock_hz"],
        "throughput_tokens_per_second": SCENARIOS["P1"][1] * cost["clock_hz"] / metric.cycles,
        "area_mm2": area_mm2,
        "peak_power_w": metric.peak_power_w,
        "functional_passed": metric.functional_passed,
    }
