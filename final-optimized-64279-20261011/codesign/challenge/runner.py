"""Fixture, functional check, timing, and immutable challenge report assembly."""

import hashlib
import json
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path

import numpy as np

from .abi import HBM_BYTES, Layout, build_layout
from .hardware import Hardware, cost_model
from .hbm_race import validate_hbm_races
from .isa import iter_parse
from .micro import MicroMachine, SparseHBM
from .pipeline import estimate_pipeline
from .reference import build_history, make_inputs, run_scenario
from .score import CASES, ScenarioMetric, evaluate_score
from .workload import MODELS, SCENARIOS, WORKLOAD

TIMING_MODEL = "pipeline-global-events-v2"
REPORT_VERSION = "challenge-report-v0.7"


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_json(value) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def seed_digest(seed: int) -> str:
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    return digest_bytes(f"challenge-seed-v1:{seed}".encode())


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    package = root / "codesign/challenge"
    # Release pins and CLI/signing wrappers do not change numerical or timing semantics.
    paths = sorted(
        path
        for path in package.glob("*.py")
        if path.name not in {"cli.py", "official.py", "release.py", "report_view.py"}
    )
    paths += sorted(package.glob("cost_*.json"))
    paths += sorted(package.glob("workload_*.json"))
    return {str(path.relative_to(root)): digest_bytes(path.read_bytes()) for path in paths}


def provenance(hardware: Hardware, programs: dict[str, str]) -> dict:
    if set(programs) != set(CASES):
        raise ValueError("Both challenge programs are required")
    sources = source_hashes()
    return {
        "hardware_sha256": digest_json(hardware.to_dict()),
        "program_sha256": {case: digest_bytes(programs[case].encode()) for case in CASES},
        "cost_sha256": digest_json(cost_model()),
        "workload_sha256": digest_json(WORKLOAD),
        "simulator_sha256": digest_json(sources),
        "source_sha256": sources,
    }


def _walk_operands(value):
    if isinstance(value, dict):
        if value.get("space") == "HBM":
            yield value
        else:
            for item in value.values():
                yield from _walk_operands(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_operands(item)


def required_hbm_words(program: str, layout: Layout) -> int:
    """Size only touched HBM, while enforcing the ABI's 2 GiB address limit."""
    words = max(
        (symbol.address + symbol.nbytes) // 4
        for symbol in layout.symbols.values()
        if symbol.name != "scratch"
    )
    for ins in iter_parse(program):
        for view in _walk_operands(ins.args):
            count = view.get("count")
            shape = view.get("shape", [count])
            strides = view.get("strides", [1])
            if (
                type(view.get("offset")) is not int
                or view["offset"] < 0
                or type(count) is not int
                or count <= 0
                or not isinstance(shape, list)
                or not isinstance(strides, list)
                or len(shape) != len(strides)
                or len(shape) not in (1, 2)
                or any(type(x) is not int or x <= 0 for x in shape + strides)
                or prod(shape) != count
            ):
                raise ValueError("Invalid HBM view")
            span = 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))
            words = max(words, view["offset"] + span)
            if words > HBM_BYTES // 4:
                raise ValueError("Program exceeds 2 GiB HBM")
    return words


def _ranges(layout: Layout, names: list[str]) -> list[tuple[int, int]]:
    return [
        (
            layout.symbols[name].address // 4,
            (layout.symbols[name].address + layout.symbols[name].nbytes) // 4,
        )
        for name in names
    ]


def _output_names(model) -> list[str]:
    return ["output/hidden"] + [
        f"layer{layer}/new_{kind}" for layer in range(model.layers) for kind in ("k", "v")
    ]


def step_requirements(case: str, layout: Layout) -> dict[int, list[tuple[int, int]]]:
    model = MODELS[case.split("_")[0]]
    scenario = case.split("_")[1]
    batch, context, generated = SCENARIOS[scenario]
    steps = 1 if scenario.startswith("P") else generated
    required = {}
    hidden = layout.symbols["output/hidden"]
    for step in range(steps):
        if scenario.startswith("P"):
            ranges = [
                (
                    hidden.address // 4 + item * (context + 1) * model.width,
                    hidden.address // 4 + (item * (context + 1) + context) * model.width,
                )
                for item in range(batch)
            ]
            begin, end = 0, context
        else:
            ranges = []
            for item in range(batch):
                start = hidden.address // 4 + (item * generated + step) * model.width
                ranges.append((start, start + model.width))
            begin, end = step, step + 1
        for layer in range(model.layers):
            for kind in ("k", "v"):
                symbol = layout.symbols[f"layer{layer}/new_{kind}"]
                extent = context + 1 if scenario.startswith("P") else generated
                for item in range(batch):
                    for head in range(model.heads):
                        start = (
                            symbol.address // 4
                            + (item * model.heads + head) * extent * model.head_width
                        )
                        ranges.append(
                            (start + begin * model.head_width, start + end * model.head_width)
                        )
        required[step] = ranges
    return required


@dataclass
class Fixture:
    machine: MicroMachine
    layout: Layout
    words: int
    requirements: dict[int, list[tuple[int, int]]]


def make_fixture(case: str, seed: int, hardware: Hardware, program: str) -> Fixture:
    if case not in CASES:
        raise ValueError("Unsupported challenge case")
    seed_digest(seed)
    model = MODELS[case.split("_")[0]]
    scenario = case.split("_")[1]
    batch, context, generated = SCENARIOS[scenario]
    layout = build_layout(model, scenario)
    words = required_hbm_words(program, layout)
    hbm = SparseHBM(words)
    weights, inputs = make_inputs(model, batch, context, generated, seed)

    def load(name, value):
        symbol = layout.symbols[name]
        flat = np.asarray(value, np.float32).ravel()
        if flat.size * 4 != symbol.nbytes:
            raise ValueError(f"Fixture shape mismatch: {name}")
        start = symbol.address // 4
        hbm[start : start + flat.size] = flat

    for layer, values in enumerate(weights):
        for name, value in values.items():
            load(f"layer{layer}/{name}", value)
    if scenario.startswith("P"):
        load("input/prompt", inputs[:, :context])
        load("input/step0", inputs[:, context])
    else:
        history = build_history(model, inputs[:, :context], weights)
        for layer, (key, value) in enumerate(history):
            load(f"layer{layer}/history_k", key)
            load(f"layer{layer}/history_v", value)
        for step in range(generated):
            load(f"input/step{step}", inputs[:, context + step])

    readonly = [
        (symbol.address // 4, (symbol.address + symbol.nbytes) // 4)
        for symbol in layout.symbols.values()
        if symbol.readonly
    ]
    required = _ranges(layout, _output_names(model))
    release = [
        (symbol.address // 4, (symbol.address + symbol.nbytes) // 4, symbol.release_step)
        for symbol in layout.symbols.values()
        if symbol.release_step > 0
    ]
    scratch = layout.symbols["scratch"].address // 4
    requirements = step_requirements(case, layout)
    machine = MicroMachine(
        hardware,
        hbm,
        readonly=readonly,
        unique_writes=required,
        release_rules=release,
        step_requirements=requirements,
        invalid_hbm=required + [(scratch, words)],
    )
    return Fixture(machine, layout, words, requirements)


def check_case(
    case: str,
    seed: int,
    hardware: Hardware,
    program: str,
    *,
    hbm_races_validated: bool = False,
) -> dict:
    if not hbm_races_validated:
        validate_hbm_races(program)
    fixture = make_fixture(case, seed, hardware, program)
    result = fixture.machine.execute(iter_parse(program), hbm_races_validated=True)
    model = MODELS[case.split("_")[0]]
    expected_hidden, expected_kv = run_scenario(model, case.split("_")[1], seed)
    expected = {"output/hidden": expected_hidden}
    for layer, (key, value) in enumerate(expected_kv):
        expected[f"layer{layer}/new_k"] = key
        expected[f"layer{layer}/new_v"] = value
    checks = {}
    for name, oracle in expected.items():
        symbol = fixture.layout.symbols[name]
        start = symbol.address // 4
        actual = result[start : start + oracle.size].reshape(oracle.shape)
        delta = np.abs(actual.astype(np.float64) - oracle.astype(np.float64))
        finite_error = bool(np.isfinite(delta).all())
        all_written = all(
            index in fixture.machine.written_hbm for index in range(start, start + oracle.size)
        )
        checks[name] = {
            "max_absolute_error": float(delta.max()) if finite_error else None,
            "allclose_1e-3_1e-3": finite_error
            and bool(np.allclose(actual, oracle, rtol=1e-3, atol=1e-3)),
            "all_elements_written": all_written,
        }
    return {
        "seed_sha256": seed_digest(seed),
        "passed": all(
            value["allclose_1e-3_1e-3"] and value["all_elements_written"]
            for value in checks.values()
        ),
        "checks": checks,
    }


def estimate_case(
    case: str,
    hardware: Hardware,
    program: str,
    *,
    hbm_races_validated: bool = False,
) -> dict:
    model = MODELS[case.split("_")[0]]
    layout = build_layout(model, case.split("_")[1])
    words = required_hbm_words(program, layout)
    timing = estimate_pipeline(
        hardware,
        iter_parse(program),
        words,
        step_requirements(case, layout),
        hbm_races_validated=hbm_races_validated,
    )
    return asdict(timing)


def evaluate_submission(
    mode: str,
    hardware: Hardware,
    programs: dict[str, str],
    seeds: list[int],
    baseline: dict | None = None,
) -> dict:
    """Run the same timing path for estimate, check and candidate grade."""
    if mode not in {"estimate", "check", "grade"}:
        raise ValueError("Unknown challenge mode")
    if set(programs) != set(CASES):
        raise ValueError("Both challenge programs are required")
    if mode != "estimate" and not seeds:
        raise ValueError("Functional modes require seeds")
    if mode == "grade" and baseline is None:
        raise ValueError("Grade requires a baseline manifest")
    seed_hashes = [seed_digest(seed) for seed in seeds]
    report = {
        "version": REPORT_VERSION,
        "mode": mode,
        "timing_model": TIMING_MODEL,
        "status": "candidate",
        "seed_source": "local_public",
        "provenance": provenance(hardware, programs),
        "seed_sha256": seed_hashes,
        "cases": {},
        "score": None,
    }
    validated_cases = set()
    for case in CASES:
        checks = []
        if mode != "estimate":
            try:
                validate_hbm_races(programs[case])
                validated_cases.add(case)
            except (ValueError, FloatingPointError, IndexError, KeyError, TypeError) as exc:
                checks.extend(
                    {"seed_sha256": seed_digest(seed), "passed": False, "error": str(exc)}
                    for seed in seeds
                )
            for seed in seeds[len(checks) :]:
                try:
                    checks.append(
                        check_case(
                            case,
                            seed,
                            hardware,
                            programs[case],
                            hbm_races_validated=True,
                        )
                    )
                except (ValueError, FloatingPointError, IndexError, KeyError, TypeError) as exc:
                    checks.append(
                        {"seed_sha256": seed_digest(seed), "passed": False, "error": str(exc)}
                    )
        passed = all(check["passed"] for check in checks) if checks else False
        report["cases"][case] = {"functional": checks, "functional_passed": passed}
    for case in CASES:
        if not any("error" in check for check in report["cases"][case]["functional"]):
            try:
                report["cases"][case]["timing"] = estimate_case(
                    case,
                    hardware,
                    programs[case],
                    hbm_races_validated=case in validated_cases,
                )
            except (ValueError, FloatingPointError, IndexError, KeyError, TypeError) as exc:
                report["cases"][case]["timing_error"] = str(exc)
    if mode == "grade":
        if any("timing" not in report["cases"][case] for case in CASES):
            report["eligible"] = False
            report["experimental_score"] = None
            return report
        if baseline["timing_model"] != TIMING_MODEL or set(baseline["baseline_cycles"]) != set(
            CASES
        ):
            raise ValueError("Baseline timing model or cases differ")
        measured = {
            case: ScenarioMetric(
                report["cases"][case]["timing"]["cycles"],
                report["cases"][case]["timing"]["peak_window_power_w"],
                report["cases"][case]["functional_passed"],
            )
            for case in CASES
        }
        score = evaluate_score(hardware.area_mm2(), measured, baseline["baseline_cycles"])
        report["gate_diagnostics"] = score
        report["experimental_score"] = score["score"]
        report["eligible"] = score["eligible"]
    return report


def write_report(path: Path, report: dict) -> None:
    """Exclusive creation keeps historical reports unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
