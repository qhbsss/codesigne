"""Shared-hardware, per-case MILP with analytical floors and verification cuts."""

import copy
import itertools
import json
import math
import shutil
import time
from contextlib import closing
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .estimate import _compute_key, estimate_program, program_records, validate_wave_capacity
from .hardware import area, validate
from .merge import merge_program
from .profiles import lower_bound_plain, profile
from .tool import (OFFICIAL, ROOT, SCORE_CASES, STARTER, command, compare_reports,
                   fresh, predict, set_field, sha, write)


def _check_spec(spec):
    cases = spec["cases"]
    domains = spec["domains"]
    local_domains = spec["case_domains"]
    case_choices = spec.get("case_choices", {})
    program_choices = spec.get("program_choices", {})
    fusion_factors = spec.get("fusion_factors", {})
    fusion_modes = spec.get("fusion_modes", {})
    software_keys = lambda choice: all(key.startswith("policy.") or
                  key in ("parallel_groups", "async_prefetch") for key in choice)
    if not cases or len({case["name"] for case in cases}) != len(cases):
        raise ValueError("cases must have distinct names")
    if any(case["weight"] <= 0 for case in cases) or not math.isclose(
            sum(case["weight"] for case in cases), 1):
        raise ValueError("positive case weights must sum to one")
    if spec.get("program_base") and (any(spec["case_domains"].values()) or case_choices or
                                     any(case.get("overrides") for case in cases)):
        raise ValueError("program_base permits hardware decisions only, except explicit wave fusion")
    if program_choices and not spec.get("program_base"):
        raise ValueError("program_choices require program_base")
    if set(program_choices) - {case["name"] for case in cases}:
        raise ValueError("program_choices contain unknown cases")
    for name, paths in program_choices.items():
        if (not paths or any(not isinstance(path, str) or not path for path in paths)
                or len(set(paths)) != len(paths)):
            raise ValueError(f"case {name} has invalid or duplicate program choices")
    if set(fusion_factors) - {case["name"] for case in cases} or set(fusion_modes) - set(fusion_factors):
        raise ValueError("fusion decisions contain unknown cases")
    for name, factors in fusion_factors.items():
        if (not factors or any(type(factor) is not int or not 1 <= factor <= 512
                               for factor in factors) or len(set(factors)) != len(factors)):
            raise ValueError(f"case {name} has invalid fusion factors")
    for name, modes in fusion_modes.items():
        if not modes or set(modes) - {"concat", "pack", "pipeline", "elastic", "adaptive"} or len(set(modes)) != len(modes):
            raise ValueError(f"case {name} has invalid fusion modes")
    if set(local_domains) - {case["name"] for case in cases}:
        raise ValueError("case_domains contains unknown cases")
    if set(case_choices) - {case["name"] for case in cases} or set(case_choices) & set(local_domains):
        raise ValueError("case_choices require known cases without case_domains overlap")
    for name, choices in case_choices.items():
        if (not choices or any(not choice or not software_keys(choice) for choice in choices)
                or len({json.dumps(choice, sort_keys=True) for choice in choices}) != len(choices)):
            raise ValueError(f"case {name} has invalid or duplicate software choices")
    if any(not key.startswith("hardware.") for key in domains):
        raise ValueError("shared domains must contain hardware fields only")
    choices = spec.get("hardware_choices")
    if choices is not None:
        if domains or not choices or any(not choice or
                any(not key.startswith("hardware.") for key in choice) for choice in choices):
            raise ValueError("hardware_choices must be nonempty hardware designs without domains")
        if len({json.dumps(choice, sort_keys=True) for choice in choices}) != len(choices):
            raise ValueError("duplicate hardware choices")
    for name, values in local_domains.items():
        if any(not (key.startswith("policy.") or key in ("parallel_groups", "async_prefetch"))
               for key in values):
            raise ValueError(f"case {name} cannot change hardware or workload")
    for values in [*domains.values(), *(value for local in local_domains.values()
                                     for value in local.values())]:
        if not values or len({json.dumps(value) for value in values}) != len(values):
            raise ValueError("empty or duplicate domain values")
    total = (len(choices) if choices is not None else math.prod(map(len, domains.values()))) * math.prod(
        (len(case_choices[case["name"]]) if case["name"] in case_choices else
         math.prod(map(len, local_domains.get(case["name"], {}).values()))) *
        len(program_choices.get(case["name"], [None])) *
        len(fusion_factors.get(case["name"], [1])) *
        len(fusion_modes.get(case["name"], ["concat"])) for case in cases)
    if total > spec.get("max_designs", 256):
        raise ValueError("declared finite space exceeds max_designs")
    if spec.get("target_score") is not None:
        if (not isinstance(spec["target_score"], (int, float)) or
                not math.isfinite(spec["target_score"]) or spec["target_score"] <= 0):
            raise ValueError("target_score must be finite and positive")
        if (set(case["name"] for case in cases) != set(SCORE_CASES) or
                any(not math.isclose(case["weight"], SCORE_CASES[case["name"]][1])
                    for case in cases)):
            raise ValueError("target_score requires all five official cases and weights")
    return total


def lower_bound_dependencies(path, profile_cache=None):
    """Critical-path floor for ordered workgroup commands and explicit waits."""
    profile_cache = {} if profile_cache is None else profile_cache
    with closing(program_records(path)) as records:
        hardware = next(records)["hardware"]
        read_bandwidth = [128, 256, 512][hardware["rf_tier"] - 1]
        write_bandwidth = read_bandwidth // 2
        floor = 0
        for record in records:
            if record["record"] == "commit":
                floor += 1
            elif record["record"] == "wave":
                group_floors = []
                for group in record["groups"]:
                    ready = 0
                    tokens = {}
                    for command in group["commands"]:
                        if command["command"] == "wait":
                            ready = max(ready, *(tokens[token] for token in command["tokens"]))
                            continue
                        instruction = command["instruction"]
                        operation = instruction["op"]
                        if (operation in ("load", "load_shared") and not hardware["cache_mib"]
                                and instruction["global"]["rows"] * instruction["global"]["cols"]):
                            duration = 250
                        elif operation in ("load", "store", "load_shared", "store_shared"):
                            duration = 0
                        else:
                            key = _compute_key(instruction)
                            measured = profile_cache.get(key)
                            if measured is None:
                                measured = profile(instruction, hardware)
                                profile_cache[key] = measured
                            stages = measured["steps"]
                            duration = sum(stage["core_duration"] +
                                math.ceil(stage["rf_read_bytes"] / read_bandwidth) +
                                math.ceil(stage["rf_write_bytes"] / write_bandwidth)
                                for stage in stages)
                        completion = ready + duration
                        if command["command"] == "async":
                            tokens[command["token"]] = completion
                        else:
                            ready = completion
                    group_floors.append(ready)
                floor += max(group_floors, default=0) + 1
            elif record["record"] not in ("input", "alloc", "release"):
                raise ValueError("dependency floor requires expanded exporter output")
        return max(floor, 1)


def _checkpoint(marker, option, config_hash):
    temporary = marker.with_name("prepared.json.tmp")
    write(temporary, dict(option, config_sha256=config_hash))
    temporary.replace(marker)


def prepare_options(spec, output, timeout, resume=False):
    cases = spec["cases"]
    domains = spec["domains"]
    case_domains = spec["case_domains"]
    case_choices = spec.get("case_choices", {})
    program_choices = spec.get("program_choices", {})
    fusion_factors = spec.get("fusion_factors", {})
    fusion_modes = spec.get("fusion_modes", {})
    base_hardware = None
    if "hardware_base_program" in spec:
        with (ROOT / spec["hardware_base_program"]).open() as source:
            base_hardware = json.loads(next(source))["hardware"]
        validate(base_hardware)
    base_programs = {}
    if spec.get("program_base"):
        for case in cases:
            config = json.loads((ROOT / case["config"]).read_text())
            paths = program_choices.get(case["name"],
                [str(ROOT / spec["program_base"] / f"{case['name']}.jsonl")])
            base_programs[case["name"]] = []
            for path in paths:
                program = ROOT / path
                with program.open() as source:
                    header = json.loads(next(source))
                if any(header[field] != config[field] for field in
                       ("shape", "batch", "decode", "hardware")):
                    raise ValueError(f"program_base mismatch for {case['name']}: {program}")
                base_programs[case["name"]].append((program, sha(program)))
    hardware_options = []
    options = []
    rejections = []
    designs = (spec["hardware_choices"] if "hardware_choices" in spec else
               (dict(zip(domains, values)) for values in itertools.product(*domains.values())))
    for index, design in enumerate(designs):
        case_options = []
        hardware = None
        for case in cases:
            name = case["name"]
            local = case_domains.get(name, {})
            alternatives = []
            software_choices = (case_choices[name] if name in case_choices else
                                (dict(zip(local, choice)) for choice in
                                 itertools.product(*local.values())))
            software_designs = (
                dict(software_choice,
                     **({"program_choice": source_index} if name in program_choices else {}),
                     **({"wave_fusion": factor, "wave_fusion_mode": mode}
                        if name in fusion_factors else {}))
                for software_choice in software_choices
                for source_index in range(len(base_programs.get(name, [None])))
                for factor, mode in itertools.product(
                    fusion_factors.get(name, [1]), fusion_modes.get(name, ["concat"])))
            for option_index, local_design in enumerate(software_designs):
                config = copy.deepcopy(json.loads((ROOT / case["config"]).read_text()))
                if base_hardware is not None:
                    config["hardware"] = copy.deepcopy(base_hardware)
                try:
                    for key, value in case.get("overrides", {}).items():
                        set_field(config, key, value)
                    for key, value in design.items():
                        set_field(config, key, value)
                    for key, value in local_design.items():
                        if key not in ("wave_fusion", "wave_fusion_mode", "program_choice"):
                            set_field(config, key, value)
                    validate(config["hardware"])
                    if not 1 <= config.get("parallel_groups", 1) <= config["hardware"]["resident_groups"]:
                        raise ValueError("parallel_groups exceeds residency")
                    if hardware is None:
                        hardware = config["hardware"]
                    if config["hardware"] != hardware:
                        raise ValueError("shared hardware constraint")
                except ValueError as error:
                    rejections.append({"hardware": design, "case": name,
                                       "software": local_design, "reason": str(error)})
                    continue
                directory = output / f"hardware-{index:04d}" / name / f"option-{option_index:04d}"
                source = directory / "config.json"
                program = directory / "program.jsonl"
                marker = directory / "prepared.json"
                if directory.exists():
                    if not resume:
                        raise FileExistsError(directory)
                    if not source.is_file() or json.loads(source.read_text()) != config:
                        raise RuntimeError(f"resumed candidate configuration changed: {directory}")
                    if marker.exists():
                        cached = json.loads(marker.read_text())
                        if (cached["program_sha256"] != sha(program) or
                                cached["hardware_index"] != index or cached["case"] != name or
                                cached["design"] != local_design or
                                cached["config_sha256"] != sha(source) or
                                (base_programs and cached["base_program_sha256"] !=
                                 base_programs[name][local_design.get("program_choice", 0)][1])):
                            raise RuntimeError(f"resumed candidate checkpoint changed: {directory}")
                        cached.pop("config_sha256")
                        if "estimated_cycles" not in cached:
                            cached["estimated_cycles"] = estimate_program(program)["cycles"]
                            _checkpoint(marker, cached, sha(source))
                        options.append(cached)
                        alternatives.append(len(options) - 1)
                        continue
                factor = local_design.get("wave_fusion", 1)
                compact_log = directory / "compact-command.json"
                reusable = (resume and base_programs and program.is_file() and
                            (factor == 1 or (compact_log.is_file() and
                             json.loads(compact_log.read_text()).get("returncode") == 0 and
                             not (directory / "fused.expanded.jsonl").exists())))
                if reusable:
                    plain = program
                    fusion_stats = None
                else:
                    if directory.exists():
                        shutil.rmtree(directory)
                    directory.mkdir(parents=True, exist_ok=False)
                    write(source, config)
                if base_programs and not reusable:
                    original, original_sha = base_programs[name][
                        local_design.get("program_choice", 0)]
                    if sha(original) != original_sha:
                        raise RuntimeError(f"program_base changed: {name}")
                    with original.open("rb") as reader, program.open("xb") as writer:
                        header = json.loads(reader.readline())
                        header["hardware"] = config["hardware"]
                        writer.write(json.dumps(header, separators=(",", ":")).encode() + b"\n")
                        shutil.copyfileobj(reader, writer)
                    if factor > 1:
                        plain = directory / "fused.expanded.jsonl"
                        fusion_stats = merge_program(program, plain, factor,
                                                    local_design.get("wave_fusion_mode", "concat"))
                        program.unlink()
                    else:
                        plain = program
                        fusion_stats = None
                elif not base_programs:
                    exported = directory / "export"
                    command([OFFICIAL, "export", source, spec.get("seed", 7), exported],
                            timeout, directory / "export-command.json")
                    plain = exported / "program.jsonl"
                    fusion_stats = None
                    if factor > 1:
                        fused = directory / "fused.expanded.jsonl"
                        fusion_stats = merge_program(plain, fused, factor,
                            local_design.get("wave_fusion_mode", "concat"))
                        plain = fused
                profile_cache = {}
                try:
                    resource_bound = lower_bound_plain(plain, profile_cache)
                    dependency_bound = lower_bound_dependencies(plain, profile_cache)
                    bound = max(resource_bound, dependency_bound)
                    if plain != program:
                        command(["python3", STARTER / "tools/compact_program.py", plain, program],
                                timeout, directory / "compact-command.json")
                        plain.unlink()
                    with program.open() as reader:
                        header = json.loads(next(reader))
                    if header["hardware"] != config["hardware"]:
                        raise ValueError("program hardware mismatch")
                    estimated_cycles = estimate_program(program)["cycles"]
                except ValueError as error:
                    rejections.append({"hardware": design, "case": name,
                                       "software": local_design, "reason": str(error)})
                    continue
                option = {"hardware_index": index, "case": name,
                          "design": local_design, "lower_bound_cycles": bound,
                          "resource_bound_cycles": resource_bound,
                          "dependency_bound_cycles": dependency_bound,
                          "estimated_cycles": estimated_cycles,
                          "program_sha256": sha(program), "program": str(program),
                          "directory": str(directory), "status": "unverified"}
                if base_programs:
                    option["base_program_sha256"] = base_programs[name][
                        local_design.get("program_choice", 0)][1]
                if fusion_stats is not None:
                    option["fusion_stats"] = fusion_stats
                _checkpoint(marker, option, sha(source))
                options.append(option)
                alternatives.append(len(options) - 1)
            case_options.append(alternatives)
        if all(case_options):
            hardware_options.append({"index": index, "design": design,
                                     "hardware": hardware, "area_au": area(hardware)})
        else:
            rejections.append({"hardware": design, "reason": "no feasible case option"})
    valid_indices = {entry["index"] for entry in hardware_options}
    options = [option for option in options if option["hardware_index"] in valid_indices]
    write(output / "preparation.json", {"hardware": hardware_options, "options": options,
                                        "rejections": rejections,
                                        "program_base": spec.get("program_base"),
                                        "bound_kind": "max of mandatory resource and workgroup dependency floors"})
    if not hardware_options:
        raise ValueError("no shared hardware with a feasible option for every case")
    return hardware_options, options


def solve_master(hardware, options, cases, solver_seconds=30, costs=None):
    """Select one hardware and one compatible program per case via integer programming."""
    count_hardware = len(hardware)
    total = count_hardware + len(options)
    objective = np.zeros(total)
    for index, option in enumerate(options):
        value = (math.log(option["verified_cycles"]) if option.get("verified_cycles")
                 else math.log(option["lower_bound_cycles"]))
        objective[count_hardware + index] = cases[option["case"]] * (
            costs[index] if costs is not None else value)
    lower = np.zeros(total)
    upper = np.ones(total)
    for index, option in enumerate(options):
        if option["status"] == "infeasible":
            upper[count_hardware + index] = 0
    rows = 2 + count_hardware * len(cases)
    matrix = lil_matrix((rows, total))
    minimum = np.zeros(rows)
    maximum = np.zeros(rows)
    matrix[0, :count_hardware] = 1
    minimum[0] = maximum[0] = 1
    matrix[1, :count_hardware] = [entry["area_au"] for entry in hardware]
    minimum[1] = -np.inf
    maximum[1] = 100
    for hardware_index, entry in enumerate(hardware):
        for case_index, name in enumerate(cases):
            row = 2 + hardware_index * len(cases) + case_index
            matrix[row, hardware_index] = -1
            for option_index, option in enumerate(options):
                if option["hardware_index"] == entry["index"] and option["case"] == name:
                    matrix[row, count_hardware + option_index] = 1
    result = milp(objective, integrality=np.ones(total), bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix.tocsr(), minimum, maximum),
                  options={"time_limit": solver_seconds, "mip_rel_gap": 0.0})
    info = {"status": int(result.status), "message": result.message,
            "variables": total, "constraints": rows,
            "mip_gap": float(result.mip_gap) if result.x is not None else None,
            "dual_bound": float(result.mip_dual_bound) if result.x is not None else None}
    if result.status != 0 or result.x is None:
        return None, info
    chosen_hardware = hardware[int(np.argmax(result.x[:count_hardware]))]
    selected = [index for index, value in enumerate(result.x[count_hardware:]) if value > 0.5]
    if len(selected) != len(cases):
        raise RuntimeError("MILP did not select exactly one option per case")
    return (chosen_hardware, selected), info


def solve(spec_path, output, timeout=600, max_verifications=32, resume=False):
    started = time.monotonic()
    output = Path(output)
    spec = json.loads(Path(spec_path).read_text())
    domain_designs = _check_spec(spec)
    if resume:
        if not output.is_dir() or json.loads((output / "spec.json").read_text()) != spec:
            raise ValueError("resume requires an existing run with identical spec")
        if (output / "solution.json").exists():
            raise ValueError("completed model-first run cannot be resumed")
        (output / "failure.json").unlink(missing_ok=True)
    else:
        fresh(output)
        write(output / "spec.json", spec)
    scope = ("declared header-replayed static programs, not regenerated code or unrestricted ISA"
             if spec.get("program_base") else
             "declared hardware/case generator domain, not unrestricted ISA")
    hardware, options = prepare_options(spec, output, timeout, resume=resume)
    cases = {case["name"]: case["weight"] for case in spec["cases"]}
    write(output / "model.json", {"binary_variables": ["x_h", "y_h,c,a"],
        "constraints": ["sum_h x_h=1", "sum_a y_h,c,a=x_h for every h,c",
                        "sum_h area(h)*x_h<=100", "infeasible y_h,c,a=0"],
        "objective": "min sum_h,c,a weight_c * log(lower_bound_cycles_h,c,a) * y_h,c,a",
        "lower_bound_source": "independent expanded-program resource and dependency floors",
        "program_origin": "header replay" if spec.get("program_base") else "official export",
        "options_prepared": len(options), "hardware_prepared": len(hardware)})
    target = spec.get("target_score")
    if target is not None:
        floor_proposal, floor_info = solve_master(
            hardware, options, cases, spec.get("solver_seconds", 30))
        if floor_proposal is None or floor_info["dual_bound"] is None:
            raise ValueError("safe score bound solver did not finish")
        log_reference = sum(cases[name] * math.log(SCORE_CASES[name][0]) for name in cases)
        log_score_ceiling = math.log(1000) + log_reference - floor_info["dual_bound"]
        score_ceiling = math.exp(log_score_ceiling)
        _, floor_selected = floor_proposal
        write(output / "bound-screen.json", {
            "provable_score_upper_bound": score_ceiling,
            "log_cost_lower_bound": floor_info["dual_bound"],
            "selected_option_indices": floor_selected,
            "target_score": target,
            "scope": scope,
            "not_a_predicted_or_verified_score": True})
        if log_score_ceiling < math.log(target) - 1e-9:
            result = {"termination": "below_proven_score_upper_bound_unverified",
                      "scope": scope,
                      "provable_score_upper_bound": score_ceiling,
                      "conditional_predicted_score": None,
                      "target_score": target, "domain_designs": domain_designs,
                      "target_unattainable_within_declared_domain": True,
                      "model_optimum_proven": False, "official_score": None,
                      "verified_options": 0, "host_seconds": time.monotonic() - started}
            write(output / "solution.json", result)
            return result
        for option in options:
            if "estimated_cycles" not in option:
                option["estimated_cycles"] = estimate_program(option["program"])["cycles"]
        predicted, predicted_info = solve_master(
            hardware, options, cases, spec.get("solver_seconds", 30),
            costs=[math.log(option["estimated_cycles"]) for option in options])
        if predicted is None:
            raise ValueError("no model-feasible combination")
        write(output / "estimation.json", {"solver": predicted_info,
            "scope": scope,
            "options": [{"case": option["case"], "hardware_index": option["hardware_index"],
                         "design": option.get("design", {}), "program_sha256": option["program_sha256"],
                         "estimated_cycles": option["estimated_cycles"],
                         "lower_bound_cycles": option["lower_bound_cycles"]}
                        for option in options]})
        _, predicted_indices = predicted
        predicted_cycles = {options[index]["case"]: options[index]["estimated_cycles"]
                            for index in predicted_indices}
        predicted_score = 1000 * math.exp(sum(cases[name] * math.log(
            SCORE_CASES[name][0] / value) for name, value in predicted_cycles.items()))
        write(output / "screen.json", {"conditional_predicted_score": predicted_score,
            "predicted_cycles": predicted_cycles, "selected": predicted_indices,
            "conditional_predicted_optimum_within_declared_domain": True,
            "program_sha256": {options[index]["case"]: options[index]["program_sha256"]
                               for index in predicted_indices},
            "not_a_lower_bound_or_correctness_certificate": True})
        if predicted_score < target:
            result = {"termination": "below_prediction_target_unverified",
                      "scope": scope,
                      "conditional_predicted_score": predicted_score,
                      "conditional_predicted_optimum_within_declared_domain": True,
                      "target_score": target, "domain_designs": domain_designs,
                      "model_optimum_proven": False, "official_score": None,
                      "verified_options": 0, "host_seconds": time.monotonic() - started}
            write(output / "solution.json", result)
            return result
    incumbent = None
    incumbent_cost = math.inf
    iterations = []
    termination = "verification_budget"
    dual = None
    for iteration in range(max_verifications + 1):
        proposal, info = solve_master(hardware, options, cases, spec.get("solver_seconds", 30))
        dual = info["dual_bound"]
        if proposal is None:
            termination = "infeasible" if info["status"] == 2 else "solver_incomplete"
            break
        chosen_hardware, selected = proposal
        if incumbent is not None and dual is not None and dual >= incumbent_cost - 1e-9:
            termination = "optimal_within_declared_domain_and_tolerance"
            break
        if iteration == max_verifications:
            break
        entry = {"iteration": iteration, "hardware": chosen_hardware["design"],
                 "selected": selected, "master": info}
        iterations.append(entry)
        write(output / "iterations.json", iterations)
        if target is not None:
            proposal_score = 1000 * math.exp(sum(
                cases[options[index]["case"]] * math.log(
                    SCORE_CASES[options[index]["case"]][0] / options[index]["estimated_cycles"])
                for index in selected))
            entry["conditional_predicted_score"] = proposal_score
            if proposal_score < target:
                termination = "master_proposal_below_prediction_target_unverified"
                break
        valid = True
        for index in selected:
            option = options[index]
            if option["status"] == "verified":
                continue
            directory = Path(option["directory"])
            if sha(option["program"]) != option["program_sha256"]:
                raise RuntimeError("program changed between model and verification")
            prediction = predict(option["program"], directory / "prediction", timeout)
            result = prediction["result"]
            cycles = result["report"]["stats"]["cycles"]
            if option["lower_bound_cycles"] > cycles:
                raise RuntimeError("analytical lower bound exceeds actual cycles")
            if not result["report"]["power_pass"]:
                option["status"] = "infeasible"
                entry["rejection"] = "power"
                valid = False
                break
            official = directory / "verification"
            try:
                command([OFFICIAL, "evaluate", option["program"], spec.get("seed", 7), official],
                        timeout, directory / "verify-command.json")
            except RuntimeError:
                result_path = official / "result.json"
                if result_path.exists():
                    error = json.loads(result_path.read_text()).get("error") or ""
                    if any(word in error for word in ("mismatch", "nonfinite", "uninitialized",
                                                     "power constraint")):
                        option["status"] = "infeasible"
                        entry["rejection"] = error
                        valid = False
                        break
                raise
            outcome = json.loads((official / "result.json").read_text())
            comparison = compare_reports(prediction, outcome, option["program_sha256"])
            if not comparison["all_timing_report_fields_equal"]:
                raise RuntimeError("timing backend changed during official verification")
            option["status"] = "verified"
            option["verified_cycles"] = cycles
            entry.setdefault("new_verified_options", []).append(index)
        if valid:
            cost = sum(cases[options[index]["case"]] * math.log(options[index]["verified_cycles"])
                       for index in selected)
            entry["verified_log_cost"] = cost
            if cost < incumbent_cost:
                incumbent_cost = cost
                incumbent = chosen_hardware, selected
                entry["new_incumbent"] = True
        write(output / "iterations.json", iterations)
    solution = {"scope": scope,
                "termination": termination, "domain_designs": domain_designs,
                "hardware_prepared": len(hardware), "options_prepared": len(options),
                "verified_options": sum(option["status"] == "verified" for option in options),
                "model_log_lower_bound": dual,
                "incumbent_log_cost": incumbent_cost if incumbent is not None else None,
                "log_gap": max(0, incumbent_cost - dual) if incumbent is not None and dual is not None else None,
                "numerical_scope": f"public seed {spec.get('seed', 7)} only",
                "official_score": None, "iterations": iterations,
                "host_seconds": time.monotonic() - started}
    if incumbent is not None:
        chosen_hardware, selected = incumbent
        fresh(output / "best")
        fresh(output / "best/programs")
        write(output / "best/hardware.json", chosen_hardware["hardware"])
        solution["best_design"] = {"hardware": chosen_hardware["design"],
                                  "cases": {options[index]["case"]: options[index]["design"]
                                            for index in selected}}
        solution["best_cycles"] = {options[index]["case"]: options[index]["verified_cycles"]
                                   for index in selected}
        solution["best_program_sha256"] = {options[index]["case"]: options[index]["program_sha256"]
                                           for index in selected}
        for index in selected:
            option = options[index]
            destination = output / "best/programs" / f"{option['case']}.jsonl"
            shutil.copyfile(option["program"], destination)
            if sha(destination) != option["program_sha256"]:
                raise RuntimeError("selected program changed")
        if target is not None:
            predicted_score = 1000 * math.exp(sum(
                cases[options[index]["case"]] * math.log(
                    SCORE_CASES[options[index]["case"]][0] / options[index]["estimated_cycles"])
                for index in selected))
            if predicted_score >= target:
                grade = output / "grade"
                command(["python3", STARTER / "tools/grade_v09.py",
                         output / "best/programs", grade, "--binary", OFFICIAL,
                         "--seed", spec.get("seed", 7), "--jobs", 2, "--cpu-budget", 4,
                         "--total-memory-gib", 8, "--timeout", timeout],
                        timeout * 3 + 60, output / "grade-command.json")
                official = json.loads((grade / "grade.json").read_text())
                if any(official["cases"][name]["program_sha256"] != digest for name, digest
                       in solution["best_program_sha256"].items()):
                    raise RuntimeError("graded program hashes differ from model solution")
                solution["official_score"] = official["score"]
                solution["grade_status"] = official["status"]
    write(output / "solution.json", solution)
    return solution
