"""Independent, approximate full-program cycle model; never an optimality bound.

The streaming compact reader and latency/throughput model do not call the
official timing backend. Estimates may be inaccurate when DMA queues, cache
hits, multicast or competing workgroups change the critical path.
"""

import json
import math
from collections import defaultdict
from functools import lru_cache
from heapq import heappop, heappush
from pathlib import Path

from .hardware import rf_latency, sh_latency, validate
from .profiles import profile


class InfeasibleProgramError(ValueError):
    pass


def validate_wave_capacity(groups, hardware):
    resources = {}
    for group in groups:
        sm = group["sm"]
        rf_kib, sh_kib = group["rf_kib"], group["sh_kib"]
        if (not 0 <= sm < hardware["sms"] or not 1 <= rf_kib <= hardware["rf_kib"]
                or not 0 <= sh_kib <= hardware["sh_kib"] or sh_kib % 4):
            raise InfeasibleProgramError("invalid workgroup reservation")
        rf_total, sh_total, slots = resources.get(sm, (0, 0, 0))
        rf_total += rf_kib
        sh_total += sh_kib
        slots += 1
        if (rf_total > hardware["rf_kib"] or sh_total > hardware["sh_kib"]
                or slots > hardware["resident_groups"]):
            raise InfeasibleProgramError("resident workgroups exceed SM RF/SH/slot capacity")
        resources[sm] = rf_total, sh_total, slots


def _expression(value, variables):
    if isinstance(value, int):
        return value
    operation, argument = next(iter(value.items()))
    if operation == "var":
        return variables[argument]
    left, right = (_expression(item, variables) for item in argument)
    if operation == "add":
        return left + right
    if operation == "mul":
        return left * right
    if operation == "sub":
        return left - right
    if operation == "div":
        return (abs(left) // abs(right)) * (-1 if (left < 0) != (right < 0) else 1)
    if operation == "mod":
        return left - _expression({"div": [left, right]}, variables) * right
    if operation == "min":
        return min(left, right)
    if operation == "max":
        return max(left, right)
    raise ValueError(f"unknown integer operation {operation}")


def _resolve(value, variables):
    if isinstance(value, dict):
        if len(value) == 1 and next(iter(value)) in {"var", "add", "sub", "mul", "div", "mod", "min", "max"}:
            return _expression(value, variables)
        return {key: _resolve(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, variables) for item in value]
    return value


def _expand(items, variables, templates, tag):
    for item in items:
        kind = item.get(tag)
        if kind == "repeat":
            start = _expression(item["start"], variables)
            count = _expression(item["count"], variables)
            step = _expression(item["step"], variables)
            for index in range(count):
                scope = dict(variables, **{item["var"]: start + index * step})
                yield from _expand(item["body"], scope, templates, tag)
        elif kind == "call":
            template = templates[item["name"]]
            if template["kind"] != tag:
                raise ValueError("template scope mismatch")
            scope = dict(zip(template["params"],
                (_expression(argument, variables) for argument in item["args"])))
            yield from _expand(template["body"], scope, templates, tag)
        elif tag == "record" and kind == "wave":
            yield {"record": "wave", "groups": [
                {"sm": _expression(group["sm"], variables),
                 "rf_kib": _expression(group["rf_kib"], variables),
                 "sh_kib": _expression(group["sh_kib"], variables),
                 "commands": list(_expand(group["commands"], variables, templates, "command"))}
                for group in item["groups"]]}
        else:
            yield _resolve(item, variables)


def program_records(path):
    """Iterate expanded records without materializing an expanded program file."""
    templates = {}
    with Path(path).open() as source:
        header = json.loads(next(source))
        yield header
        for line in source:
            record = json.loads(line)
            if record["record"] == "template":
                templates[record["name"]] = record
            else:
                yield from _expand([record], {}, templates, "record")


def _compute_key(instruction):
    operation = instruction["op"]
    if operation in ("fill", "reduce"):
        return operation, instruction["len"]
    if operation == "mma":
        return operation, instruction["m"], instruction["n"], instruction["k"]
    if operation == "vector":
        operands = tuple((argument["kind"], argument["stride"])
                         if argument["kind"] != "imm" else ("imm",)
                         for argument in (instruction["a"], instruction["b"]))
        return operation, instruction["kind"], instruction["len"], operands
    return json.dumps(instruction, sort_keys=True, separators=(",", ":"))


def _compute_cycles(instruction, hardware, cache):
    key = _compute_key(instruction)
    if key in cache:
        return cache[key]
    current = 1
    read_bw = [128, 256, 512][hardware["rf_tier"] - 1]
    write_bw = read_bw // 2
    for step in profile(instruction, hardware)["steps"]:
        delay = 0
        if step["rf_read_bytes"]:
            delay = math.ceil(step["rf_read_bytes"] / read_bw) + rf_latency(hardware)
        if step["sh_read_bytes"]:
            delay = max(delay, step["sh_read_cycles"] + sh_latency(hardware))
        if step["sh_tc_bytes"]:
            delay = max(delay, step["sh_read_cycles"] + sh_latency(hardware)
                + math.ceil(step["sh_tc_bytes"] / hardware["sh_tc_bw"]) + 2)
        current += delay + step["core_duration"]
        delay = 0
        if step["rf_write_bytes"]:
            delay = math.ceil(step["rf_write_bytes"] / write_bw) + rf_latency(hardware)
        if step["sh_write_bytes"]:
            delay = max(delay, step["sh_write_cycles"] + sh_latency(hardware))
        current += delay
    if len(cache) > 100000:
        cache.clear()
    cache[key] = current
    return current


@lru_cache(maxsize=200000)
def _dma_lines(base, rows, cols, row_stride, col_stride, channels):
    """Count distinct 64-byte lines in each 16-word address-generation batch."""
    counts = [0] * channels
    batch = set()
    previous = set()
    blocked_lines = 0
    for row in range(rows):
        row_base = base + row * row_stride
        for col in range(cols):
            line = (row_base + col * col_stride) // 16
            batch.add(line)
            if (row * cols + col + 1) % 16 == 0:
                blocked_lines += len(batch & previous)
                for address in batch:
                    counts[(address // 4) % channels] += 1
                previous, batch = batch, set()
    blocked_lines += len(batch & previous)
    for address in batch:
        counts[(address // 4) % channels] += 1
    return tuple(counts), blocked_lines


def estimate_wave(groups, hardware, cache=None, details=False, bulk_dma_approximation=True):
    """Reserve group commands on engines and bulk HBM channels in issue order."""
    cache = {} if cache is None else cache
    channels = hardware["hbm_channels"]
    channel_free = [0] * channels
    channel_lines = [0] * channels
    sm_issue = [0] * hardware["sms"]
    compute_work = [0] * hardware["sms"]
    capacity = {"Vector": 1, "Sfu": 1, "Reduction": max(hardware["reduction_units"], 1),
                "Tc": hardware["tc_count"], "Dma": hardware["dma_engines"]}
    engines = [{name: [0] * count for name, count in capacity.items()}
               for _ in range(hardware["sms"])]
    writers = defaultdict(set)
    tokens = [{} for _ in groups]
    events = [(0, index, 0) for index in range(len(groups))]
    finish = 0
    had_dma = False
    dma_count = 0
    serialized_writes = False
    while events:
        ready, group_index, pc = heappop(events)
        group = groups[group_index]
        if pc == len(group["commands"]):
            finish = max(finish, ready)
            continue
        command = group["commands"][pc]
        sm = group["sm"]
        if command["command"] == "wait":
            ready = max(ready, *(tokens[group_index][token] for token in command["tokens"]))
            heappush(events, (max(ready, sm_issue[sm]) + 1, group_index, pc + 1))
            continue
        instruction = command["instruction"]
        operation = instruction["op"]
        dma = operation in ("load", "store", "load_shared", "store_shared")
        if dma:
            had_dma = True
            engine = "Dma"
        elif operation in ("mma", "mma_shared"):
            engine = "Tc"
        elif operation == "reduce":
            engine = "Reduction" if hardware["reduction_units"] else "Vector"
        elif operation == "vector" and instruction["kind"] in ("exp", "tanh", "rsqrt"):
            engine = "Sfu"
        elif operation in ("shared_read", "shared_write"):
            engine = "Dma"
        else:
            engine = "Vector"
        slot = min(range(len(engines[sm][engine])), key=lambda index: engines[sm][engine][index])
        start = max(ready, sm_issue[sm], engines[sm][engine][slot])
        sm_issue[sm] = start + 1
        if not dma:
            completion = start + _compute_cycles(instruction, hardware, cache)
            compute_work[sm] += completion - start
        else:
            dma_count += 1
            view = instruction["global"]
            lines, blocked_lines = _dma_lines(view["base"] % (64 * channels), view["rows"], view["cols"],
                view["row_stride"], view["col_stride"], channels)
            for index, count in enumerate(lines):
                channel_lines[index] += count
            packets = sum(lines)
            write = operation in ("store", "store_shared")
            if write and len(groups) > 1:
                for row in range(view["rows"]):
                    base = view["base"] + row * view["row_stride"]
                    for col in range(view["cols"]):
                        writers[(instruction["tensor"],
                            (base + col * view["col_stride"]) // 16)].add(group_index)
            elements = view["rows"] * view["cols"]
            bytes_moved = packets * (80 if write else 72)
            local_bw = [128, 256, 512][hardware["rf_tier"] - 1] // (1 if write else 2)
            latency = sh_latency(hardware) if "shared" in operation else rf_latency(hardware)
            first_ready = start + 1 + 6 + 250 + (latency + 1 if write else 0)
            completed = first_ready
            for index, count in enumerate(lines):
                if count:
                    channel_free[index] = max(channel_free[index], first_ready) + 2 * count
                    completed = max(completed, channel_free[index])
            throughput = max(math.ceil(elements / 16), math.ceil(packets / 2),
                math.ceil(bytes_moved / hardware["sm_noc"]),
                math.ceil(bytes_moved / hardware["global_noc"]),
                math.ceil(elements * 4 / local_bw))
            completion = max(completed + 6 + latency + 1, first_ready + throughput)
            if write:
                serialized_writes |= blocked_lines > 0
                completion += blocked_lines * 270
        engines[sm][engine][slot] = completion
        if dma and command["command"] == "async":
            tokens[group_index][command["token"]] = completion
            ready = start + 1
        else:
            ready = completion
        heappush(events, (ready, group_index, pc + 1))
    overlaps = set()
    depth = 1
    for accesses in writers.values():
        if len(accesses) > 1:
            depth = max(depth, len(accesses))
            indices = sorted(accesses)
            overlaps.update(zip(indices, indices[1:]))
    collision = max(260 * (depth - 1), math.ceil(260 * len(overlaps) / 4))
    if collision:
        simultaneous = max(sum(group["sm"] == sm for group in groups)
            for sm in range(hardware["sms"]))
        collision *= min(hardware["dma_engines"], simultaneous)
    if bulk_dma_approximation and dma_count >= 128 and not serialized_writes:
        fluid = 2 * max(channel_lines) + math.ceil(dma_count / hardware["sms"]) * 310
        finish = min(finish, max(fluid, max(compute_work, default=0)))
    total = finish + collision + (21 if had_dma else 1)
    if details:
        return {"cycles":total,"reservation":finish,"collision":collision,
                "channels":channel_free,"overlap_edges":len(overlaps)}
    return total


def estimate_program(path, bulk_dma_approximation=True):
    records = program_records(path)
    header = next(records)
    hardware = header["hardware"]
    validate(hardware)
    cache = {}
    total = 0
    waves = 0
    for record in records:
        if record["record"] == "wave":
            validate_wave_capacity(record["groups"], hardware)
            total += estimate_wave(record["groups"], hardware, cache,
                                   bulk_dma_approximation=bulk_dma_approximation)
            waves += 1
        elif record["record"] == "commit":
            total += 1
        elif record["record"] not in ("alloc", "release", "input"):
            raise ValueError("unsupported record")
    return {"cycles": total, "waves": waves, "independent": True,
        "confidence": "cache/multicast unmodeled" if hardware["cache_mib"] or hardware["multicast"] else "empirical contention correction",
        "model": "command dependency / SM engines / HBM channel / DMA contention"}
