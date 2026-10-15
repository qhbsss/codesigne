"""Fuse provably independent adjacent static waves without changing the ISA."""

import argparse
import heapq
import json
from collections import defaultdict
from pathlib import Path

from .estimate import InfeasibleProgramError, program_records, validate_wave_capacity


DMA_OPS = {"load", "store", "load_shared", "store_shared"}
STORE_OPS = {"store", "store_shared"}
MAX_COMMANDS = 65536
MAX_ACCESS_INTERVALS = 262144


def _footprint(record):
    groups = record["groups"]
    if not groups:
        return None
    group_keys = tuple((group["sm"], group["rf_kib"], group["sh_kib"])
                       for group in groups)
    operations = set()
    readers = set()
    writers = set()
    writes = defaultdict(list)
    loads = []
    access_intervals = 0
    for group in groups:
        for command in group["commands"]:
            instruction = command.get("instruction")
            if instruction is None:
                continue
            operation = instruction["op"]
            operations.add(operation)
            if operation not in DMA_OPS:
                continue
            tensor = instruction["tensor"]
            access_intervals += instruction["global"]["rows"]
            if operation not in STORE_OPS:
                readers.add(tensor)
                loads.append((tensor, instruction["global"]))
                continue
            writers.add(tensor)
            view = instruction["global"]
            if view["cols"] != 1 and view["col_stride"] != 1:
                return None
            for row in range(view["rows"]):
                start = view["base"] + row * view["row_stride"]
                writes[tensor].append((start, start + view["cols"] - 1))
    if not writers:
        return None
    kind = ("gemm" if operations & {"mma", "mma_shared"} else
            "dma" if operations <= DMA_OPS else "other")
    reads = defaultdict(list)
    for tensor, view in loads:
        if tensor not in writers:
            continue
        for row in range(view["rows"]):
            start = view["base"] + row * view["row_stride"]
            end = start + (view["cols"] - 1) * view["col_stride"]
            reads[tensor].append((min(start, end), max(start, end)))
    for addresses in (*writes.values(), *reads.values()):
        addresses.sort()
    return {"key": (kind, group_keys, frozenset(readers), frozenset(writers),
                    frozenset(operations)),
            "intervals": {"writes": writes, "reads": reads},
            "access_intervals": access_intervals,
            "commands": sum(len(group["commands"]) for group in groups)}


def _pipeline_footprint(record):
    groups = record["groups"]
    if not groups:
        return None
    accesses = []
    access_intervals = 0
    for group in groups:
        reads = defaultdict(list)
        writes = defaultdict(list)
        for command in group["commands"]:
            instruction = command.get("instruction", {})
            operation = instruction.get("op")
            if operation not in DMA_OPS:
                continue
            view = instruction["global"]
            access_intervals += view["rows"]
            if operation in STORE_OPS and view["col_stride"] != 1 and view["cols"] != 1:
                return None
            intervals = writes if operation in STORE_OPS else reads
            for row in range(view["rows"]):
                start = view["base"] + row * view["row_stride"]
                end = start + (view["cols"] - 1) * view["col_stride"]
                intervals[instruction["tensor"]].append((min(start, end), max(start, end)))
        for addresses in (*reads.values(), *writes.values()):
            addresses.sort()
        accesses.append({"reads": reads, "writes": writes})
    return {"key": ("pipeline", tuple((group["sm"], group["rf_kib"], group["sh_kib"])
                                     for group in groups)),
            "intervals": accesses,
            "access_intervals": access_intervals,
            "commands": sum(len(group["commands"]) for group in groups)}


def _elastic_footprint(record):
    commands = [command for group in record["groups"] for command in group["commands"]]
    if (not commands or any(command["command"] != "run" or
            command["instruction"]["op"] not in ("fill", "load", "store") for command in commands)
            or not any(command["instruction"]["op"] == "store" for command in commands)):
        return None
    footprint = _pipeline_footprint(record)
    if footprint is not None:
        footprint["key"] = ("elastic", tuple((group["sm"], group["sh_kib"])
                                               for group in record["groups"]))
    return footprint


def _adaptive_footprint(record):
    if (not record["groups"] or any(group["sh_kib"] for group in record["groups"])):
        return None
    if any(command.get("instruction", {}).get("op") not in
           ("fill", "load", "store", "vector", "reduce", "mma")
           for group in record["groups"] for command in group["commands"]
           if "instruction" in command):
        return None
    footprint = _pipeline_footprint(record)
    if footprint is not None:
        footprint["key"] = ("adaptive", tuple(group["sm"] for group in record["groups"]))
    return footprint


def _overlaps(left, right):
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        if left_start <= right_end and right_start <= left_end:
            return True
        if left_end < right_start:
            left_index += 1
        else:
            right_index += 1
    return False


def _disjoint_and_combine(previous, current):
    for tensor in previous["writes"]:
        if (_overlaps(previous["writes"][tensor], current["writes"][tensor]) or
                _overlaps(previous["writes"][tensor], current["reads"].get(tensor, [])) or
                _overlaps(previous["reads"].get(tensor, []), current["writes"][tensor])):
            return None
    return {kind: {tensor: list(heapq.merge(addresses, current[kind].get(tensor, [])))
                   for tensor, addresses in previous[kind].items()}
            for kind in ("writes", "reads")}


def _pipeline_and_combine(previous, current):
    for old_group, old in enumerate(previous):
        for new_group, new in enumerate(current):
            if old_group == new_group:
                continue
            for tensor, writes in old["writes"].items():
                if (_overlaps(writes, new["reads"].get(tensor, [])) or
                        _overlaps(writes, new["writes"].get(tensor, []))):
                    return None
            for tensor, writes in new["writes"].items():
                if _overlaps(old["reads"].get(tensor, []), writes):
                    return None
    return [
        {kind: {tensor: list(heapq.merge(addresses, new[kind].get(tensor, [])))
                for tensor, addresses in (
                    (tensor, old[kind].get(tensor, []))
                    for tensor in old[kind].keys() | new[kind].keys())}
         for kind in ("reads", "writes")}
        for old, new in zip(previous, current)
    ]


def _append_commands(destination, source):
    for old_group, new_group in zip(destination["groups"], source["groups"]):
        offset = max((command["token"] for command in old_group["commands"]
                      if command["command"] == "async"), default=-1) + 1
        for command in new_group["commands"]:
            if command["command"] == "async":
                command["token"] += offset
            elif command["command"] == "wait":
                command["tokens"] = [token + offset for token in command["tokens"]]
            old_group["commands"].append(command)


def merge_program(source, destination, factor, mode="concat"):
    """Write expanded JSONL with at most factor independent waves per barrier."""
    if not 1 <= factor <= 512:
        raise ValueError("wave merge factor must be 1..512")
    if mode not in ("concat", "pack", "pipeline", "elastic", "adaptive"):
        raise ValueError("wave merge mode must be concat, pack, pipeline, elastic or adaptive")
    stats = {"factor": factor, "source_waves": 0, "output_waves": 0,
             "mode": mode, "fused_gemm_waves": 0, "fused_dma_waves": 0,
             "fused_other_waves": 0, "fused_pipeline_waves": 0,
             "fused_elastic_waves": 0, "fused_adaptive_waves": 0}
    records = program_records(source)
    pending = None
    hardware = None
    footprint = None
    combined_intervals = None
    count = 0

    def emit(writer):
        nonlocal pending, footprint, combined_intervals, count
        if pending is not None:
            writer.write(json.dumps(pending, separators=(",", ":")) + "\n")
            stats["output_waves"] += 1
        pending = footprint = combined_intervals = None
        count = 0

    with Path(destination).open("x") as output:
        for record in records:
            if record.get("record") != "wave":
                emit(output)
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                if "hardware" in record:
                    hardware = record["hardware"]
                continue
            stats["source_waves"] += 1
            candidate = ((_pipeline_footprint(record) if mode == "pipeline" else
                          _elastic_footprint(record) if mode == "elastic" else
                          _adaptive_footprint(record) if mode == "adaptive" else
                          _footprint(record)) if factor > 1 else None)
            merged = None
            if (pending is not None and footprint is not None and candidate is not None
                    and count < factor and footprint["key"] == candidate["key"]
                    and footprint["commands"] + candidate["commands"] <= MAX_COMMANDS
                    and footprint["access_intervals"] + candidate["access_intervals"]
                    <= MAX_ACCESS_INTERVALS):
                merged = (_pipeline_and_combine(combined_intervals, candidate["intervals"])
                          if mode in ("pipeline", "elastic", "adaptive") else
                          _disjoint_and_combine(combined_intervals, candidate["intervals"]))
                if merged is not None and mode == "pack":
                    try:
                        validate_wave_capacity(pending["groups"] + record["groups"], hardware)
                    except InfeasibleProgramError:
                        merged = None
                if merged is not None and mode in ("elastic", "adaptive"):
                    expanded = [dict(group, rf_kib=max(group["rf_kib"], added["rf_kib"]))
                                for group, added in zip(pending["groups"], record["groups"])]
                    try:
                        validate_wave_capacity(expanded, hardware)
                    except InfeasibleProgramError:
                        merged = None
            if merged is not None:
                if mode != "pack":
                    _append_commands(pending, record)
                    if mode in ("elastic", "adaptive"):
                        for group, enlarged in zip(pending["groups"], expanded):
                            group["rf_kib"] = enlarged["rf_kib"]
                else:
                    pending["groups"].extend(record["groups"])
                combined_intervals = merged
                footprint["commands"] += candidate["commands"]
                footprint["access_intervals"] += candidate["access_intervals"]
                count += 1
                stats[f"fused_{candidate['key'][0]}_waves"] += 1
            else:
                emit(output)
                pending = record
                footprint = candidate
                combined_intervals = candidate["intervals"] if candidate else None
                count = 1
        emit(output)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--mode", choices=("concat", "pack", "pipeline", "elastic", "adaptive"), default="concat")
    args = parser.parse_args()
    print(json.dumps(merge_program(args.source, args.destination, args.factor, args.mode), indent=2))


if __name__ == "__main__":
    main()
