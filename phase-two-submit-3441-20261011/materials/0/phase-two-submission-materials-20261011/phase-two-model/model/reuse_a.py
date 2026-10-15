"""Pair independent GEMM output columns while keeping one activation A tile in RF."""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from .estimate import program_records, validate_wave_capacity
from .reuse_b import _tile


def _signature(tile):
    return (tile["signature"][:2], tuple(
        (activation["tensor"], tuple(sorted(activation["global"].items())),
         mma["m"], mma["n"], mma["k"], mma["a"])
        for activation, _, mma in tile["chunks"]))


def _paired(first, second, first_tile, second_tile):
    reservation = first["rf_kib"] * 256
    output_size = first_tile["fill"]["len"]
    second_panel = reservation + output_size
    commands = [
        {"command": "run", "instruction": copy.deepcopy(first_tile["fill"])},
        {"command": "run", "instruction": dict(second_tile["fill"], dst=reservation)},
    ]
    token = 0
    for (activation, panel, mma), (_, other_panel, other_mma) in zip(
            first_tile["chunks"], second_tile["chunks"]):
        for load in (activation, panel):
            token += 1
            commands.append({"command": "async", "token": token,
                             "instruction": copy.deepcopy(load)})
        commands.append({"command": "wait", "tokens": [token - 1, token]})
        token += 1
        instruction = copy.deepcopy(other_panel)
        instruction["local"]["base"] = second_panel
        commands.append({"command": "async", "token": token,
                         "instruction": instruction})
        commands.append({"command": "run", "instruction": copy.deepcopy(mma)})
        commands.append({"command": "wait", "tokens": [token]})
        commands.append({"command": "run", "instruction": dict(other_mma,
                         a=mma["a"], b=second_panel, c=reservation)})
    commands.append({"command": "run", "instruction": copy.deepcopy(first_tile["store"])})
    other_store = copy.deepcopy(second_tile["store"])
    other_store["local"]["base"] = reservation
    commands.append({"command": "run", "instruction": other_store})
    panel_words = max(panel["global"]["rows"] * panel["global"]["cols"]
                      for _, panel, _ in second_tile["chunks"])
    return {"sm": first["sm"], "rf_kib": (second_panel + panel_words + 255) // 256,
            "sh_kib": first["sh_kib"], "commands": commands}


def _transform(waves, hardware, stats):
    groups = [group for wave in waves for group in wave["groups"]]
    tiles = [_tile(group) for group in groups]
    if any(tile is None for tile in tiles) or len({_signature(tile)[0] for tile in tiles}) != 1:
        return waves
    intervals = sorted(address for tile in tiles for address in tile["footprint"])
    if any(first[1] >= second[0] for first, second in zip(intervals, intervals[1:])):
        return waves
    waiting = defaultdict(list)
    combined = []
    paired = removed = 0
    for group, tile in zip(groups, tiles):
        signature = _signature(tile)
        matches = waiting[signature]
        if matches:
            index, previous, old_tile = matches.pop(0)
            if (previous["rf_kib"] == group["rf_kib"] and
                    previous["sh_kib"] == group["sh_kib"] and
                    not previous["sh_kib"]):
                joined = _paired(previous, group, old_tile, tile)
                if joined["rf_kib"] <= hardware["rf_kib"]:
                    combined[index] = joined
                    paired += 1
                    removed += len(tile["chunks"])
                    continue
            matches.insert(0, (index, previous, old_tile))
        waiting[signature].append((len(combined), group, tile))
        combined.append(group)
    if not paired:
        return waves
    output = []
    capacity = min(hardware["sms"] * hardware["resident_groups"], 16)
    wave_groups = []
    for group in combined:
        placed = False
        for sm in range(hardware["sms"]):
            on_sm = [current for current in wave_groups if current["sm"] == sm]
            if (len(on_sm) < hardware["resident_groups"] and
                    sum(current["rf_kib"] for current in on_sm) + group["rf_kib"] <=
                    hardware["rf_kib"]):
                selected = copy.deepcopy(group)
                selected["sm"] = sm
                wave_groups.append(selected)
                placed = True
                break
        if not placed or len(wave_groups) == capacity:
            if not placed:
                validate_wave_capacity(wave_groups, hardware)
                output.append({"record": "wave", "groups": wave_groups})
                selected = copy.deepcopy(group)
                selected["sm"] = 0
                wave_groups = [selected]
            else:
                validate_wave_capacity(wave_groups, hardware)
                output.append({"record": "wave", "groups": wave_groups})
                wave_groups = []
    if wave_groups:
        validate_wave_capacity(wave_groups, hardware)
        output.append({"record": "wave", "groups": wave_groups})
    stats["paired_groups"] += paired
    stats["removed_a_loads"] += removed
    return output


def reuse_a_program(source, destination, window=32):
    if not 2 <= window <= 64:
        raise ValueError("reuse window must be 2..64")
    records = program_records(source)
    header = next(records)
    hardware = header["hardware"]
    stats = {"window": window, "source_waves": 0, "output_waves": 0,
             "paired_groups": 0, "removed_a_loads": 0}
    pending = []
    stage = None

    def write(output, record):
        output.write(json.dumps(record, separators=(",", ":")) + "\n")

    def flush(output):
        nonlocal stage
        if pending:
            for wave in _transform(pending, hardware, stats):
                write(output, wave)
                stats["output_waves"] += 1
            pending.clear()
        stage = None

    with Path(destination).open("x") as output:
        write(output, header)
        for record in records:
            if record.get("record") != "wave":
                flush(output)
                write(output, record)
                continue
            stats["source_waves"] += 1
            tiles = [_tile(group) for group in record["groups"]]
            signatures = {_signature(tile)[0] for tile in tiles if tile is not None}
            key = next(iter(signatures)) if (tiles and all(tile is not None for tile in tiles)
                                               and len(signatures) == 1) else None
            if key is None or (stage is not None and key != stage):
                flush(output)
            if key is None:
                write(output, record)
                stats["output_waves"] += 1
                continue
            stage = key
            pending.append(record)
            if len(pending) >= window:
                flush(output)
        flush(output)
    if not stats["paired_groups"]:
        raise ValueError("no matching activation panels")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--window", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(reuse_a_program(args.source, args.destination, args.window), indent=2))


if __name__ == "__main__":
    main()
