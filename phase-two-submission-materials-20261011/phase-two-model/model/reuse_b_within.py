"""Share a GEMM B panel across adjacent independent tiles of one workgroup."""

import argparse
import copy
import json
from pathlib import Path

from .estimate import program_records, validate_wave_capacity
from .reuse_b import _paired, _tile


def _tiles(group):
    commands = group["commands"]
    if not commands:
        return None
    tiles = []
    start = 0
    for index, command in enumerate(commands):
        if command.get("instruction", {}).get("op") != "store":
            continue
        segment = commands[start:index + 1]
        tile = _tile(dict(group, commands=segment))
        if tile is None:
            return None
        tiles.append((segment, tile))
        start = index + 1
    return tiles if start == len(commands) else None


def _independent(first, second):
    if first["signature"] != second["signature"]:
        return False
    return all(left_end < right_start or right_end < left_start
               for left_start, left_end in first["footprint"]
               for right_start, right_end in second["footprint"])


def _renumber(commands):
    seen = {}
    next_token = 0
    for command in commands:
        if command["command"] == "async":
            next_token += 1
            old_token = command["token"]
            seen[old_token] = next_token
            command["token"] = next_token
        elif command["command"] == "wait":
            command["tokens"] = [seen[token] for token in command["tokens"]]
    return commands


def _group(group):
    tiles = _tiles(group)
    if tiles is None or len(tiles) < 2:
        return group, 0, 0
    commands = []
    maximum_rf = group["rf_kib"]
    pairs = removed_loads = 0
    index = 0
    while index < len(tiles):
        original, first = tiles[index]
        if index + 1 < len(tiles):
            _, second = tiles[index + 1]
            reservation = group["rf_kib"] * 256
            required_rf = (reservation + first["fill"]["len"] + 255) // 256
            if _independent(first, second):
                combined = _paired(group, group, first, second, reservation)
                commands.extend(combined["commands"])
                maximum_rf = max(maximum_rf, required_rf)
                pairs += 1
                removed_loads += len(first["chunks"])
                index += 2
                continue
        commands.extend(original)
        index += 1
    if not pairs:
        return group, 0, 0
    return dict(group, rf_kib=maximum_rf,
                commands=_renumber(copy.deepcopy(commands))), pairs, removed_loads


def reuse_b_within_program(source, destination):
    records = program_records(source)
    header = next(records)
    stats = {"waves": 0, "paired_tiles": 0, "removed_b_loads": 0,
             "rejected_by_rf_capacity": 0}
    with Path(destination).open("x") as writer:
        writer.write(json.dumps(header, separators=(",", ":")) + "\n")
        for record in records:
            if record.get("record") == "wave":
                stats["waves"] += 1
                selected = list(record["groups"])
                for index, group in enumerate(record["groups"]):
                    changed, pairs, loads = _group(group)
                    if not pairs:
                        continue
                    selected[index] = changed
                    try:
                        validate_wave_capacity(selected, header["hardware"])
                    except ValueError:
                        selected[index] = group
                        stats["rejected_by_rf_capacity"] += pairs
                    else:
                        stats["paired_tiles"] += pairs
                        stats["removed_b_loads"] += loads
                record = dict(record, groups=selected)
            writer.write(json.dumps(record, separators=(",", ":")) + "\n")
    if not stats["paired_tiles"]:
        raise ValueError("no capacity-feasible adjacent GEMM tiles share B panels")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(reuse_b_within_program(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
