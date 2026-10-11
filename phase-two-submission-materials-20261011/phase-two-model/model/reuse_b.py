"""Co-schedule independent GEMM tiles sharing a B panel in private RF."""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from .estimate import program_records, validate_wave_capacity


def _tile(group):
    commands = group["commands"]
    if len(commands) < 6 or (len(commands) - 2) % 4:
        return None
    fill, store = commands[0], commands[-1]
    if (fill["command"] != "run" or fill["instruction"]["op"] != "fill" or
            store["command"] != "run" or store["instruction"]["op"] != "store"):
        return None
    dimensions = None
    chunks = []
    for offset in range(1, len(commands) - 1, 4):
        first, second, wait, run = commands[offset:offset + 4]
        if (first["command"] != "async" or second["command"] != "async" or
                wait != {"command": "wait", "tokens": [first["token"], second["token"]]} or
                run["command"] != "run"):
            return None
        activation, panel, mma = (command["instruction"] for command in (first, second, run))
        if (activation["op"], panel["op"], mma["op"]) != ("load", "load", "mma"):
            return None
        size = mma["m"] * mma["n"]
        if (mma["c"] != 0 or mma["a"] != size or
                activation["local"] != {"base": size, "rows": mma["m"],
                                         "cols": mma["k"], "row_stride": mma["k"],
                                         "col_stride": 1} or
                panel["local"] != {"base": mma["b"], "rows": mma["k"],
                                    "cols": mma["n"], "row_stride": mma["n"],
                                    "col_stride": 1}):
            return None
        if dimensions is None:
            dimensions = mma["m"], mma["n"], activation["tensor"], panel["tensor"]
        if dimensions != (mma["m"], mma["n"], activation["tensor"], panel["tensor"]):
            return None
        chunks.append((activation, panel, mma))
    output = store["instruction"]
    output_view = output["global"]
    if (output["tensor"] in dimensions[2:] or fill["instruction"]["dst"] != 0 or
            fill["instruction"]["len"] != dimensions[0] * dimensions[1] or
            output["local"] != {"base": 0, "rows": dimensions[0],
                                "cols": dimensions[1], "row_stride": dimensions[1],
                                "col_stride": 1} or
            (output_view["cols"] != 1 and output_view["col_stride"] != 1)):
        return None
    footprint = []
    for row in range(output_view["rows"]):
        start = output_view["base"] + row * output_view["row_stride"]
        end = start + (output_view["cols"] - 1) * output_view["col_stride"]
        footprint.append((min(start, end), max(start, end)))
    signature = (dimensions, output["tensor"], tuple(
        (panel["tensor"], tuple(sorted(panel["global"].items())), mma["k"], mma["b"])
        for _, panel, mma in chunks))
    return {"chunks": chunks, "store": output, "fill": fill["instruction"],
            "signature": signature, "footprint": footprint}


def _paired(first, second, first_tile, second_tile, reservation):
    commands = []
    for tile, address in ((first_tile, 0), (second_tile, reservation)):
        fill = dict(tile["fill"], dst=address)
        commands.append({"command": "run", "instruction": fill})
    token = 0
    for (activation, panel, mma), (other_activation, _, other_mma) in zip(
            first_tile["chunks"], second_tile["chunks"]):
        token += 1
        commands.append({"command": "async", "instruction": copy.deepcopy(panel),
                         "token": token})
        panel_token = token
        for tile_index, (load, operation, address) in enumerate(
                ((activation, mma, 0), (other_activation, other_mma, reservation))):
            token += 1
            commands.append({"command": "async", "instruction": copy.deepcopy(load),
                             "token": token})
            commands.append({"command": "wait", "tokens":
                             [panel_token, token] if tile_index == 0 else [token]})
            commands.append({"command": "run", "instruction": dict(operation, c=address)})
    for tile, address in ((first_tile, 0), (second_tile, reservation)):
        store = copy.deepcopy(tile["store"])
        store["local"]["base"] = address
        commands.append({"command": "run", "instruction": store})
    size = first_tile["fill"]["len"]
    return {"sm": first["sm"], "rf_kib": (reservation + size + 255) // 256,
            "sh_kib": first["sh_kib"], "commands": commands}


def _transform(waves, hardware, stats):
    groups = [group for wave in waves for group in wave["groups"]]
    tiles = [_tile(group) for group in groups]
    if any(tile is None for tile in tiles):
        return waves
    signatures = {tile["signature"][:2] for tile in tiles}
    if len(signatures) != 1:
        return waves
    intervals = sorted(address for tile in tiles for address in tile["footprint"])
    if any(left[1] >= right[0] for left, right in zip(intervals, intervals[1:])):
        return waves
    waiting = defaultdict(list)
    combined = []
    paired = 0
    removed = 0
    for group, tile in zip(groups, tiles):
        reservation = group["rf_kib"] * 256
        needed = (reservation + tile["fill"]["len"] + 255) // 256
        if (needed * 2 <= hardware["rf_kib"] and group["sh_kib"] * 2 <= hardware["sh_kib"] and
                hardware["resident_groups"] >= 2):
            matches = waiting[tile["signature"]]
            if matches:
                index, prior, prior_tile = matches.pop(0)
                if prior["rf_kib"] == group["rf_kib"] and prior["sh_kib"] == group["sh_kib"]:
                    combined[index] = _paired(prior, group, prior_tile, tile, reservation)
                    paired += 1
                    removed += len(tile["chunks"])
                    continue
                matches.insert(0, (index, prior, prior_tile))
        waiting[tile["signature"]].append((len(combined), group, tile))
        combined.append(group)
    if not paired:
        return waves
    capacity = min(hardware["sms"] * hardware["resident_groups"], 16)
    output = []
    for offset in range(0, len(combined), capacity):
        wave_groups = copy.deepcopy(combined[offset:offset + capacity])
        for index, group in enumerate(wave_groups):
            group["sm"] = index % hardware["sms"]
        validate_wave_capacity(wave_groups, hardware)
        output.append({"record": "wave", "groups": wave_groups})
    stats["paired_groups"] += paired
    stats["removed_b_loads"] += removed
    return output


def _wave_key(record):
    if record.get("record") != "wave" or not record["groups"]:
        return None
    tiles = [_tile(group) for group in record["groups"]]
    if any(tile is None for tile in tiles):
        return None
    signatures = {tile["signature"][:2] for tile in tiles}
    return next(iter(signatures)) if len(signatures) == 1 else None


def reuse_b_program(source, destination, window=32):
    if not 2 <= window <= 64:
        raise ValueError("reuse window must be 2..64")
    records = program_records(source)
    header = next(records)
    hardware = header["hardware"]
    stats = {"window": window, "source_waves": 0, "output_waves": 0,
             "paired_groups": 0, "removed_b_loads": 0}
    pending = []
    stage = None

    def write(writer, record):
        writer.write(json.dumps(record, separators=(",", ":")) + "\n")

    def flush(writer):
        nonlocal stage
        if pending:
            for wave in _transform(pending, hardware, stats):
                write(writer, wave)
                stats["output_waves"] += 1
            pending.clear()
        stage = None

    with Path(destination).open("x") as writer:
        write(writer, header)
        for record in records:
            if record.get("record") != "wave":
                flush(writer)
                write(writer, record)
                continue
            stats["source_waves"] += 1
            key = _wave_key(record)
            if key is None or (stage is not None and stage != key):
                flush(writer)
            if key is None:
                write(writer, record)
                stats["output_waves"] += 1
                continue
            stage = key
            pending.append(record)
            if len(pending) >= window:
                flush(writer)
        flush(writer)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--window", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(reuse_b_program(args.source, args.destination, args.window), indent=2))


if __name__ == "__main__":
    main()
