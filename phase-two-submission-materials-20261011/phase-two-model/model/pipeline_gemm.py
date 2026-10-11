"""Overlap the next GEMM input DMA with the current paid MMA instruction."""

import argparse
import copy
import json
from pathlib import Path

from .estimate import program_records, validate_wave_capacity


def _instruction(command):
    return command.get("instruction", {})


def _pipeline_tile(commands, reserved_words):
    if (len(commands) < 10 or (len(commands) - 2) % 4 or
            commands[0]["command"] != "run" or
            _instruction(commands[0]).get("op") != "fill" or
            commands[-1]["command"] != "run" or
            _instruction(commands[-1]).get("op") != "store"):
        return None
    chunks = []
    output = _instruction(commands[-1])["tensor"]
    for offset in range(1, len(commands) - 1, 4):
        chunk = commands[offset:offset + 4]
        first, second, wait, mma = chunk
        if (first["command"] != "async" or second["command"] != "async" or
                wait != {"command": "wait", "tokens": [first["token"], second["token"]]} or
                mma["command"] != "run" or
                tuple(_instruction(item).get("op") for item in (first, second, mma)) !=
                ("load", "load", "mma")):
            return None
        activation, weight, compute = map(_instruction, (first, second, mma))
        if (output in (activation["tensor"], weight["tensor"]) or
                compute["a"] != activation["local"]["base"] or
                compute["b"] != weight["local"]["base"] or
                compute["c"] != _instruction(commands[0])["dst"] or
                activation["local"]["row_stride"] != activation["local"]["cols"] or
                weight["local"]["row_stride"] != weight["local"]["cols"] or
                activation["local"]["col_stride"] != 1 or
                weight["local"]["col_stride"] != 1):
            return None
        chunks.append(chunk)
    if len(chunks) < 2:
        return None
    max_activation = max(_instruction(chunk[0])["local"]["rows"] *
                         _instruction(chunk[0])["local"]["cols"] for chunk in chunks)
    max_weight = max(_instruction(chunk[1])["local"]["rows"] *
                     _instruction(chunk[1])["local"]["cols"] for chunk in chunks)
    alternate_activation = reserved_words
    alternate_weight = alternate_activation + max_activation
    required_words = alternate_weight + max_weight
    prepared = []
    for index, chunk in enumerate(chunks):
        current = copy.deepcopy(chunk)
        if index % 2:
            activation = _instruction(current[0])
            weight = _instruction(current[1])
            compute = _instruction(current[3])
            activation["local"]["base"] = compute["a"] = alternate_activation
            weight["local"]["base"] = compute["b"] = alternate_weight
        prepared.append(current)
    reordered = [commands[0], *prepared[0][:3]]
    for index, chunk in enumerate(prepared):
        if index + 1 < len(prepared):
            reordered.extend(prepared[index + 1][:2])
        reordered.append(chunk[3])
        if index + 1 < len(prepared):
            reordered.append(prepared[index + 1][2])
    reordered.append(commands[-1])
    return reordered, (required_words + 255) // 256, len(chunks)


def _pipeline_group(group):
    commands = group["commands"]
    updated = []
    maximum_reservation = group["rf_kib"]
    tiles = 0
    pipelined_chunks = 0
    cursor = 0
    while cursor < len(commands):
        if _instruction(commands[cursor]).get("op") != "fill":
            updated.append(commands[cursor])
            cursor += 1
            continue
        ending = next((index for index in range(cursor + 1, len(commands))
                       if _instruction(commands[index]).get("op") == "store"), None)
        if ending is None:
            updated.extend(commands[cursor:])
            break
        tile = commands[cursor:ending + 1]
        converted = _pipeline_tile(tile, group["rf_kib"] * 256)
        if converted:
            tile, reservation, chunks = converted
            maximum_reservation = max(maximum_reservation, reservation)
            tiles += 1
            pipelined_chunks += chunks
        updated.extend(tile)
        cursor = ending + 1
    return dict(group, rf_kib=maximum_reservation, commands=updated), tiles, pipelined_chunks


def pipeline_decode_program(source, destination):
    records = program_records(source)
    header = next(records)
    if not header["decode"]:
        raise ValueError("GEMM prefetch transformation requires decode")
    hardware = header["hardware"]
    stats = {"pipelined_tiles": 0, "pipelined_chunks": 0,
             "unchanged_tiles_due_to_rf_capacity": 0, "waves": 0}
    with Path(destination).open("x") as output:
        output.write(json.dumps(header, separators=(",", ":")) + "\n")
        for record in records:
            if record["record"] == "wave":
                stats["waves"] += 1
                original = record["groups"]
                selected = list(original)
                for index, group in enumerate(original):
                    converted, tiles, chunks = _pipeline_group(group)
                    if not tiles:
                        continue
                    selected[index] = converted
                    try:
                        validate_wave_capacity(selected, hardware)
                    except ValueError:
                        selected[index] = group
                        stats["unchanged_tiles_due_to_rf_capacity"] += tiles
                        continue
                    stats["pipelined_tiles"] += tiles
                    stats["pipelined_chunks"] += chunks
                record = {"record": "wave", "groups": selected}
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
    if not stats["pipelined_tiles"]:
        raise ValueError("no capacity-feasible multi-chunk GEMM tiles")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(pipeline_decode_program(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
