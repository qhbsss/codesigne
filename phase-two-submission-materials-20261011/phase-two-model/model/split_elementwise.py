"""Restore a barrier between merged bias and residual passes before RF fusion."""

import argparse
import json
from pathlib import Path

from .estimate import program_records, validate_wave_capacity
from .fuse_bias import _bias_wave, _instruction
from .fuse_residual import _residual_wave


def _split_wave(record, hardware):
    if record.get("record") != "wave" or not record["groups"]:
        return None
    prefixes = []
    suffixes = []
    for group in record["groups"]:
        commands = group["commands"]
        if len(commands) < 8:
            return None
        asynchronous = all(command["command"] == "async" for command in commands[:2])
        size = 5 if asynchronous else 4
        if len(commands) % size:
            return None
        pieces = [commands[index:index + size] for index in range(0, len(commands), size)]
        pattern = (("load", "load", None, "vector", "store") if asynchronous else
                   ("load", "load", "vector", "store"))
        if any(tuple(_instruction(command).get("op") for command in piece) !=
               pattern for piece in pieces):
            return None
        output = _instruction(pieces[0][-1]).get("tensor")
        boundary = next((index for index, piece in enumerate(pieces)
                         if _instruction(piece[0]).get("tensor") != output), None)
        if boundary is None or boundary == 0:
            return None
        prefix = dict(group, commands=commands[:boundary * size])
        suffix = dict(group, commands=commands[boundary * size:])
        prefixes.append(prefix)
        suffixes.append(suffix)
    first = {"record": "wave", "groups": prefixes}
    second = {"record": "wave", "groups": suffixes}
    bias = _bias_wave(first)
    residual = _residual_wave(second)
    if bias is None or residual is None or bias[0] != residual[0] or bias[2] != residual[2]:
        return None
    validate_wave_capacity(prefixes, hardware)
    validate_wave_capacity(suffixes, hardware)
    return first, second


def split_elementwise_program(source, destination):
    records = program_records(source)
    header = next(records)
    stats = {"compound_waves_split": 0, "barriers_inserted": 0}
    with Path(destination).open("x") as writer:
        writer.write(json.dumps(header, separators=(",", ":")) + "\n")
        for record in records:
            pieces = _split_wave(record, header["hardware"])
            if pieces is None:
                pieces = (record,)
            else:
                stats["compound_waves_split"] += 1
                stats["barriers_inserted"] += 1
            for piece in pieces:
                writer.write(json.dumps(piece, separators=(",", ":")) + "\n")
    if not stats["compound_waves_split"]:
        raise ValueError("no merged bias/residual wave found")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(split_elementwise_program(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
