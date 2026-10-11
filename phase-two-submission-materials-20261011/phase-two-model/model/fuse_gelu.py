"""Fold the immediate in-place GELU pass into a bias-fused GEMM's RF output."""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from .estimate import program_records, validate_wave_capacity
from .fuse_bias import _instruction


GELU_KINDS = ("square", "mul", "mul", "add", "mul", "tanh", "add", "mul", "mul")


def _gemm_bias_wave(record):
    if record.get("record") != "wave" or not record["groups"]:
        return None
    destinations = set()
    for group in record["groups"]:
        commands = group["commands"]
        cursor = 0
        while cursor < len(commands):
            if _instruction(commands[cursor]).get("op") != "fill":
                return None
            end = next((index for index in range(cursor + 1, len(commands))
                        if _instruction(commands[index]).get("op") == "store"), None)
            if end is None:
                return None
            tile = commands[cursor:end + 1]
            store = _instruction(tile[-1])
            matrix = [index for index, command in enumerate(tile)
                      if _instruction(command).get("op") == "mma"]
            if not matrix:
                return None
            mma = _instruction(tile[matrix[-1]])
            rows, cols = store["global"]["rows"], store["global"]["cols"]
            tail = tile[matrix[-1] + 1:-1]
            if (mma["m"] != rows or mma["n"] != cols or mma["c"] != 0 or
                    len(tail) != rows + 1 or
                    _instruction(tail[0]).get("op") != "load" or
                    _instruction(tail[0])["local"]["base"] != mma["b"] or
                    _instruction(tail[0])["global"]["base"] !=
                    store["global"]["base"] % store["global"]["row_stride"] or
                    any(_instruction(command).get("op") != "vector" or
                        _instruction(command).get("kind") != "add" or
                        _instruction(command)["dst"] != row * cols or
                        _instruction(command)["len"] != cols or
                        _instruction(command)["a"] !=
                        {"kind": "reg", "base": row * cols, "stride": 1} or
                        _instruction(command)["b"] !=
                        {"kind": "reg", "base": mma["b"], "stride": 1}
                        for row, command in enumerate(tail[1:])) or
                    store["local"]["base"] != 0 or
                    store["local"]["row_stride"] != cols or
                    store["global"]["col_stride"] != 1 or
                    mma["b"] + 2 * cols > group["rf_kib"] * 256):
                return None
            destinations.add(store["tensor"])
            cursor = end + 1
    return next(iter(destinations)) if len(destinations) == 1 else None


def _gelu_wave(record):
    if record.get("record") != "wave" or not record["groups"]:
        return None
    tensors = set()
    templates = set()
    for group in record["groups"]:
        commands = group["commands"]
        cursor = 0
        if not commands:
            return None
        while cursor < len(commands):
            load = commands[cursor]
            source = _instruction(load)
            if source.get("op") != "load":
                return None
            cursor += 1
            if cursor < len(commands) and commands[cursor]["command"] == "wait":
                if (load["command"] != "async" or
                        commands[cursor]["tokens"] != [load["token"]]):
                    return None
                cursor += 1
            elif load["command"] != "run":
                return None
            body = commands[cursor:cursor + 10]
            if len(body) != 10 or any(command["command"] != "run" for command in body):
                return None
            operations = [_instruction(command) for command in body[:-1]]
            store = _instruction(body[-1])
            length = source["global"]["cols"]
            if (length < 1 or length > 1024 or
                    store.get("op") != "store" or
                    source["tensor"] != store["tensor"] or
                    source["global"] != store["global"] or
                    source["local"] != store["local"] or
                    source["global"]["rows"] != 1 or
                    source["global"]["col_stride"] != 1 or
                    source["local"] != {"base": 0, "rows": 1, "cols": length,
                                        "row_stride": length, "col_stride": 1} or
                    tuple(operation.get("kind") for operation in operations) != GELU_KINDS or
                    any(operation["len"] != length for operation in operations) or
                    [operation["dst"] for operation in operations] !=
                    [length] * 8 + [0]):
                return None
            normalized = []
            for operation in operations:
                selected = []
                for key in ("a", "b"):
                    argument = operation[key]
                    if argument["kind"] == "reg":
                        if argument["stride"] != 1 or argument["base"] not in (0, length):
                            return None
                        selected.append(("reg", argument["base"] // length))
                    elif argument["kind"] == "imm":
                        selected.append(("imm", argument["value"]))
                    else:
                        return None
                normalized.append((operation["kind"], operation["dst"] // length,
                                   tuple(selected)))
            tensors.add(store["tensor"])
            templates.add(tuple(normalized))
            cursor += 10
    if len(tensors) != 1 or len(templates) != 1:
        return None
    return next(iter(tensors)), next(iter(templates))


def _coverage(gemm_waves, gelu_waves, output, width, rows):
    if width < 1 or width * rows < 1:
        return False
    intervals = [defaultdict(list), defaultdict(list)]
    for choice, waves in enumerate((gemm_waves, gelu_waves)):
        for wave in waves:
            for group in wave["groups"]:
                for command in group["commands"]:
                    instruction = _instruction(command)
                    if instruction.get("op") != "store":
                        continue
                    view = instruction["global"]
                    if (instruction["tensor"] != output or view["col_stride"] != 1 or
                            (view["rows"] > 1 and view["row_stride"] != width)):
                        return False
                    first_row, col = divmod(view["base"], width)
                    if first_row + view["rows"] > rows or col + view["cols"] > width:
                        return False
                    for row in range(first_row, first_row + view["rows"]):
                        intervals[choice][row].append((col, col + view["cols"]))
    for addresses in intervals:
        if set(addresses) != set(range(rows)):
            return False
        for segments in addresses.values():
            cursor = 0
            for first, last in sorted(segments):
                if first != cursor:
                    return False
                cursor = last
            if cursor != width:
                return False
    return True


def _fused_wave(record, template, hardware):
    for group in record["groups"]:
        rewritten = []
        for command in group["commands"]:
            instruction = _instruction(command)
            if instruction.get("op") == "store":
                store = instruction
                rows, cols = store["global"]["rows"], store["global"]["cols"]
                last_mma = next(_instruction(previous) for previous in reversed(rewritten)
                                if _instruction(previous).get("op") == "mma")
                scratch = last_mma["b"]
                for row in range(rows):
                    registers = (row * cols, scratch, scratch + cols)
                    for kind, dst, arguments in template:
                        mapped = []
                        for arg_kind, value in arguments:
                            mapped.append({"kind": "reg", "base": registers[value], "stride": 1}
                                          if arg_kind == "reg" else
                                          {"kind": "imm", "value": value})
                        rewritten.append({"command": "run", "instruction": {
                            "op": "vector", "kind": kind, "dst": registers[dst],
                            "len": cols, "a": mapped[0], "b": mapped[1]}})
            rewritten.append(command)
        group["commands"] = rewritten
    validate_wave_capacity(record["groups"], hardware)
    return record


def fuse_gelu_program(source, destination):
    records = program_records(source)
    header = next(records)
    hardware = header["hardware"]
    rows = header["batch"] * (1 if header["decode"] else header["shape"]["prefill"])
    gemm_waves = []
    gelu_waves = []
    target = template = None
    stats = {"stages_fused": 0, "gemm_waves_changed": 0,
             "gelu_waves_removed": 0, "gelu_elements_relocated": 0}

    def write(output, record):
        output.write(json.dumps(record, separators=(",", ":")) + "\n")

    def flush(output):
        nonlocal target, template
        width = header["shape"]["f"]
        success = (template is not None and
                   _coverage(gemm_waves, gelu_waves, target, width, rows))
        if success:
            for wave in gemm_waves:
                write(output, _fused_wave(wave, template, hardware))
            stats["stages_fused"] += 1
            stats["gemm_waves_changed"] += len(gemm_waves)
            stats["gelu_waves_removed"] += len(gelu_waves)
            stats["gelu_elements_relocated"] += rows * width
        else:
            for wave in (*gemm_waves, *gelu_waves):
                write(output, wave)
        gemm_waves.clear()
        gelu_waves.clear()
        target = template = None

    with Path(destination).open("x") as output:
        write(output, header)
        for record in records:
            gemm_target = _gemm_bias_wave(record)
            gelu = _gelu_wave(record) if gemm_target is None else None
            if gelu_waves:
                if gelu == (target, template):
                    gelu_waves.append(record)
                    continue
                flush(output)
            elif gemm_waves:
                if gemm_target == target:
                    gemm_waves.append(record)
                    continue
                if gelu is not None and gelu[0] == target:
                    template = gelu[1]
                    gelu_waves.append(record)
                    continue
                flush(output)
            if gemm_target is not None:
                target = gemm_target
                gemm_waves.append(record)
            else:
                write(output, record)
        flush(output)
    if not stats["stages_fused"]:
        raise ValueError("no complete bias-fused GEMM plus GELU stage was found")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(fuse_gelu_program(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
