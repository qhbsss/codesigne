"""Fold a complete in-place bias pass into its preceding independent GEMM tiles."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .estimate import program_records, validate_wave_capacity


def _instruction(command):
    return command.get("instruction", {})


def _contiguous(view, rows, cols, base=None, row_stride=None):
    return (view["rows"] == rows and view["cols"] == cols and
            view["col_stride"] == 1 and
            view["row_stride"] == (cols if row_stride is None else row_stride) and
            (base is None or view["base"] == base))


def _gemm_wave(record):
    if record.get("record") != "wave" or not record["groups"]:
        return None
    targets = set()
    for group in record["groups"]:
        commands = group["commands"]
        cursor = 0
        if not commands:
            return None
        while cursor < len(commands):
            if (commands[cursor]["command"] != "run" or
                    _instruction(commands[cursor]).get("op") != "fill"):
                return None
            end = next((index for index in range(cursor + 1, len(commands))
                        if _instruction(commands[index]).get("op") == "store"), None)
            if end is None:
                return None
            tile = commands[cursor:end + 1]
            fill, store = _instruction(tile[0]), _instruction(tile[-1])
            matrix = [(_instruction(command), index) for index, command in enumerate(tile)
                      if _instruction(command).get("op") == "mma"]
            if (not matrix or matrix[-1][1] != len(tile) - 2 or
                    fill["value"] != 0 or fill["dst"] != 0 or
                    any(_instruction(command).get("op") not in ("fill", "load", "mma", "store", None)
                        for command in tile) or
                    any(_instruction(command).get("op") == "load" and
                        _instruction(command)["tensor"] == store["tensor"] for command in tile)):
                return None
            last = matrix[-1][0]
            rows, cols = last["m"], last["n"]
            if (fill["len"] != rows * cols or
                    not _contiguous(store["local"], rows, cols, base=0) or
                    not _contiguous(store["global"], rows, cols, row_stride=store["global"]["row_stride"]) or
                    last["c"] != 0 or last["b"] + cols > group["rf_kib"] * 256 or
                    any(mma["m"] != rows or mma["n"] != cols or mma["c"] != 0
                        for mma, _ in matrix)):
                return None
            targets.add(store["tensor"])
            cursor = end + 1
    return next(iter(targets)) if len(targets) == 1 else None


def _bias_wave(record):
    if record.get("record") != "wave" or not record["groups"]:
        return None
    signatures = set()
    for group in record["groups"]:
        commands = group["commands"]
        cursor = 0
        if not commands:
            return None
        while cursor < len(commands):
            start = commands[cursor:cursor + 2]
            if len(start) != 2 or any(_instruction(command).get("op") != "load"
                                      for command in start):
                return None
            first, bias = (_instruction(command) for command in start)
            cursor += 2
            if cursor < len(commands) and commands[cursor]["command"] == "wait":
                if (any(command["command"] != "async" for command in start) or
                        set(commands[cursor]["tokens"]) !=
                        {command["token"] for command in start}):
                    return None
                cursor += 1
            elif any(command["command"] != "run" for command in start):
                return None
            if cursor + 2 > len(commands):
                return None
            add, store = (_instruction(command) for command in commands[cursor:cursor + 2])
            if (commands[cursor]["command"] != "run" or
                    commands[cursor + 1]["command"] != "run" or
                    add.get("op") != "vector" or add["kind"] != "add" or
                    store.get("op") != "store"):
                return None
            width = add["len"]
            if (width < 1 or first["tensor"] != store["tensor"] or
                    bias["tensor"] == store["tensor"] or
                    first["global"] != store["global"] or
                    first["local"] != store["local"] or
                    not _contiguous(first["global"], 1, width,
                                    row_stride=width) or
                    first["global"]["base"] % width or
                    not _contiguous(first["local"], 1, width, base=0) or
                    not _contiguous(bias["global"], 1, width, base=0) or
                    not _contiguous(bias["local"], 1, width, base=width) or
                    add["dst"] != 0 or
                    add["a"] != {"kind": "reg", "base": 0, "stride": 1} or
                    add["b"] != {"kind": "reg", "base": width, "stride": 1}):
                return None
            signatures.add((store["tensor"], bias["tensor"], width))
            cursor += 2
    return next(iter(signatures)) if len(signatures) == 1 else None


def _full_output_coverage(gemm_waves, bias_waves, output, width, rows):
    columns = defaultdict(list)
    for record in gemm_waves:
        for group in record["groups"]:
            for command in group["commands"]:
                instruction = _instruction(command)
                if instruction.get("op") != "store":
                    continue
                if instruction["tensor"] != output:
                    return False
                view = instruction["global"]
                if view["row_stride"] != width:
                    return False
                start_row, start_col = divmod(view["base"], width)
                if start_row + view["rows"] > rows or start_col + view["cols"] > width:
                    return False
                for row in range(start_row, start_row + view["rows"]):
                    columns[row].append((start_col, start_col + view["cols"]))
    bias_rows = set()
    for record in bias_waves:
        for group in record["groups"]:
            for command in group["commands"]:
                instruction = _instruction(command)
                if instruction.get("op") == "store":
                    row = instruction["global"]["base"] // width
                    if row in bias_rows:
                        return False
                    bias_rows.add(row)
    if bias_rows != set(range(rows)) or set(columns) != bias_rows:
        return False
    for intervals in columns.values():
        cursor = 0
        for first, last in sorted(intervals):
            if first != cursor:
                return False
            cursor = last
        if cursor != width:
            return False
    return True


def _fused_wave(record, bias_tensor, width, hardware):
    for group in record["groups"]:
        fused = []
        for command in group["commands"]:
            instruction = _instruction(command)
            if instruction.get("op") == "store":
                store = instruction
                last = _instruction(fused[-1])
                rows, cols = store["global"]["rows"], store["global"]["cols"]
                bias_base = last["b"]
                bias_col = store["global"]["base"] % width
                fused.append({"command": "run", "instruction": {
                    "op": "load", "tensor": bias_tensor,
                    "global": {"base": bias_col, "rows": 1, "cols": cols,
                               "row_stride": cols, "col_stride": 1},
                    "local": {"base": bias_base, "rows": 1, "cols": cols,
                              "row_stride": cols, "col_stride": 1}}})
                for row in range(rows):
                    base = store["local"]["base"] + row * cols
                    fused.append({"command": "run", "instruction": {
                        "op": "vector", "kind": "add", "dst": base, "len": cols,
                        "a": {"kind": "reg", "base": base, "stride": 1},
                        "b": {"kind": "reg", "base": bias_base, "stride": 1}}})
            fused.append(command)
        group["commands"] = fused
    validate_wave_capacity(record["groups"], hardware)
    return record


def fuse_bias_program(source, destination):
    records = program_records(source)
    header = next(records)
    rows = header["batch"] * (1 if header["decode"] else header["shape"]["prefill"])
    hardware = header["hardware"]
    gemm_waves = []
    bias_waves = []
    target = None
    signature = None
    stats = {"stages_fused": 0, "gemm_waves_changed": 0, "bias_waves_removed": 0,
             "bias_tensor_loads_removed": 0, "output_tensor_loads_removed": 0,
             "output_tensor_stores_removed": 0}

    def write(output, record):
        output.write(json.dumps(record, separators=(",", ":")) + "\n")

    def flush(output):
        nonlocal target, signature
        success = (signature is not None and
                   _full_output_coverage(gemm_waves, bias_waves, target, signature[2], rows))
        if success:
            for record in gemm_waves:
                write(output, _fused_wave(record, signature[1], signature[2], hardware))
            stats["stages_fused"] += 1
            stats["gemm_waves_changed"] += len(gemm_waves)
            stats["bias_waves_removed"] += len(bias_waves)
            for record in bias_waves:
                for group in record["groups"]:
                    stores = sum(_instruction(command).get("op") == "store"
                                 for command in group["commands"])
                    stats["bias_tensor_loads_removed"] += stores
                    stats["output_tensor_loads_removed"] += stores
                    stats["output_tensor_stores_removed"] += stores
        else:
            for record in (*gemm_waves, *bias_waves):
                write(output, record)
        gemm_waves.clear()
        bias_waves.clear()
        signature = target = None

    with Path(destination).open("x") as output:
        write(output, header)
        for record in records:
            gemm_target = _gemm_wave(record)
            bias = _bias_wave(record) if gemm_target is None else None
            if bias_waves:
                if bias == signature:
                    bias_waves.append(record)
                    continue
                flush(output)
            elif gemm_waves:
                if gemm_target == target:
                    gemm_waves.append(record)
                    continue
                if bias is not None and bias[0] == target:
                    signature = bias
                    bias_waves.append(record)
                    continue
                flush(output)
            if gemm_target is not None:
                target = gemm_target
                gemm_waves.append(record)
            else:
                write(output, record)
        flush(output)
    if not stats["stages_fused"]:
        raise ValueError("no complete GEMM plus bias stage was found")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(fuse_bias_program(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
