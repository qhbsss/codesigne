"""Keep the bias-fused GEMM output in RF through its immediate residual add."""

import argparse
import json
from pathlib import Path

from .estimate import program_records, validate_wave_capacity
from .fuse_bias import _instruction
from .fuse_gelu import _coverage, _gemm_bias_wave


def _residual_wave(record):
    if record.get("record") != "wave" or not record["groups"]:
        return None
    signatures = set()
    for group in record["groups"]:
        commands = group["commands"]
        cursor = 0
        if not commands:
            return None
        while cursor < len(commands):
            pair = commands[cursor:cursor + 2]
            if len(pair) != 2 or any(_instruction(command).get("op") != "load"
                                      for command in pair):
                return None
            first, second = (_instruction(command) for command in pair)
            cursor += 2
            if cursor < len(commands) and commands[cursor]["command"] == "wait":
                if (any(command["command"] != "async" for command in pair) or
                        set(commands[cursor]["tokens"]) !=
                        {command["token"] for command in pair}):
                    return None
                cursor += 1
            elif any(command["command"] != "run" for command in pair):
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
            if (width < 1 or first["tensor"] == store["tensor"] or
                    second["tensor"] != store["tensor"] or
                    first["global"] != store["global"] or
                    second["global"] != store["global"] or
                    first["local"] != {"base": 0, "rows": 1, "cols": width,
                                       "row_stride": width, "col_stride": 1} or
                    second["local"] != {"base": width, "rows": 1, "cols": width,
                                        "row_stride": width, "col_stride": 1} or
                    store["local"] != first["local"] or
                    store["global"]["rows"] != 1 or
                    store["global"]["cols"] != width or
                    store["global"]["col_stride"] != 1 or
                    store["global"]["base"] % width or
                    add["dst"] != 0 or
                    add["a"] != {"kind": "reg", "base": 0, "stride": 1} or
                    add["b"] != {"kind": "reg", "base": width, "stride": 1}):
                return None
            signatures.add((store["tensor"], first["tensor"], width))
            cursor += 2
    return next(iter(signatures)) if len(signatures) == 1 else None


def _can_fuse(gemm_waves, width):
    for wave in gemm_waves:
        for group in wave["groups"]:
            commands = group["commands"]
            for index, command in enumerate(commands):
                instruction = _instruction(command)
                if instruction.get("op") != "store":
                    continue
                last = next((_instruction(previous) for previous in
                             reversed(commands[:index])
                             if _instruction(previous).get("op") == "mma"), None)
                if (last is None or instruction["global"]["row_stride"] != width or
                        last["b"] + instruction["global"]["rows"] *
                        instruction["global"]["cols"] > group["rf_kib"] * 256):
                    return False
    return True


def _fused_wave(wave, input_tensor, width, hardware):
    for group in wave["groups"]:
        rewritten = []
        for command in group["commands"]:
            instruction = _instruction(command)
            if instruction.get("op") == "store":
                view = instruction["global"]
                rows, cols = view["rows"], view["cols"]
                last = next(_instruction(previous) for previous in reversed(rewritten)
                            if _instruction(previous).get("op") == "mma")
                input_base = last["b"]
                rewritten.append({"command": "run", "instruction": {
                    "op": "load", "tensor": input_tensor,
                    "global": dict(view),
                    "local": {"base": input_base, "rows": rows, "cols": cols,
                              "row_stride": cols, "col_stride": 1}}})
                for row in range(rows):
                    output_base = row * cols
                    rewritten.append({"command": "run", "instruction": {
                        "op": "vector", "kind": "add", "dst": output_base,
                        "len": cols,
                        "a": {"kind": "reg", "base": input_base + output_base,
                              "stride": 1},
                        "b": {"kind": "reg", "base": output_base, "stride": 1}}})
            rewritten.append(command)
        group["commands"] = rewritten
    validate_wave_capacity(wave["groups"], hardware)
    return wave


def fuse_residual_program(source, destination):
    records = program_records(source)
    header = next(records)
    hardware = header["hardware"]
    rows = header["batch"] * (1 if header["decode"] else header["shape"]["prefill"])
    gemm_waves = []
    residual_waves = []
    target = signature = None
    stats = {"stages_fused": 0, "gemm_waves_changed": 0,
             "residual_waves_removed": 0, "output_loads_removed": 0,
             "output_stores_removed": 0}

    def write(output, record):
        output.write(json.dumps(record, separators=(",", ":")) + "\n")

    def flush(output):
        nonlocal target, signature
        width = signature[2] if signature is not None else 0
        success = (signature is not None and
                   _coverage(gemm_waves, residual_waves, target, width, rows) and
                   _can_fuse(gemm_waves, width))
        if success:
            for wave in gemm_waves:
                write(output, _fused_wave(wave, signature[1], width, hardware))
            stats["stages_fused"] += 1
            stats["gemm_waves_changed"] += len(gemm_waves)
            stats["residual_waves_removed"] += len(residual_waves)
            count = sum(_instruction(command).get("op") == "store"
                        for wave in residual_waves for group in wave["groups"]
                        for command in group["commands"])
            stats["output_loads_removed"] += count
            stats["output_stores_removed"] += count
        else:
            for wave in (*gemm_waves, *residual_waves):
                write(output, wave)
        gemm_waves.clear()
        residual_waves.clear()
        target = signature = None

    with Path(destination).open("x") as output:
        write(output, header)
        for record in records:
            gemm_target = _gemm_bias_wave(record)
            residual = _residual_wave(record) if gemm_target is None else None
            if residual_waves:
                if residual == signature:
                    residual_waves.append(record)
                    continue
                flush(output)
            elif gemm_waves:
                if gemm_target == target:
                    gemm_waves.append(record)
                    continue
                if residual is not None and residual[0] == target:
                    signature = residual
                    residual_waves.append(record)
                    continue
                flush(output)
            if gemm_target is not None:
                target = gemm_target
                gemm_waves.append(record)
            else:
                write(output, record)
        flush(output)
    if not stats["stages_fused"]:
        raise ValueError("no complete bias-fused GEMM plus residual stage was found")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(fuse_residual_program(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
