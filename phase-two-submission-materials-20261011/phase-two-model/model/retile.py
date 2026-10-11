"""Retile the initial persistent-key transpose without changing tensor values."""

import argparse
import copy
import json
from pathlib import Path

from .estimate import program_records, validate_wave_capacity


def _view(base, rows, cols, row_stride, col_stride=1):
    return {"base": base, "rows": rows, "cols": cols,
            "row_stride": row_stride, "col_stride": col_stride}


def _pair(group, source_id, target_id, row, col, rows, cols, capacity):
    commands = group["commands"]
    instructions = [command["instruction"] for command in commands
                    if "instruction" in command]
    nr, nc = min(16, rows - row), min(16, cols - col)
    return (len(instructions) == 2 and
            [instruction["op"] for instruction in instructions] == ["load", "store"] and
            [instruction["tensor"] for instruction in instructions] == [source_id, target_id] and
            instructions[0]["global"] == _view(col * rows + row, nc, nr, rows) and
            instructions[0]["local"] == _view(0, nc, nr, nr) and
            instructions[1]["global"] == _view(row * capacity + col, nr, nc, capacity) and
            instructions[1]["local"] == _view(0, nr, nc, 1, nr))


def _replacement(template, source_id, target_id, rows, cols, capacity, tile, hardware):
    wave = []
    group_index = 0
    maximum_rf_kib = (min(tile, rows) * min(tile, cols) * 4 + 1023) // 1024
    group_limit = hardware["sms"] * min(hardware["resident_groups"],
                                          hardware["rf_kib"] // maximum_rf_kib)
    if group_limit == 0:
        raise ValueError("transpose tile exceeds RF capacity")
    for row in range(0, rows, tile):
        for col in range(0, cols, tile):
            nr, nc = min(tile, rows - row), min(tile, cols - col)
            group = copy.deepcopy(template)
            group["sm"] = group_index % hardware["sms"]
            group["rf_kib"] = (nr * nc * 4 + 1023) // 1024
            instructions = [command["instruction"] for command in group["commands"]
                            if "instruction" in command]
            instructions[0]["global"] = _view(col * rows + row, nc, nr, rows)
            instructions[0]["local"] = _view(0, nc, nr, nr)
            instructions[1]["global"] = _view(row * capacity + col, nr, nc, capacity)
            instructions[1]["local"] = _view(0, nr, nc, 1, nr)
            wave.append(group)
            group_index += 1
            if len(wave) == group_limit:
                validate_wave_capacity(wave, hardware)
                yield {"record": "wave", "groups": wave}
                wave = []
    if wave:
        validate_wave_capacity(wave, hardware)
        yield {"record": "wave", "groups": wave}


def retile_program(source, destination, tile):
    if tile not in (16, 32, 64, 128):
        raise ValueError("transpose tile must be one of 16,32,64,128")
    records = program_records(source)
    header = next(records)
    if not header["decode"]:
        raise ValueError("initial key transpose only exists for decode")
    rows = header["batch"] * header["shape"]["d"]
    cols = header["shape"]["history"]
    capacity = cols + header["shape"]["steps"]
    tiles_per_row = (cols + 15) // 16
    tiles_total = ((rows + 15) // 16) * tiles_per_row
    hardware = header["hardware"]
    tensor_ids = []
    initial_keys = set()
    completed = set()
    active = None
    statistics = {"tile": tile, "initial_transposes": 0,
                  "old_waves": 0, "new_waves": 0, "old_tiles": 0}

    def emit(writer, record):
        writer.write(json.dumps(record, separators=(",", ":")) + "\n")

    with Path(destination).open("x") as writer:
        emit(writer, header)
        for record in records:
            kind = record.get("record")
            if kind != "wave" and active is not None:
                raise ValueError("initial transpose interrupted before all tiles")
            if kind == "input":
                tensor_id = len(tensor_ids)
                tensor_ids.append(record["len"])
                if record["source"].endswith(".past_k"):
                    initial_keys.add(tensor_id)
            elif kind == "alloc":
                tensor_ids.append(record["len"])
            elif kind == "release":
                if record["tensor"] != len(tensor_ids) - 1:
                    raise ValueError("unexpected tensor release order")
                tensor_ids.pop()
            if kind == "wave" and record["groups"]:
                group = record["groups"][0]
                instructions = [command["instruction"] for command in group["commands"]
                                if "instruction" in command]
                if len(instructions) == 2 and instructions[0]["op"] == "load":
                    source_id = instructions[0]["tensor"]
                    target_id = instructions[1].get("tensor")
                    if source_id in initial_keys and source_id not in completed and (
                            target_id is not None and target_id < len(tensor_ids) and
                            tensor_ids[target_id] == capacity * rows):
                        if active is None:
                            active = (source_id, target_id, copy.deepcopy(group), 0)
                        old_source, old_target, template, seen = active
                        if (source_id, target_id) != (old_source, old_target):
                            raise ValueError("initial transpose changed tensor ids")
                        for member in record["groups"]:
                            row = seen // tiles_per_row * 16
                            col = seen % tiles_per_row * 16
                            if seen >= tiles_total or not _pair(
                                    member, source_id, target_id, row, col,
                                    rows, cols, capacity):
                                raise ValueError("initial transpose no longer matches the generator")
                            seen += 1
                        active = (source_id, target_id, template, seen)
                        statistics["old_waves"] += 1
                        statistics["old_tiles"] += len(record["groups"])
                        if seen == tiles_total:
                            for wave in _replacement(template, source_id, target_id,
                                                     rows, cols, capacity, tile, hardware):
                                emit(writer, wave)
                                statistics["new_waves"] += 1
                            completed.add(source_id)
                            statistics["initial_transposes"] += 1
                            active = None
                        continue
            if active is not None:
                raise ValueError("initial transpose interrupted before all tiles")
            emit(writer, record)
        if active is not None or statistics["initial_transposes"] != header["shape"]["layers"]:
            raise ValueError("did not find every initial key transpose")
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--tile", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(retile_program(args.source, args.destination, args.tile), indent=2))


if __name__ == "__main__":
    main()
