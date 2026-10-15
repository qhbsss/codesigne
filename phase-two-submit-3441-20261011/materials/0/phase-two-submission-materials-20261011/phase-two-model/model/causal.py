"""Prune causal-attention score tiles and KV products never read by a row tile."""

import argparse
import copy
import json
from pathlib import Path

from .estimate import program_records


def _score_tensors(source, rows, heads, layers):
    consecutive = []
    selected = []
    tensor_id = 0
    for record in program_records(source):
        if record.get("record") not in ("input", "alloc"):
            continue
        if record["record"] == "alloc" and record["len"] == rows * rows:
            consecutive.append(tensor_id)
        else:
            if len(consecutive) == heads:
                selected.extend(consecutive)
            consecutive = []
        tensor_id += 1
    if len(consecutive) == heads:
        selected.extend(consecutive)
    if len(selected) != heads * layers:
        raise ValueError("could not uniquely identify the attention score tensors")
    return set(selected)


def _limit(row, tile_rows, context, reduction_tile):
    return min(context, ((row + tile_rows + reduction_tile - 1) // reduction_tile) * reduction_tile)


def _opcode(command):
    return command.get("instruction", {}).get("op")


def _trim_mask(tile, tensor_ids, context, row_tile, reduction_tile, stats):
    fill = tile[0]["instruction"]
    store = tile[-1]["instruction"]
    view = store["global"]
    row, start = divmod(view["base"], context)
    if (store["tensor"] not in tensor_ids or len(tile) != 2 or
            view["rows"] != 1 or view["cols"] != context - start or
            start != row + 1 or view["col_stride"] != 1 or
            fill["len"] != view["cols"] or fill["value"] != 0 or
            store["local"]["rows"] != 1 or
            store["local"]["cols"] != view["cols"]):
        raise ValueError("unexpected attention mask layout")
    row_start = row // row_tile * row_tile
    limit = _limit(row_start, min(row_tile, context - row_start), context, reduction_tile)
    retained = min(view["cols"], max(0, limit - start))
    stats["mask_words_removed"] += view["cols"] - retained
    if not retained:
        stats["mask_pairs_removed"] += 1
        return []
    if retained == view["cols"]:
        return tile
    kept = copy.deepcopy(tile)
    kept[0]["instruction"]["len"] = retained
    kept[-1]["instruction"]["global"]["cols"] = retained
    kept[-1]["instruction"]["local"]["cols"] = retained
    return kept


def _prune_qk(tile, tensor_ids, context, row_tile, reduction_tile, stats):
    store = tile[-1]["instruction"]
    view = store["global"]
    row, column = divmod(view["base"], context)
    if (store["tensor"] not in tensor_ids or view["row_stride"] != context or
            view["col_stride"] != 1 or view["cols"] > reduction_tile or
            view["rows"] != min(row_tile, context - row) or
            not all(command["command"] == "wait" or
                    _opcode(command) in ("fill", "load", "mma") for command in tile[:-1])):
        raise ValueError("unexpected attention QK tile layout")
    if column < _limit(row, view["rows"], context, reduction_tile):
        return tile
    stats["qk_tiles_removed"] += 1
    stats["qk_commands_removed"] += len(tile)
    return []


def _prune_av(tile, tensor_ids, context, row_tile, reduction_tile, stats):
    body = tile[1:-1]
    kept = [tile[0]]
    score_tensor = None
    offset = 0
    while offset < len(body):
        first, second = body[offset:offset + 2]
        asynchronous = first["command"] == "async"
        width = 4 if asynchronous else 3
        chunk = body[offset:offset + width]
        if len(chunk) != width:
            raise ValueError("incomplete attention AV chunk")
        mma = chunk[-1]
        if (second["command"] != first["command"] or
                first["command"] not in ("run", "async") or
                (asynchronous and chunk[2] != {"command": "wait", "tokens":
                    [first["token"], second["token"]]}) or
                _opcode(first) != "load" or _opcode(second) != "load" or
                _opcode(mma) != "mma" or mma["command"] != "run"):
            raise ValueError("unexpected attention AV command ordering")
        instruction = first["instruction"]
        view = instruction["global"]
        row, column = divmod(view["base"], context)
        if (instruction["tensor"] not in tensor_ids or
                view["row_stride"] != context or view["col_stride"] != 1 or
                view["rows"] != min(row_tile, context - row) or
                view["cols"] > reduction_tile or column % reduction_tile):
            raise ValueError("unexpected attention AV score view")
        if score_tensor is None:
            score_tensor = instruction["tensor"]
        if score_tensor != instruction["tensor"]:
            raise ValueError("attention AV mixes score tensors")
        if column >= _limit(row, view["rows"], context, reduction_tile):
            stats["av_chunks_removed"] += 1
            stats["av_commands_removed"] += len(chunk)
        else:
            kept.extend(chunk)
        offset += width
    return kept + [tile[-1]]


def _transform_group(group, tensor_ids, context, row_tile, reduction_tile, stats):
    commands = group["commands"]
    retained = []
    cursor = 0
    while cursor < len(commands):
        if _opcode(commands[cursor]) != "fill":
            retained.append(commands[cursor])
            cursor += 1
            continue
        end = next((index for index in range(cursor + 1, len(commands))
                    if _opcode(commands[index]) == "store"), None)
        if end is None:
            retained.extend(commands[cursor:])
            break
        tile = commands[cursor:end + 1]
        score_stores = tile[-1]["instruction"]["tensor"] in tensor_ids
        score_loads = any(_opcode(command) == "load" and
                          command["instruction"]["tensor"] in tensor_ids
                          for command in tile[:-1])
        if score_stores and score_loads:
            raise ValueError("unsupported in-place score tile")
        if score_stores:
            tile = (_trim_mask(tile, tensor_ids, context, row_tile, reduction_tile, stats)
                    if len(tile) == 2 else
                    _prune_qk(tile, tensor_ids, context, row_tile, reduction_tile, stats))
        elif score_loads:
            tile = _prune_av(tile, tensor_ids, context, row_tile, reduction_tile, stats)
        retained.extend(tile)
        cursor = end + 1
    return dict(group, commands=retained)


def prune_causal_program(source, destination, row_tile, reduction_tile):
    records = program_records(source)
    header = next(records)
    if header["decode"] or header["batch"] != 1 or row_tile < 1 or reduction_tile < 1:
        raise ValueError("causal tile pruning supports batch-one prefill and positive tiles only")
    shape = header["shape"]
    context = shape["prefill"]
    score_ids = _score_tensors(source, context, shape["heads"], shape["layers"])
    stats = {"score_tensors": sorted(score_ids), "qk_tiles_removed": 0,
             "qk_commands_removed": 0, "av_chunks_removed": 0,
             "av_commands_removed": 0, "mask_pairs_removed": 0,
             "mask_words_removed": 0, "source_waves": 0, "output_waves": 0}
    with Path(destination).open("x") as output:
        output.write(json.dumps(header, separators=(",", ":")) + "\n")
        for record in records:
            if record["record"] == "wave":
                stats["source_waves"] += 1
                groups = [_transform_group(group, score_ids, context, row_tile,
                                           reduction_tile, stats) for group in record["groups"]]
                groups = [group for group in groups if group["commands"]]
                if not groups:
                    continue
                record = {"record": "wave", "groups": groups}
                stats["output_waves"] += 1
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
    if not stats["qk_tiles_removed"] or not stats["av_chunks_removed"] or not stats["mask_words_removed"]:
        raise ValueError("expected causal QK, AV and mask work was not removed")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--row-tile", type=int, required=True)
    parser.add_argument("--reduction-tile", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(prune_causal_program(args.source, args.destination,
                                          args.row_tile, args.reduction_tile), indent=2))


if __name__ == "__main__":
    main()
