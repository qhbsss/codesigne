#!/usr/bin/env python3
"""Losslessly fold affine repetition in expanded static JSONL; no tensor data."""

import argparse
import copy
import json
from pathlib import Path

MODEL = "phase-two-compact-v2"
CONTRACT = "phase-two-static-v1"


def affine(a, b, var):
    """Return first value with differing integer leaves replaced by affine terms."""
    if type(a) is int and type(b) is int:
        if a == b:
            return a
        return {"add": [a, {"mul": [{"var": var}, b - a]}]}
    if type(a) is not type(b):
        raise ValueError("different shapes")
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("different fields")
        return {k: affine(a[k], b[k], var) for k in a}
    if isinstance(a, list):
        if len(a) != len(b):
            raise ValueError("different lengths")
        return [affine(x, y, var) for x, y in zip(a, b)]
    if a != b:
        raise ValueError("different constants")
    return a


def matches(a, b, x, iteration):
    if type(a) is int and type(b) is int:
        return type(x) is int and x == a + iteration * (b - a)
    if type(a) is not type(x):
        return False
    if isinstance(a, dict):
        return a.keys() == x.keys() and all(matches(a[k], b[k], x[k], iteration) for k in a)
    if isinstance(a, list):
        return len(a) == len(x) and all(matches(i, j, k, iteration) for i, j, k in zip(a, b, x))
    return a == x


def repeat(tag, var, count, body):
    return {tag: "repeat", "var": var, "start": 0, "count": count, "step": 1, "body": body}


def key(command):
    instruction = command.get("instruction", {})
    return command.get("command"), instruction.get("op"), instruction.get("kind")


def fold_commands(commands, depth=0):
    if depth >= 3:
        return commands
    result, at = [], 0
    var = f"c{depth}"
    while at < len(commands):
        best = None
        # Search a small number of plausible periods rather than O(n^2) matching.
        periods = [
            p
            for p in range(1, min(128, (len(commands) - at) // 3) + 1)
            if key(commands[at]) == key(commands[at + p])
        ][:12]
        for period in periods:
            a, b = commands[at : at + period], commands[at + period : at + 2 * period]
            try:
                body = affine(a, b, var)
            except ValueError:
                continue
            count = 2
            while at + (count + 1) * period <= len(commands) and matches(
                a, b, commands[at + count * period : at + (count + 1) * period], count
            ):
                count += 1
            if count >= 3 and (best is None or (count - 1) * period > best[0]):
                best = (count - 1) * period, period, count, body
        if best:
            _, period, count, body = best
            result.append(repeat("command", var, count, fold_commands(body, depth + 1)))
            at += period * count
        else:
            result.append(commands[at])
            at += 1
    return result


def compact_record(record):
    if record.get("record") == "wave":
        record = copy.deepcopy(record)
        for group in record["groups"]:
            group["commands"] = fold_commands(group["commands"])
    return record


def convert(source, destination):
    stats = {"source_records": 0, "compact_records": 0, "top_repeats": 0}
    with source.open() as reader, destination.open("x") as writer:
        header = json.loads(reader.readline())
        if header.get("contract") not in {
            "vnext-static-v09",
            "phase-two-long-waves-static-v1",
            CONTRACT,
        }:
            raise ValueError("converter expects an expanded v0.9 or long-waves program")
        header.update(model=MODEL, contract=CONTRACT)
        writer.write(json.dumps(header, separators=(",", ":")) + "\n")
        first = second = None
        count = 0

        def emit():
            if first is None:
                return
            if count >= 3:
                body = compact_record(affine(first, second, "r"))
                output = repeat("record", "r", count, [body])
                stats["top_repeats"] += 1
                stats["compact_records"] += 1
                writer.write(json.dumps(output, separators=(",", ":")) + "\n")
            else:
                for record in [first, second][:count]:
                    stats["compact_records"] += 1
                    writer.write(json.dumps(compact_record(record), separators=(",", ":")) + "\n")

        for line in reader:
            record = json.loads(line)
            if record.get("record") not in {"input", "alloc", "release", "wave", "commit"}:
                raise ValueError("converter requires expanded input records")
            stats["source_records"] += 1
            if first is None:
                first, count = record, 1
                continue
            if second is None:
                try:
                    affine(first, record, "r")
                except ValueError:
                    emit()
                    first, count = record, 1
                else:
                    second, count = record, 2
            elif matches(first, second, record, count):
                count += 1
            else:
                emit()
                first, second, count = record, None, 1
        emit()
    stats.update(source_bytes=source.stat().st_size, compact_bytes=destination.stat().st_size)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--share-templates", action="store_true")
    args = parser.parse_args()
    action = share_templates if args.share_templates else compile_compact
    print(json.dumps(action(args.source, args.destination), indent=2))


def share_templates(source, destination, freeze_shapes=False):
    """Two-pass structural sharing; integer columns with proportional deltas share a parameter."""
    import hashlib
    import math

    frozen = {"rows", "cols", "row_stride", "col_stride", "len", "m", "n", "k", "count", "step"}

    if not freeze_shapes:
        frozen = set()

    def structure(value):
        digest, integers = hashlib.sha256(), []

        def visit(v, fixed=False):
            if type(v) is int and not fixed:
                digest.update(b"I;")
                integers.append(v)
            elif isinstance(v, dict):
                digest.update(b"{")
                for key in sorted(v):
                    digest.update(json.dumps(key).encode())
                    visit(v[key], fixed or key in frozen)
                digest.update(b"}")
            elif isinstance(v, list):
                digest.update(b"[")
                for item in v:
                    visit(item, fixed)
                digest.update(b"]")
            else:
                digest.update(json.dumps(v).encode())
                digest.update(b";")

        visit(value)
        return digest.hexdigest(), integers

    clusters = {}
    with source.open() as reader:
        header = json.loads(reader.readline())
        for line in reader:
            record = json.loads(line)
            signature, numbers = structure(record)
            cluster = clusters.setdefault(signature, {"first": record, "rows": []})
            cluster["rows"].append(numbers)
    retained = 0
    templates = {}
    for signature, cluster in clusters.items():
        rows = cluster["rows"]
        if len(rows) < 2 or len(templates) >= 4096:
            continue
        columns, parameters = [], {}
        for i, base in enumerate(rows[0]):
            deltas = [row[i] - base for row in rows]
            divisor = math.gcd(*deltas)
            if not divisor:
                columns.append(base)
                continue
            if next(d for d in deltas if d) < 0:
                divisor = -divisor
            vector = tuple(d // divisor for d in deltas)
            index = parameters.setdefault(vector, len(parameters))
            columns.append({"add": [base, {"mul": [{"var": f"p{index}"}, divisor]}]})
        if len(parameters) > 64:
            continue
        iterator = iter(columns)

        def substitute(v, fixed=False, iterator=iterator):
            if type(v) is int and not fixed:
                return next(iterator)
            if isinstance(v, dict):
                return {key: substitute(v[key], fixed or key in frozen) for key in sorted(v)}
            if isinstance(v, list):
                return [substitute(item, fixed) for item in v]
            return v

        name = f"t{len(templates)}"
        definition = {
            "record": "template",
            "kind": "record",
            "name": name,
            "params": [f"p{i}" for i in range(len(parameters))],
            "body": [substitute(cluster["first"])],
        }
        encoded = json.dumps(definition, separators=(",", ":"))
        if retained + len(encoded.encode()) > 8 * 1024**2:
            continue
        retained += len(encoded.encode())
        vectors = list(parameters)
        args = [[vector[i] for vector in vectors] for i in range(len(rows))]
        templates[signature] = (name, encoded, iter(args))
    with source.open() as reader, destination.open("x") as writer:
        reader.readline()
        writer.write(json.dumps(header, separators=(",", ":")) + "\n")
        for _, encoded, _ in templates.values():
            writer.write(encoded + "\n")
        for line in reader:
            record = json.loads(line)
            signature, _ = structure(record)
            if signature in templates:
                name, _, args = templates[signature]
                record = {"record": "call", "name": name, "args": next(args)}
                writer.write(json.dumps(record, separators=(",", ":")) + "\n")
            else:
                writer.write(line)
    return {
        "source_bytes": source.stat().st_size,
        "compact_bytes": destination.stat().st_size,
        "templates": len(templates),
        "retained_template_bytes": retained,
    }


def compile_compact(source, destination):
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(prefix="phase-two-compile-") as temp:
        root = Path(temp)
        folded = root / "folded.jsonl"
        first = convert(source, folded)
        choices = [(folded.stat().st_size, folded, {"mode": "loops"})]
        for freeze in (False, True):
            path = root / f"templates-{freeze}.jsonl"
            stats = share_templates(folded, path, freeze_shapes=freeze)
            choices.append((path.stat().st_size, path, stats))
        _, chosen, stats = min(choices, key=lambda choice: choice[0])
        with chosen.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        return {"fold": first, "selected": stats, "bytes": destination.stat().st_size}


if __name__ == "__main__":
    main()
