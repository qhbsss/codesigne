"""Attribute independent-model cycles to paid wave outputs and operation types."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .estimate import estimate_wave, program_records, validate_wave_capacity


def stage_profile(source):
    digest = hashlib.sha256()
    with Path(source).open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
    records = program_records(source)
    header = next(records)
    hardware = header["hardware"]
    tensors = []
    cache = {}
    stages = defaultdict(lambda: {"cycles": 0, "waves": 0, "operations": Counter()})
    commits = 0
    for record in records:
        record_type = record["record"]
        if record_type in ("input", "alloc"):
            tensors.append(record.get("source", f"scratch:{record['len']}"))
            continue
        if record_type == "release":
            if record["tensor"] != len(tensors) - 1:
                raise ValueError("tensor release order changed")
            tensors.pop()
            continue
        if record_type == "commit":
            commits += 1
            continue
        if record_type != "wave":
            raise ValueError(f"unsupported record: {record_type}")
        groups = record["groups"]
        validate_wave_capacity(groups, hardware)
        ops = Counter()
        outputs = set()
        for group in groups:
            for command in group["commands"]:
                instruction = command.get("instruction", {})
                operation = instruction.get("op")
                if operation is None:
                    continue
                ops[operation] += 1
                if operation in ("store", "store_shared"):
                    outputs.add(tensors[instruction["tensor"]])
        family = ("mma" if ops["mma"] or ops["mma_shared"] else
                  "elementwise" if ops["vector"] or ops["reduce"] else "dma")
        label = f"{family}: {','.join(sorted(outputs)) or 'no-output'}"
        entry = stages[label]
        entry["cycles"] += estimate_wave(groups, hardware, cache)
        entry["waves"] += 1
        entry["operations"].update(ops)
    ordered = dict(sorted(stages.items(), key=lambda item: -item[1]["cycles"]))
    return {"cycles": commits + sum(entry["cycles"] for entry in ordered.values()),
            "commits": commits, "waves": sum(entry["waves"] for entry in ordered.values()),
            "program_sha256": digest.hexdigest(), "independent": True,
            "not_an_official_score": True,
            "stages": ordered}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.write_text(json.dumps(stage_profile(args.source), indent=2) + "\n")


if __name__ == "__main__":
    main()
