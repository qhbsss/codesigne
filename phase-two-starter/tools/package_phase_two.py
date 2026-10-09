#!/usr/bin/env python3
"""Package an official five-case self-test and optional materials without build caches."""

import argparse
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("selftest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--materials", type=Path, action="append", default=[])
    args = p.parse_args()
    files = []
    for name in ("grade.json", "contract/result.json", "workers/summary.json"):
        files.append((args.selftest / name, "selftest/" + name))
    for case in ("w-p", "a-p", "w-d1", "w-d16", "w-d4l"):
        files.append((args.selftest / "programs" / f"{case}.jsonl", f"programs/{case}.jsonl"))
        name = f"workers/{case}/evaluation/result.json"
        files.append((args.selftest / name, "selftest/" + name))
    for i, material in enumerate(args.materials):
        candidates = sorted(material.rglob("*")) if material.is_dir() else [material]
        for file in candidates:
            relative = file.relative_to(material) if material.is_dir() else Path(file.name)
            if any(
                part in {".git", "target", "__pycache__", ".venv", "node_modules"}
                for part in relative.parts
            ):
                continue
            if file.is_symlink():
                raise ValueError(f"Symlink not supported: {file}")
            if file.is_file():
                files.append((file, f"materials/{i}/{material.name}/" + relative.as_posix()))
    if any(not file.is_file() for file, _ in files):
        raise ValueError("Missing self-test files; use output from tools/grade_v09.py")
    if len(files) > 10000 or sum(file.stat().st_size for file, _ in files) > 1024**3:
        raise ValueError("At most 10000 files / 1 GiB expanded")
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file, name in files:
            z.write(file, name)
    if args.output.stat().st_size > 256 * 1024**2:
        raise ValueError("ZIP exceeds 256 MiB; remove rebuildable/redundant material")
    print(f"{args.output}: {args.output.stat().st_size} bytes, {len(files)} files")


if __name__ == "__main__":
    main()
