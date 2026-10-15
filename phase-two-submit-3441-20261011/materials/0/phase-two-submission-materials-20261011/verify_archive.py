#!/usr/bin/env python3
"""Check a Phase Two archive's five-program local self-test and ZIP contract."""

import hashlib
import json
import math
import stat
import sys
import zipfile
from pathlib import PurePosixPath


CASES = ("w-p", "a-p", "w-d1", "w-d16", "w-d4l")
MATERIALS = "materials/0/phase-two-submission-materials-20261011/"


def read_json(archive, name):
    return json.loads(archive.read(name))


def check(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        assert len(entries) <= 10000 and len(names) == len(set(names))
        assert sum(entry.file_size for entry in entries) <= 1024**3
        for entry in entries:
            parts = PurePosixPath(entry.filename).parts
            assert parts and all(part not in ("", ".", "..") for part in parts)
            assert not entry.filename.startswith("/") and not (entry.flag_bits & 1)
            assert not stat.S_ISLNK(entry.external_attr >> 16)
        assert archive.testzip() is None

        grade = read_json(archive, "selftest/grade.json")
        contract = read_json(archive, "selftest/contract/result.json")
        summary = read_json(archive, "selftest/workers/summary.json")
        screen = read_json(archive, MATERIALS + "selection/screen.json")
        hardware = read_json(archive, MATERIALS + "hardware.json")
        assert grade["status"] == summary["status"] == "complete"
        assert summary["all_checked"] and summary["all_feasible"]
        assert set(grade["cases"]) == set(screen["program_sha256"]) == set(CASES)
        assert grade["hardware"] == hardware
        assert grade["contract"] == contract["contract"]
        assert grade["model"] == contract["model"]
        assert grade["source_sha256"] == contract["source_sha256"]

        weighted_log = 0.0
        for case in CASES:
            program = archive.read(f"programs/{case}.jsonl")
            digest = hashlib.sha256(program).hexdigest()
            result = read_json(archive, f"selftest/workers/{case}/evaluation/result.json")
            evaluation = result["evaluation"]
            receipt = grade["cases"][case]
            assert result["status"] == "complete" and result["error"] is None
            assert result["source_sha256"] == grade["source_sha256"]
            assert evaluation["program_sha256"] == receipt["program_sha256"]
            assert digest == receipt["program_sha256"] == screen["program_sha256"][case]
            assert evaluation["header"]["hardware"] == hardware
            assert evaluation["report"]["power_pass"]
            cycles = evaluation["report"]["stats"]["cycles"]
            assert cycles == receipt["cycles"] and cycles > 0
            reference = contract["evaluation"]["cases"][case]
            for key in ("shape", "batch", "decode"):
                assert evaluation["header"][key] == reference[key]
            weighted_log += reference["weight"] * math.log(
                reference["reference_cycles"] / cycles
            )
        assert math.isclose(grade["score"], 1000 * math.exp(weighted_log), rel_tol=1e-12)
        print(f"Archive verified: {len(entries)} files, seed {grade['seed']}, "
              f"five cases, local score {grade['score']:.6f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 verify_archive.py submission.zip")
    check(sys.argv[1])
