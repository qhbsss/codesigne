import copy
import json
import tempfile
import unittest
from pathlib import Path

from model.estimate import program_records
from model.merge import merge_program
from model.reuse_b_within import _group, reuse_b_within_program
from model.tool import OFFICIAL, ROOT, STARTER, command


def tile(row):
    view = lambda base, rows, cols: {
        "base": base, "rows": rows, "cols": cols,
        "row_stride": cols, "col_stride": 1}
    return [
        {"command": "run", "instruction": {
            "op": "fill", "dst": 0, "len": 32, "value": 0.0}},
        {"command": "async", "token": 1, "instruction": {
            "op": "load", "tensor": 1, "global": view(row * 32, 2, 16),
            "local": view(32, 2, 16)}},
        {"command": "async", "token": 2, "instruction": {
            "op": "load", "tensor": 2, "global": view(0, 16, 16),
            "local": view(64, 16, 16)}},
        {"command": "wait", "tokens": [1, 2]},
        {"command": "run", "instruction": {
            "op": "mma", "m": 2, "n": 16, "k": 16, "a": 32, "b": 64, "c": 0}},
        {"command": "run", "instruction": {
            "op": "store", "tensor": 3, "global": view(row * 32, 2, 16),
            "local": view(0, 2, 16)}}]


class ReuseBWithinTests(unittest.TestCase):
    def test_reuses_panel_for_adjacent_output_tiles(self):
        group = {"sm": 0, "rf_kib": 2, "sh_kib": 0,
                 "commands": tile(0) + tile(1) + tile(2)}
        transformed, pairs, removed = _group(group)
        self.assertEqual((pairs, removed), (1, 1))
        self.assertEqual(transformed["rf_kib"], 3)
        self.assertEqual(len([command for command in transformed["commands"]
                              if command.get("instruction", {}).get("tensor") == 2]), 2)
        self.assertEqual([command["instruction"]["c"]
                          for command in transformed["commands"]
                          if command.get("instruction", {}).get("op") == "mma"],
                         [0, 512, 0])
        tokens = [command["token"] for command in transformed["commands"]
                  if command["command"] == "async"]
        self.assertEqual(tokens, list(range(1, len(tokens) + 1)))
        self.assertEqual(group["commands"], tile(0) + tile(1) + tile(2))

    def test_rejects_overlapping_or_different_panels(self):
        group = {"sm": 0, "rf_kib": 2, "sh_kib": 0,
                 "commands": tile(0) + tile(0)}
        self.assertEqual(_group(group)[1:], (0, 0))
        changed = copy.deepcopy(tile(1))
        changed[2]["instruction"]["global"]["base"] = 16
        group["commands"] = tile(0) + changed
        self.assertEqual(_group(group)[1:], (0, 0))

    def test_program_keeps_other_records(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            destination = Path(directory) / "output.jsonl"
            with (STARTER / "examples/small-prefill.jsonl").open() as original:
                header = json.loads(next(original))
            wave = {"record": "wave", "groups": [
                {"sm": 0, "rf_kib": 2, "sh_kib": 0,
                 "commands": tile(0) + tile(1)}]}
            source.write_text("".join(json.dumps(record) + "\n" for record in
                                      [header, {"record": "alloc", "len": 32}, wave]))
            stats = reuse_b_within_program(source, destination)
            self.assertEqual(stats["paired_tiles"], 1)
            self.assertEqual(stats["removed_b_loads"], 1)
            records = list(program_records(destination))
            self.assertEqual(records[1], {"record": "alloc", "len": 32})

    def test_educational_program_preserves_numerical_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = ROOT / "phase-two-model/specs/reuse-b-within-educational.json"
            command([OFFICIAL, "export", config, 7, root / "export"], 60,
                    root / "export-command.json")
            original = root / "export/program.jsonl"
            merged = root / "merged.jsonl"
            reused = root / "reused.jsonl"
            merge_program(original, merged, 64, "concat")
            stats = reuse_b_within_program(merged, reused)
            self.assertGreater(stats["paired_tiles"], 0)
            reports = []
            for name, program in (("original", original), ("merged", merged),
                                  ("reused", reused)):
                command([OFFICIAL, "evaluate", program, 7, root / name], 60,
                        root / f"{name}-command.json")
                reports.append(json.loads((root / name / "result.json").read_text()))
            self.assertTrue(all(report["status"] == "complete" and
                                report["evaluation"]["report"]["power_pass"]
                                for report in reports))
            self.assertEqual(len({report["evaluation"]["output_sha256"]
                                  for report in reports}), 1)
            self.assertLess(reports[2]["evaluation"]["report"]["stats"]["cycles"],
                            reports[1]["evaluation"]["report"]["stats"]["cycles"])


if __name__ == "__main__":
    unittest.main()
