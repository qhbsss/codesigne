import json
import tempfile
import unittest
from pathlib import Path

from model.estimate import program_records
from model.reuse_b import reuse_b_program
from model.tool import STARTER


def tile(row):
    activation = {"op": "load", "tensor": 1,
                  "global": {"base": row * 32, "rows": 2, "cols": 16,
                             "row_stride": 16, "col_stride": 1},
                  "local": {"base": 32, "rows": 2, "cols": 16,
                            "row_stride": 16, "col_stride": 1}}
    panel = {"op": "load", "tensor": 2,
             "global": {"base": 0, "rows": 16, "cols": 16,
                        "row_stride": 16, "col_stride": 1},
             "local": {"base": 64, "rows": 16, "cols": 16,
                       "row_stride": 16, "col_stride": 1}}
    output = {"op": "store", "tensor": 3,
              "global": {"base": row * 32, "rows": 2, "cols": 16,
                         "row_stride": 16, "col_stride": 1},
              "local": {"base": 0, "rows": 2, "cols": 16,
                        "row_stride": 16, "col_stride": 1}}
    return {"sm": 0, "rf_kib": 2, "sh_kib": 0, "commands": [
        {"command": "run", "instruction": {"op": "fill", "dst": 0,
                                          "len": 32, "value": 0.0}},
        {"command": "async", "token": 1, "instruction": activation},
        {"command": "async", "token": 2, "instruction": panel},
        {"command": "wait", "tokens": [1, 2]},
        {"command": "run", "instruction": {"op": "mma", "m": 2, "n": 16,
                                          "k": 16, "a": 32, "b": 64, "c": 0}},
        {"command": "run", "instruction": output}]}


class ReuseBTests(unittest.TestCase):
    def transform(self, records):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            destination = Path(directory) / "output.jsonl"
            with (STARTER / "examples/small-prefill.jsonl").open() as example:
                header = json.loads(next(example))
            source.write_text("\n".join(json.dumps(record) for record in
                                         [header, *records]) + "\n")
            stats = reuse_b_program(source, destination)
            return stats, list(program_records(destination))[1:]

    def test_shared_panel_loads_once_for_independent_tiles(self):
        stats, records = self.transform([{"record": "wave", "groups": [tile(0)]},
                                         {"record": "wave", "groups": [tile(1)]}])
        self.assertEqual(stats["paired_groups"], 1)
        self.assertEqual(stats["removed_b_loads"], 1)
        self.assertEqual(stats["output_waves"], 1)
        commands = records[0]["groups"][0]["commands"]
        self.assertEqual(sum(command.get("instruction", {}).get("tensor") == 2
                             for command in commands), 1)
        self.assertEqual([command["instruction"]["c"] for command in commands
                          if command.get("instruction", {}).get("op") == "mma"], [0, 512])
        self.assertEqual(len({command["token"] for command in commands
                              if command["command"] == "async"}), 3)
        self.assertEqual([command["tokens"] for command in commands
                          if command["command"] == "wait"], [[1, 2], [3]])

    def test_conflicts_and_non_wave_records_prevent_reordering(self):
        first = {"record": "wave", "groups": [tile(0)]}
        overlapping = {"record": "wave", "groups": [tile(0)]}
        stats, records = self.transform([first, overlapping])
        self.assertEqual(stats["paired_groups"], 0)
        self.assertEqual(len(records), 2)
        stats, records = self.transform([first, {"record": "commit", "outputs": []},
                                         {"record": "wave", "groups": [tile(1)]}])
        self.assertEqual(stats["paired_groups"], 0)
        self.assertEqual(len(records), 3)


if __name__ == "__main__":
    unittest.main()
