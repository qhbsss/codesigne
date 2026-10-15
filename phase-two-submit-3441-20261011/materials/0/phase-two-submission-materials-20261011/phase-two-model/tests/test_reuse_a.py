import copy
import json
import tempfile
import unittest
from pathlib import Path

from model.estimate import program_records
from model.reuse_a import reuse_a_program
from model.tool import STARTER
from tests.test_reuse_b import tile


class ReuseATests(unittest.TestCase):
    def test_pair_shares_activation_without_exceeding_live_token_limit(self):
        first = tile(0)
        second = copy.deepcopy(first)
        first["commands"][-1]["instruction"]["global"]["row_stride"] = 32
        second["commands"][2]["instruction"]["global"]["base"] = 16
        second["commands"][-1]["instruction"]["global"].update(base=16,
                                                                     row_stride=32)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            destination = Path(directory) / "output.jsonl"
            with (STARTER / "examples/small-prefill.jsonl").open() as original:
                header = json.loads(next(original))
            source.write_text("\n".join(json.dumps(record) for record in (
                header, {"record": "wave", "groups": [first]},
                {"record": "wave", "groups": [second]})) + "\n")
            statistics = reuse_a_program(source, destination)
            self.assertEqual(statistics["paired_groups"], 1)
            self.assertEqual(statistics["removed_a_loads"], 1)
            waves = [record for record in program_records(destination)
                     if record.get("record") == "wave"]
            self.assertEqual(len(waves), 1)
            group = waves[0]["groups"][0]
            self.assertEqual(group["rf_kib"], 4)
            commands = group["commands"]
            self.assertEqual([command["tokens"] for command in commands
                              if command["command"] == "wait"], [[1, 2], [3]])
            self.assertEqual([command["instruction"]["c"] for command in commands
                              if command.get("instruction", {}).get("op") == "mma"],
                             [0, 512])
            self.assertEqual(sum(command.get("instruction", {}).get("tensor") == 1
                                 for command in commands), 1)
            self.assertEqual(sum(command.get("instruction", {}).get("tensor") == 2
                                 for command in commands), 2)


if __name__ == "__main__":
    unittest.main()
