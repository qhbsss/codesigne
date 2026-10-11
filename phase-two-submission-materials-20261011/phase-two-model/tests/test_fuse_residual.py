import json
import tempfile
import unittest
from pathlib import Path

from model.estimate import estimate_program, program_records
from model.fuse_bias import fuse_bias_program
from model.fuse_residual import fuse_residual_program
from model.tool import OFFICIAL, STARTER, command


class ResidualFusionTests(unittest.TestCase):
    def test_output_tensor_is_not_read_and_stored_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = STARTER / "examples/small-prefill.jsonl"
            bias = root / "bias.jsonl"
            fused = root / "bias-residual.jsonl"
            fuse_bias_program(original, bias)
            statistics = fuse_residual_program(bias, fused)
            self.assertEqual(statistics["stages_fused"], 2)
            self.assertEqual(statistics["output_loads_removed"], 8)
            self.assertLess(estimate_program(fused)["cycles"],
                            estimate_program(bias)["cycles"])
            for name, program in (("original", original), ("fused", fused)):
                command([OFFICIAL, "evaluate", program, 7, root / name], 60,
                        root / f"{name}-command.json")
            before = json.loads((root / "original/result.json").read_text())
            after = json.loads((root / "fused/result.json").read_text())
            self.assertEqual(before["status"], "complete")
            self.assertEqual(after["status"], "complete")
            self.assertEqual(before["evaluation"]["output_sha256"],
                             after["evaluation"]["output_sha256"])
            self.assertTrue(after["evaluation"]["report"]["power_pass"])

    def test_incomplete_residual_stage_is_not_fused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bias = root / "bias.jsonl"
            fuse_bias_program(STARTER / "examples/small-prefill.jsonl", bias)
            records = list(program_records(bias))
            for record in records:
                if record.get("record") != "wave":
                    continue
                for group in record["groups"]:
                    commands = group["commands"]
                    if (len(commands) >= 4 and
                            [command.get("instruction", {}).get("op")
                             for command in commands[:4]] ==
                            ["load", "load", "vector", "store"] and
                            commands[0]["instruction"]["tensor"] !=
                            commands[3]["instruction"]["tensor"] and
                            commands[1]["instruction"]["tensor"] ==
                            commands[3]["instruction"]["tensor"]):
                        commands[3]["instruction"]["global"]["base"] += 1
                        break
                else:
                    continue
                break
            corrupted = root / "corrupted.jsonl"
            corrupted.write_text("".join(json.dumps(record) + "\n" for record in records))
            output = root / "partially-fused.jsonl"
            statistics = fuse_residual_program(corrupted, output)
            self.assertEqual(statistics["stages_fused"], 1)


if __name__ == "__main__":
    unittest.main()
