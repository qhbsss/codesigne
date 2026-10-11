import json
import tempfile
import unittest
from pathlib import Path

from model.estimate import estimate_program, program_records
from model.fuse_bias import fuse_bias_program
from model.fuse_gelu import fuse_gelu_program
from model.tool import OFFICIAL, STARTER, command


class GeluFusionTests(unittest.TestCase):
    def test_up_projection_and_gelu_share_rf_without_output_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = STARTER / "examples/small-prefill.jsonl"
            bias = root / "bias.jsonl"
            fused = root / "bias-gelu.jsonl"
            fuse_bias_program(original, bias)
            statistics = fuse_gelu_program(bias, fused)
            self.assertEqual(statistics["stages_fused"], 1)
            self.assertEqual(statistics["gelu_elements_relocated"], 128)
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

    def test_mutated_gelu_view_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bias = root / "bias.jsonl"
            fuse_bias_program(STARTER / "examples/small-prefill.jsonl", bias)
            records = list(program_records(bias))
            changed = False
            for record in records:
                if record.get("record") != "wave":
                    continue
                for group in record["groups"]:
                    if not any(command.get("instruction", {}).get("kind") == "tanh"
                               for command in group["commands"]):
                        continue
                    store = next(command["instruction"] for command in group["commands"]
                                 if command.get("instruction", {}).get("op") == "store")
                    store["global"]["base"] += 1
                    changed = True
                    break
                if changed:
                    break
            self.assertTrue(changed)
            corrupted = root / "corrupted.jsonl"
            corrupted.write_text("".join(json.dumps(record) + "\n" for record in records))
            with self.assertRaisesRegex(ValueError, "no complete bias-fused GEMM"):
                fuse_gelu_program(corrupted, root / "output.jsonl")


if __name__ == "__main__":
    unittest.main()
