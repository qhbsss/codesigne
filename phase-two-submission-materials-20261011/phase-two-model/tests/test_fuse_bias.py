import json
import tempfile
import unittest
from pathlib import Path

from model.estimate import estimate_program, program_records
from model.fuse_bias import _gemm_wave, fuse_bias_program
from model.tool import OFFICIAL, STARTER, command


class BiasFusionTests(unittest.TestCase):
    def setUp(self):
        self.source = STARTER / "examples/small-prefill.jsonl"

    def test_complete_gemm_bias_pass_removes_all_bias_waves(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fused.jsonl"
            statistics = fuse_bias_program(self.source, output)
            self.assertEqual(statistics["stages_fused"], 4)
            self.assertEqual(statistics["bias_waves_removed"], 4)
            self.assertEqual(statistics["output_tensor_stores_removed"], 16)
            self.assertEqual(estimate_program(output)["waves"], 20)
            self.assertLess(estimate_program(output)["cycles"],
                            estimate_program(self.source)["cycles"])

    def test_incomplete_gemm_coverage_is_not_fused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = list(program_records(self.source))
            gemm = next(record for record in records if _gemm_wave(record) is not None)
            store = next(command["instruction"] for group in gemm["groups"]
                         for command in group["commands"]
                         if command.get("instruction", {}).get("op") == "store")
            store["global"]["base"] += 1
            corrupted = root / "corrupted.jsonl"
            corrupted.write_text("".join(json.dumps(record) + "\n" for record in records))
            output = root / "partially-fused.jsonl"
            statistics = fuse_bias_program(corrupted, output)
            self.assertEqual(statistics["stages_fused"], 3)
            original_gemm = next(record for record in program_records(corrupted)
                                 if _gemm_wave(record) is not None)
            rewritten_gemm = next(record for record in program_records(output)
                                  if _gemm_wave(record) is not None)
            self.assertEqual(original_gemm, rewritten_gemm)

    def test_public_seed_output_matches_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fused = root / "fused.jsonl"
            fuse_bias_program(self.source, fused)
            for name, program in (("original", self.source), ("fused", fused)):
                command([OFFICIAL, "evaluate", program, 7, root / name], 60,
                        root / f"{name}-command.json")
            original = json.loads((root / "original/result.json").read_text())
            updated = json.loads((root / "fused/result.json").read_text())
            self.assertEqual(original["status"], "complete")
            self.assertEqual(updated["status"], "complete")
            self.assertEqual(original["evaluation"]["output_sha256"],
                             updated["evaluation"]["output_sha256"])
            self.assertTrue(updated["evaluation"]["report"]["power_pass"])


if __name__ == "__main__":
    unittest.main()
