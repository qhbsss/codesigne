import copy
import json
import tempfile
import unittest
from pathlib import Path

from model.estimate import estimate_program, program_records
from model.fuse_bias import _bias_wave, fuse_bias_program
from model.fuse_residual import _residual_wave, fuse_residual_program
from model.split_elementwise import _split_wave, split_elementwise_program
from model.tool import OFFICIAL, STARTER, command


class SplitElementwiseTests(unittest.TestCase):
    def test_restores_bias_residual_barriers_without_changing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = STARTER / "examples/small-prefill.jsonl"
            records = list(program_records(original))
            merged = []
            index = 0
            while index < len(records):
                if (index + 1 < len(records) and
                        (bias := _bias_wave(records[index])) is not None and
                        (residual := _residual_wave(records[index + 1])) is not None and
                        bias[0] == residual[0] and bias[2] == residual[2]):
                    first, second = records[index:index + 2]
                    self.assertEqual([(group["sm"], group["rf_kib"], group["sh_kib"])
                                      for group in first["groups"]],
                                     [(group["sm"], group["rf_kib"], group["sh_kib"])
                                      for group in second["groups"]])
                    compound = copy.deepcopy(first)
                    for group, added in zip(compound["groups"], second["groups"]):
                        group["commands"].extend(added["commands"])
                    merged.append(compound)
                    index += 2
                else:
                    merged.append(records[index])
                    index += 1
            source = root / "compound.jsonl"
            source.write_text("".join(json.dumps(record) + "\n" for record in merged))
            restored = root / "restored.jsonl"
            stats = split_elementwise_program(source, restored)
            self.assertEqual(stats["compound_waves_split"], 2)
            self.assertEqual(estimate_program(original)["cycles"],
                             estimate_program(restored)["cycles"])
            bias = root / "bias.jsonl"
            fused = root / "fused.jsonl"
            self.assertEqual(fuse_bias_program(restored, bias)["stages_fused"], 4)
            self.assertEqual(fuse_residual_program(bias, fused)["stages_fused"], 2)
            for name, program in (("original", original), ("compound", source),
                                  ("restored", restored), ("fused", fused)):
                command([OFFICIAL, "evaluate", program, 7, root / name], 60,
                        root / f"{name}-command.json")
            digests = [json.loads((root / name / "result.json").read_text())
                       ["evaluation"]["output_sha256"]
                       for name in ("original", "compound", "restored", "fused")]
            self.assertEqual(digests, [digests[0]] * 4)

    def test_rejects_incompatible_second_pass(self):
        records = list(program_records(STARTER / "examples/small-prefill.jsonl"))
        first = copy.deepcopy(records[41])
        second = copy.deepcopy(records[42])
        self.assertIsNotNone(_bias_wave(first))
        self.assertIsNotNone(_residual_wave(second))
        second["groups"][0]["commands"][-1]["instruction"]["global"]["base"] += 1
        for group, added in zip(first["groups"], second["groups"]):
            group["commands"].extend(added["commands"])
        self.assertIsNone(_split_wave(first, records[0]["hardware"]))

    def test_async_compound_pass_restores_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = STARTER / "examples/small-prefill.jsonl"
            records = list(program_records(original))
            compound = copy.deepcopy(records[41])
            for group, residual in zip(compound["groups"], records[42]["groups"]):
                result = []
                for stage_index, commands in enumerate(
                        (group["commands"], residual["commands"])):
                    first, second, add, store = copy.deepcopy(commands)
                    first["command"] = second["command"] = "async"
                    first["token"], second["token"] = stage_index * 2 + 1, stage_index * 2 + 2
                    result.extend((first, second, {"command": "wait", "tokens": [
                        first["token"], second["token"]]}, add, store))
                group["commands"] = result
            pieces = _split_wave(compound, records[0]["hardware"])
            self.assertEqual(len(pieces), 2)
            self.assertIsNotNone(_bias_wave(pieces[0]))
            self.assertIsNotNone(_residual_wave(pieces[1]))
            merged = root / "compound.jsonl"
            merged.write_text("".join(json.dumps(record) + "\n" for record in
                                      [*records[:41], compound, *records[43:]]))
            restored = root / "restored.jsonl"
            self.assertEqual(split_elementwise_program(merged, restored)
                             ["compound_waves_split"], 1)
            reports = []
            for name, program in (("original", original), ("compound", merged),
                                  ("restored", restored)):
                command([OFFICIAL, "evaluate", program, 7, root / name], 60,
                        root / f"{name}-command.json")
                reports.append(json.loads((root / name / "result.json").read_text()))
            self.assertEqual(len({report["evaluation"]["output_sha256"]
                                  for report in reports}), 1)
            self.assertTrue(all(report["evaluation"]["report"]["power_pass"]
                                for report in reports))


if __name__ == "__main__":
    unittest.main()
