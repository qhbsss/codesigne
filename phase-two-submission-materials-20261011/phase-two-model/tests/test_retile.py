import json
import tempfile
import unittest
from pathlib import Path

from model.retile import retile_program
from model.tool import OFFICIAL, STARTER, command


class InitialTransposeTests(unittest.TestCase):
    def test_replacement_preserves_each_matrix_address(self):
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads((STARTER / "examples/small-config.json").read_text())
            config["decode"] = True
            config["batch"] = 4
            config["shape"]["history"] = 64
            config["shape"]["steps"] = 2
            config["policy"]["persistent_keys"] = True
            configuration = Path(directory) / "config.json"
            configuration.write_text(json.dumps(config))
            original = Path(directory) / "export"
            command([OFFICIAL, "export", configuration, 7, original],
                    60, Path(directory) / "export-command.json")
            source = original / "program.jsonl"
            rewritten = Path(directory) / "retiled.jsonl"
            statistics = retile_program(source, rewritten, 64)
            self.assertEqual(statistics["initial_transposes"], 1)
            self.assertEqual(statistics["old_tiles"], 16)
            self.assertEqual(statistics["new_waves"], 1)
            variants = []
            for index, candidate in enumerate((source, rewritten)):
                target = Path(directory) / f"evaluation-{index}"
                command([OFFICIAL, "evaluate", candidate, 7, target],
                        60, Path(directory) / f"evaluate-command-{index}.json")
                variants.append(json.loads((target / "result.json").read_text()))
            self.assertTrue(all(result["status"] == "complete" for result in variants))
            self.assertEqual(variants[0]["evaluation"]["output_sha256"],
                             variants[1]["evaluation"]["output_sha256"])

    def test_unrecognized_tile_rejected(self):
        with self.assertRaisesRegex(ValueError, "transpose tile"):
            retile_program("source", "destination", 48)


if __name__ == "__main__":
    unittest.main()
