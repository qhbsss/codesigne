import json
import tempfile
import unittest
from pathlib import Path

from model.pipeline_gemm import _pipeline_tile, pipeline_decode_program


def chunk(offset, token):
    return [{"command": "async", "token": token,
             "instruction": {"op": "load", "tensor": 1,
                             "local": {"base": 16, "rows": 1, "cols": 32,
                                       "row_stride": 32, "col_stride": 1},
                             "global": {"base": offset, "rows": 1, "cols": 32,
                                        "row_stride": 32, "col_stride": 1}}},
            {"command": "async", "token": token + 1,
             "instruction": {"op": "load", "tensor": 2,
                             "local": {"base": 48, "rows": 32, "cols": 16,
                                       "row_stride": 16, "col_stride": 1},
                             "global": {"base": offset * 16, "rows": 32, "cols": 16,
                                        "row_stride": 16, "col_stride": 1}}},
            {"command": "wait", "tokens": [token, token + 1]},
            {"command": "run", "instruction": {"op": "mma", "a": 16, "b": 48,
                                                    "c": 0, "m": 1, "n": 16, "k": 32}}]


def tile():
    return [{"command": "run", "instruction": {"op": "fill", "dst": 0,
                                               "len": 16, "value": 0.0}},
            *chunk(0, 1), *chunk(32, 3), *chunk(64, 5),
            {"command": "run", "instruction": {"op": "store", "tensor": 3}}]


class GemmPrefetchTests(unittest.TestCase):
    def test_double_buffers_precede_the_previous_mma(self):
        original = tile()
        transformed, rf_kib, count = _pipeline_tile(original, 256)
        self.assertEqual(count, 3)
        self.assertEqual(rf_kib, 4)
        self.assertEqual([command["command"] for command in transformed[:8]],
                         ["run", "async", "async", "wait", "async", "async", "run", "wait"])
        self.assertEqual([command["instruction"]["a"] for command in transformed
                          if command.get("instruction", {}).get("op") == "mma"],
                         [16, 256, 16])
        self.assertEqual([command["instruction"]["b"] for command in transformed
                          if command.get("instruction", {}).get("op") == "mma"],
                         [48, 288, 48])
        self.assertEqual([command["tokens"] for command in transformed
                          if command["command"] == "wait"], [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(original[5]["instruction"]["local"]["base"], 16)

    def test_rf_capacity_rejects_an_oversubscribed_group(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "output.jsonl"
            source.write_text("\n".join(json.dumps(item) for item in [
                {"decode": True, "hardware": {"sms": 1, "rf_kib": 4,
                                               "sh_kib": 0, "resident_groups": 2}},
                {"record": "wave", "groups": [
                    {"sm": 0, "rf_kib": 2, "sh_kib": 0, "commands": tile()},
                    {"sm": 0, "rf_kib": 2, "sh_kib": 0, "commands": tile()}]}]) + "\n")
            with self.assertRaisesRegex(ValueError, "no capacity-feasible"):
                pipeline_decode_program(source, output)


if __name__ == "__main__":
    unittest.main()
