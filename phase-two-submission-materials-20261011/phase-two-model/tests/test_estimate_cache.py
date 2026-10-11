import json
import unittest

from model.estimate import _compute_cycles
from model.tool import STARTER


class ComputeCacheTests(unittest.TestCase):
    def setUp(self):
        with (STARTER / "examples/small-prefill.jsonl").open() as source:
            self.hardware = json.loads(next(source))["hardware"]

    def test_mma_register_offsets_do_not_change_latency(self):
        cache = {}
        first = {"op": "mma", "a": 64, "b": 128, "c": 0,
                 "m": 4, "n": 16, "k": 16}
        second = dict(first, a=256, b=320, c=192)
        self.assertEqual(_compute_cycles(first, self.hardware, cache),
                         _compute_cycles(second, self.hardware, cache))
        self.assertEqual(len(cache), 1)

    def test_vector_immediate_values_do_not_change_latency(self):
        cache = {}
        first = {"op": "vector", "kind": "mul", "len": 64,
                 "a": {"kind": "reg", "offset": 0, "stride": 1},
                 "b": {"kind": "imm", "value": 2.0}}
        second = dict(first, a={"kind": "reg", "offset": 128, "stride": 1},
                      b={"kind": "imm", "value": 3.0})
        self.assertEqual(_compute_cycles(first, self.hardware, cache),
                         _compute_cycles(second, self.hardware, cache))
        self.assertEqual(len(cache), 1)

    def test_shared_bank_views_remain_distinct(self):
        cache = {}
        first = {"op": "shared_read", "shared": {"base": 0, "rows": 1,
                 "cols": 16, "row_stride": 16, "col_stride": 1},
                 "local": {"base": 0, "rows": 1, "cols": 16,
                           "row_stride": 16, "col_stride": 1}}
        second = dict(first, shared=dict(first["shared"], base=1))
        _compute_cycles(first, self.hardware, cache)
        _compute_cycles(second, self.hardware, cache)
        self.assertEqual(len(cache), 2)


if __name__ == "__main__":
    unittest.main()
