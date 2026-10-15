import unittest

from model.causal import _prune_av, _prune_qk, _trim_mask


def view(base, rows=1, cols=16, row_stride=256):
    return {"base": base, "rows": rows, "cols": cols,
            "row_stride": row_stride, "col_stride": 1}


def command(operation, **arguments):
    return {"command": "run", "instruction": {"op": operation, **arguments}}


class CausalPruningTests(unittest.TestCase):
    def setUp(self):
        self.stats = {"mask_words_removed": 0, "mask_pairs_removed": 0,
                      "qk_tiles_removed": 0, "qk_commands_removed": 0,
                      "av_chunks_removed": 0, "av_commands_removed": 0}

    def test_mask_preserves_only_the_attention_row_tiles_read(self):
        fill = command("fill", dst=0, len=255, value=0.0)
        store = command("store", tensor=10, local=view(0, cols=255, row_stride=255),
                        **{"global": view(1, cols=255)})
        result = _trim_mask([fill, store], {10}, 256, 32, 128, self.stats)
        self.assertEqual(result[0]["instruction"]["len"], 127)
        self.assertEqual(result[1]["instruction"]["global"]["cols"], 127)
        self.assertEqual(fill["instruction"]["len"], 255)
        last = command("store", tensor=10, local=view(0, cols=128, row_stride=128),
                       **{"global": view(127 * 256 + 128, cols=128)})
        self.assertEqual(_trim_mask([command("fill", dst=0, len=128, value=0.0),
                                     last], {10}, 256, 32, 128, self.stats), [])

    def test_qk_discards_only_wholly_future_score_tiles(self):
        fill = command("fill", dst=0, len=2048, value=0.0)
        score = command("store", tensor=10,
                        **{"global": view(128, rows=32, cols=64)})
        self.assertEqual(_prune_qk([fill, score], {10}, 256, 32, 128, self.stats), [])
        self.assertEqual(_prune_qk([fill, command("store", tensor=10,
            **{"global": view(64, rows=32, cols=64)})], {10}, 256, 32, 128, self.stats)[0], fill)

    def test_av_retains_first_chunk_and_drops_unread_future_chunk(self):
        def chunk(base, token):
            first = {"command": "async", "token": token,
                     "instruction": {"op": "load", "tensor": 10,
                                     "global": view(base, rows=32, cols=128)}}
            second = {"command": "async", "token": token + 1,
                      "instruction": {"op": "load", "tensor": 11,
                                      "global": view(base, rows=128, cols=64)}}
            return [first, second, {"command": "wait", "tokens": [token, token + 1]},
                    command("mma", m=32, n=64, k=128)]

        tile = [command("fill", dst=0, len=2048, value=0.0),
                *chunk(0, 1), *chunk(128, 3),
                command("store", tensor=12,
                        **{"global": view(0, rows=32, cols=64)})]
        result = _prune_av(tile, {10}, 256, 32, 128, self.stats)
        self.assertEqual(len(result), 6)
        self.assertEqual(self.stats["av_chunks_removed"], 1)
        self.assertEqual(result[-1], tile[-1])
        with self.assertRaisesRegex(ValueError, "score view"):
            _prune_av(tile, {10}, 256, 16, 128, self.stats)


if __name__ == "__main__":
    unittest.main()
