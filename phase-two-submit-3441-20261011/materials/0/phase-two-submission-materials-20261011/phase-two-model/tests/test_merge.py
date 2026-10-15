import json
import tempfile
import unittest
from pathlib import Path

from model.estimate import program_records
from model.merge import merge_program


def wave(base, token=0, reader="source", writer="output"):
    view = {"base": base, "rows": 1, "cols": 16,
            "row_stride": 16, "col_stride": 1}
    return {"record": "wave", "groups": [{"sm": 0, "rf_kib": 4,
        "sh_kib": 0, "commands": [
            {"command": "async", "token": token,
             "instruction": {"op": "load", "tensor": reader, "global": view}},
            {"command": "wait", "tokens": [token]},
            {"command": "run", "instruction": {"op": "store", "tensor": writer,
                                            "global": view}}]}]}


class WaveFusionTests(unittest.TestCase):
    def merge(self, waves, factor=16, mode="concat", hardware=None):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            destination = Path(directory) / "merged.jsonl"
            source.write_text("".join(json.dumps(record) + "\n" for record in
                                      [{"hardware": hardware or {"sms": 1}}, *waves]))
            stats = merge_program(source, destination, factor, mode)
            records = list(program_records(destination))
        return stats, records[1:]

    def test_disjoint_writes_fuse_with_unique_async_tokens(self):
        stats, records = self.merge([wave(0), wave(16), wave(32)])
        self.assertEqual(stats["output_waves"], 1)
        commands = records[0]["groups"][0]["commands"]
        self.assertEqual([command["token"] for command in commands
                          if command["command"] == "async"], [0, 1, 2])
        self.assertEqual([command["tokens"] for command in commands
                          if command["command"] == "wait"], [[0], [1], [2]])

    def test_overlapping_writes_or_cross_wave_reads_do_not_fuse(self):
        for later in [wave(0), wave(16, reader="output", writer="next")]:
            with self.subTest(later=later):
                stats, _ = self.merge([wave(0), later])
                self.assertEqual(stats["output_waves"], 2)

    def test_non_wave_record_and_factor_stop_fusion(self):
        waves = [wave(0), wave(16), {"record": "commit"}, wave(32), wave(48)]
        stats, records = self.merge(waves, factor=2)
        self.assertEqual(stats["output_waves"], 2)
        self.assertEqual([record["record"] for record in records],
                         ["wave", "commit", "wave"])

    def test_larger_merge_factor_obeys_access_limits(self):
        stats, records = self.merge([wave(index * 16) for index in range(70)],
                                    factor=256)
        self.assertEqual(stats["output_waves"], 1)
        self.assertEqual(len(records[0]["groups"][0]["commands"]), 210)
        with self.assertRaisesRegex(ValueError, "1..512"):
            self.merge([wave(0)], factor=513)

    def test_in_place_reads_require_disjoint_regions(self):
        left = wave(0, reader="output")
        disjoint = wave(16, reader="output")
        overlapping = wave(0, reader="output")
        stats, _ = self.merge([left, disjoint])
        self.assertEqual(stats["output_waves"], 1)
        stats, _ = self.merge([left, overlapping])
        self.assertEqual(stats["output_waves"], 2)

    def test_compute_waves_require_same_operation_set(self):
        left = wave(0)
        right = wave(16)
        for record in (left, right):
            record["groups"][0]["commands"].insert(0, {"command": "run",
                "instruction": {"op": "fill", "dst": 0, "len": 16, "value": 0.0}})
        stats, _ = self.merge([left, right])
        self.assertEqual(stats["fused_other_waves"], 1)

    def test_pack_needs_enough_resident_group_capacity(self):
        hardware = {"sms": 1, "rf_kib": 16, "sh_kib": 0, "resident_groups": 2}
        stats, records = self.merge([wave(0), wave(16), wave(32)],
                                    mode="pack", hardware=hardware)
        self.assertEqual(stats["output_waves"], 2)
        self.assertEqual([len(record["groups"]) for record in records], [2, 1])
        self.assertEqual([command["token"] for group in records[0]["groups"]
                          for command in group["commands"]
                          if command["command"] == "async"], [0, 0])

    def test_pipeline_allows_ordered_same_group_dependencies_only(self):
        writer = wave(0)
        reader = wave(0, reader="output", writer="next")
        stats, _ = self.merge([writer, reader], mode="pipeline")
        self.assertEqual(stats["fused_pipeline_waves"], 1)
        writer["groups"].append(wave(16)["groups"][0])
        reader["groups"].append(wave(16, reader="output", writer="next")["groups"][0])
        stats, _ = self.merge([writer, reader], mode="pipeline")
        self.assertEqual(stats["output_waves"], 1)
        reader["groups"][1]["commands"][0]["instruction"]["global"]["base"] = 0
        stats, _ = self.merge([writer, reader], mode="pipeline")
        self.assertEqual(stats["output_waves"], 2)

    def test_elastic_fuses_disjoint_stores_with_different_rf_reservations(self):
        def clearing(base, reservation):
            return {"record": "wave", "groups": [{"sm": 0, "rf_kib": reservation,
                "sh_kib": 0, "commands": [
                    {"command": "run", "instruction": {"op": "fill", "dst": 0,
                        "len": 16, "value": 0.0}},
                    {"command": "run", "instruction": {"op": "store", "tensor": "score",
                        "global": {"base": base, "rows": 1, "cols": 16,
                                   "row_stride": 16, "col_stride": 1}}}]}]}

        hardware = {"sms": 1, "rf_kib": 8, "sh_kib": 0, "resident_groups": 1}
        stats, records = self.merge([clearing(0, 2), clearing(16, 4)],
                                    mode="elastic", hardware=hardware)
        self.assertEqual(stats["fused_elastic_waves"], 1)
        self.assertEqual(records[0]["groups"][0]["rf_kib"], 4)
        self.assertEqual(len(records[0]["groups"][0]["commands"]), 4)
        stats, _ = self.merge([clearing(0, 2), clearing(0, 4)],
                              mode="elastic", hardware=hardware)
        self.assertEqual(stats["output_waves"], 1)
        stats, records = self.merge([clearing(0, 2), clearing(16, 9)],
                                    mode="elastic", hardware=hardware)
        self.assertEqual(stats["output_waves"], 2)
        self.assertEqual(records[0]["groups"][0]["rf_kib"], 2)

    def test_elastic_checks_dma_cross_group_hazards(self):
        def copying(read, write, reservation):
            def view(base):
                return {"base": base, "rows": 1, "cols": 16,
                        "row_stride": 16, "col_stride": 1}

            return {"record": "wave", "groups": [
                {"sm": 0, "rf_kib": reservation, "sh_kib": 0, "commands": [
                    {"command": "run", "instruction": {"op": "load", "tensor": "input",
                        "global": view(read)}},
                    {"command": "run", "instruction": {"op": "store", "tensor": "output",
                        "global": view(write)}}]},
                {"sm": 0, "rf_kib": reservation, "sh_kib": 0, "commands": [
                    {"command": "run", "instruction": {"op": "load", "tensor": "output",
                        "global": view(read)}},
                    {"command": "run", "instruction": {"op": "store", "tensor": "next",
                        "global": view(write)}}]}]}

        hardware = {"sms": 1, "rf_kib": 16, "sh_kib": 0, "resident_groups": 2}
        first = copying(0, 0, 2)
        separate = copying(16, 16, 4)
        stats, records = self.merge([first, separate], mode="elastic", hardware=hardware)
        self.assertEqual(stats["output_waves"], 1)
        self.assertEqual([group["rf_kib"] for group in records[0]["groups"]], [4, 4])
        conflicting = copying(0, 0, 4)
        stats, _ = self.merge([first, conflicting], mode="elastic", hardware=hardware)
        self.assertEqual(stats["output_waves"], 2)

    def test_model_first_prepares_distinct_fusion_options(self):
        try:
            from model.structured import _check_spec, prepare_options, solve_master
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        spec = {"cases": [{"name": "small-prefill",
                          "config": "phase-two-starter/examples/small-config.json",
                          "weight": 1.0}], "domains": {}, "case_domains": {},
                "program_base": "phase-two-starter/examples",
                "fusion_factors": {"small-prefill": [1, 4]}, "max_designs": 2}
        self.assertEqual(_check_spec(spec), 2)
        with tempfile.TemporaryDirectory() as directory:
            hardware, options = prepare_options(spec, Path(directory), timeout=60)
            self.assertEqual(len(hardware), 1)
            self.assertEqual(len(options), 2)
            self.assertEqual({option["design"]["wave_fusion"] for option in options},
                             {1, 4})
            self.assertTrue(all(option["lower_bound_cycles"] > 0 for option in options))
            resumed_hardware, resumed_options = prepare_options(
                spec, Path(directory), timeout=60, resume=True)
            self.assertEqual(hardware, resumed_hardware)
            self.assertEqual([option["program_sha256"] for option in options],
                             [option["program_sha256"] for option in resumed_options])
            proposal, _ = solve_master(hardware, options, {"small-prefill": 1.0})
            self.assertEqual(len(proposal[1]), 1)


    def test_model_first_can_fuse_exported_software_choices(self):
        try:
            from model.structured import _check_spec, prepare_options
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        spec = {"cases": [{"name": "small-prefill",
                          "config": "phase-two-starter/examples/small-config.json",
                          "weight": 1.0}], "domains": {}, "case_domains": {},
                "case_choices": {"small-prefill": [{"policy.m": 4}]},
                "fusion_factors": {"small-prefill": [1, 4]}, "max_designs": 2}
        self.assertEqual(_check_spec(spec), 2)
        with tempfile.TemporaryDirectory() as directory:
            hardware, options = prepare_options(spec, Path(directory), timeout=60)
            self.assertEqual(len(hardware), 1)
            self.assertEqual(len(options), 2)
            self.assertEqual([option["design"]["wave_fusion"] for option in options],
                             [1, 4])
            self.assertTrue(all(option["design"]["policy.m"] == 4 for option in options))
            self.assertNotIn("fusion_stats", options[0])
            self.assertGreater(options[1]["fusion_stats"]["source_waves"], 0)
            self.assertTrue(all((Path(option["directory"]) / "export" / "result.json").exists()
                                for option in options))
            self.assertTrue((Path(options[1]["directory"]) / "export" / "program.jsonl").exists())
            resumed_hardware, resumed_options = prepare_options(
                spec, Path(directory), timeout=60, resume=True)
            self.assertEqual(hardware, resumed_hardware)
            self.assertEqual([option["program_sha256"] for option in options],
                             [option["program_sha256"] for option in resumed_options])


    def test_model_first_selects_static_program_bodies_by_sha(self):
        try:
            from model.structured import _check_spec, prepare_options, solve_master
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        from model.tool import STARTER, sha
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            original = base / "small-prefill.jsonl"
            original.write_bytes((STARTER / "examples/small-prefill.jsonl").read_bytes())
            alternative = root / "alternative.jsonl"
            merge_program(original, alternative, 4)
            spec = {"cases": [{"name": "small-prefill",
                              "config": "phase-two-starter/examples/small-config.json",
                              "weight": 1.0}], "domains": {}, "case_domains": {},
                    "program_base": str(base), "program_choices": {
                        "small-prefill": [str(original), str(alternative)]},
                    "max_designs": 2}
            self.assertEqual(_check_spec(spec), 2)
            output = root / "result"
            output.mkdir()
            hardware, options = prepare_options(spec, output, timeout=60)
            self.assertEqual(len(options), 2)
            self.assertEqual([option["base_program_sha256"] for option in options],
                             [sha(original), sha(alternative)])
            self.assertNotEqual(options[0]["program_sha256"], options[1]["program_sha256"])
            self.assertEqual(len(solve_master(hardware, options, {"small-prefill": 1.0})[0][1]), 1)
            _, resumed = prepare_options(spec, output, timeout=60, resume=True)
            self.assertEqual([option["program_sha256"] for option in options],
                             [option["program_sha256"] for option in resumed])
            alternative.write_text(alternative.read_text() + "\n")
            with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
                prepare_options(spec, output, timeout=60, resume=True)


if __name__ == "__main__":
    unittest.main()
