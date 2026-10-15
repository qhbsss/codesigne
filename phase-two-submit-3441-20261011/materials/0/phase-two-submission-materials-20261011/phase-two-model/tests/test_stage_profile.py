import unittest

from model.estimate import estimate_program
from model.stage_profile import stage_profile
from model.tool import STARTER, sha


class StageProfileTests(unittest.TestCase):
    def test_stage_cycles_reconcile_with_independent_estimator(self):
        program = STARTER / "examples/small-prefill.jsonl"
        profile = stage_profile(program)
        total = estimate_program(program)
        self.assertEqual(profile["cycles"], total["cycles"])
        self.assertEqual(profile["waves"], total["waves"])
        self.assertEqual(profile["cycles"], profile["commits"] +
                         sum(stage["cycles"] for stage in profile["stages"].values()))
        self.assertTrue(any(name.startswith("mma:") for name in profile["stages"]))
        self.assertEqual(profile["program_sha256"], sha(program))
        self.assertTrue(profile["not_an_official_score"])


if __name__ == "__main__":
    unittest.main()
