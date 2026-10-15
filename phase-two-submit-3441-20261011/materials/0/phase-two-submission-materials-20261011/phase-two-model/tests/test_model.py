import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model.hardware import area,base_power,validate
from model.profiles import lower_bound_plain,lower_bound_enhanced,profile
from model.estimate import InfeasibleProgramError,estimate_program,estimate_wave,program_records
from model.serial import serial_wave
from model.compute import compute_wave
from model.tool import RUNTIME,STARTER,OFFICIAL,command,grade_if_target,prepare,search_estimate,write

def view(base,rows,cols,stride=None):
    return dict(base=base,rows=rows,cols=cols,row_stride=cols if stride is None else stride,col_stride=1)

class BoundTests(unittest.TestCase):
    def test_dependency_floor_respects_async_wait_and_hbm_readiness(self):
        try:
            from model.structured import lower_bound_dependencies
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        hardware=json.loads((STARTER/"examples/small-config.json").read_text())["hardware"]
        load={"command":"async","token":1,"instruction":{"op":"load", "tensor":0,
             "global":view(0,1,16),"local":view(0,1,16)}}
        fill={"command":"run","instruction":{"op":"fill","dst":256,"len":32,"value":0.0}}
        wait={"command":"wait","tokens":[1]}
        with tempfile.TemporaryDirectory() as directory:
            program=Path(directory)/"expanded.jsonl"
            program.write_text(json.dumps({"hardware":hardware})+"\n"+
                json.dumps({"record":"wave","groups":[{"sm":0,"rf_kib":1,"sh_kib":0,
                    "commands":[load,fill,wait]}]})+"\n")
            self.assertEqual(lower_bound_dependencies(program),251)

    def test_model_first_master_keeps_one_shared_hardware(self):
        try:
            from model.structured import solve_master
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        hardware=[{"index":0,"area_au":99,"design":{"hardware.rf_tier":1}},
                  {"index":1,"area_au":99,"design":{"hardware.rf_tier":2}}]
        options=[{"hardware_index":0,"case":"prefill","lower_bound_cycles":10,"status":"unverified"},
                 {"hardware_index":0,"case":"decode","lower_bound_cycles":40,"status":"unverified"},
                 {"hardware_index":1,"case":"prefill","lower_bound_cycles":20,"status":"unverified"},
                 {"hardware_index":1,"case":"decode","lower_bound_cycles":10,"status":"unverified"}]
        selected,report=solve_master(hardware,options,{"prefill":0.5,"decode":0.5})
        self.assertEqual(report["status"],0)
        self.assertEqual(selected[0]["index"],1)
        self.assertEqual(selected[1],[2,3])
        options[3]["verified_cycles"]=100
        selected,_=solve_master(hardware,options,{"prefill":0.5,"decode":0.5})
        self.assertEqual(selected[0]["index"],0)
        options[1]["status"]="infeasible"
        selected,_=solve_master(hardware,options,{"prefill":0.5,"decode":0.5})
        self.assertEqual(selected[0]["index"],1)

    def test_model_first_replays_compact_program_with_hardware_choices(self):
        from model.structured import _check_spec, lower_bound_dependencies, prepare_options
        from model.estimate import program_records
        from model.profiles import lower_bound_plain
        from model.tool import STARTER, sha
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            base=root/"base"
            base.mkdir()
            program=base/"small-prefill.jsonl"
            program.write_bytes((STARTER/"examples/small-prefill.jsonl").read_bytes())
            spec={"cases":[{"name":"small-prefill",
                            "config":"phase-two-starter/examples/small-config.json",
                            "weight":1.0}],
                  "domains":{"hardware.rf_tier":[1,2]},"case_domains":{},
                  "program_base":str(base),"max_designs":2}
            self.assertEqual(_check_spec(spec),2)
            output=root/"replay"
            output.mkdir()
            hardware,options=prepare_options(spec,output,timeout=60)
            self.assertEqual(len(hardware),2)
            self.assertEqual(len(options),2)
            self.assertTrue(all(option["base_program_sha256"]==sha(program)
                                for option in options))
            self.assertTrue(all(not (Path(option["directory"])/"export").exists()
                                for option in options))
            self.assertEqual({json.loads(Path(option["program"]).read_text().splitlines()[0])
                              ["hardware"]["rf_tier"] for option in options},{1,2})
            expanded=root/"expanded.jsonl"
            with expanded.open("w") as destination:
                for record in program_records(program):
                    destination.write(json.dumps(record)+"\n")
            self.assertEqual(lower_bound_plain(program),lower_bound_plain(expanded))
            self.assertEqual(lower_bound_dependencies(program),
                             lower_bound_dependencies(expanded))
            spec["case_domains"]={"small-prefill":{"policy.m":[2,4]}}
            with self.assertRaisesRegex(ValueError,"hardware decisions only"):
                _check_spec(spec)
            spec["case_domains"]={}
            spec["domains"]={}
            spec["hardware_choices"]=[{"hardware.rf_tier":1},{"hardware.rf_tier":2}]
            self.assertEqual(_check_spec(spec),2)
            spec["hardware_choices"].append({"hardware.rf_tier":2})
            with self.assertRaisesRegex(ValueError,"duplicate hardware choices"):
                _check_spec(spec)
            spec.pop("hardware_choices")
            spec["case_choices"]={"small-prefill":[{"policy.m":2},{"policy.m":4}]}
            with self.assertRaisesRegex(ValueError,"hardware decisions only"):
                _check_spec(spec)
            spec.pop("program_base")
            self.assertEqual(_check_spec(spec),2)
            spec["case_domains"]={"small-prefill":{"policy.n":[16]}}
            with self.assertRaisesRegex(ValueError,"without case_domains overlap"):
                _check_spec(spec)

    def test_model_first_small_problem_builds_bounds_before_verification(self):
        try:
            from model.structured import solve
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            spec=Path(__file__).resolve().parents[1]/"specs/small-model-first.json"
            result=solve(spec,root/"solution",timeout=60,max_verifications=10)
            self.assertEqual(result["termination"],"optimal_within_declared_domain_and_tolerance")
            self.assertEqual(result["domain_designs"],4)
            self.assertGreater(result["verified_options"],0)
            self.assertAlmostEqual(result["log_gap"],0,places=8)
            self.assertLessEqual(result["model_log_lower_bound"],result["incumbent_log_cost"]+1e-8)
            self.assertTrue((root/"solution/model.json").exists())
            self.assertTrue((root/"solution/best/programs/small-prefill.jsonl").exists())
            prepared=json.loads((root/"solution/preparation.json").read_text())
            self.assertTrue(all(option["dependency_bound_cycles"]>option["resource_bound_cycles"]
                                for option in prepared["options"]))
            for option in prepared["options"]:
                actual=json.loads((Path(option["directory"])/"verification/result.json").read_text())
                self.assertLessEqual(option["lower_bound_cycles"],
                                     actual["evaluation"]["report"]["stats"]["cycles"])

    def test_model_first_case_choices_export_distinct_software(self):
        from model.structured import _check_spec, prepare_options
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            spec={"cases":[{"name":"small-prefill",
                            "config":"phase-two-starter/examples/small-config.json",
                            "weight":1.0}],
                  "domains":{},"case_domains":{},
                  "case_choices":{"small-prefill":[{"policy.m":2},{"policy.m":4}]},
                  "seed":7,"max_designs":2}
            self.assertEqual(_check_spec(spec),2)
            output=root/"generated"
            output.mkdir()
            hardware,options=prepare_options(spec,output,timeout=60)
            self.assertEqual(len(hardware),1)
            self.assertEqual([option["design"] for option in options],
                             spec["case_choices"]["small-prefill"])
            self.assertEqual(len({option["program_sha256"] for option in options}),2)
            self.assertTrue(all((Path(option["directory"])/"export-command.json").exists()
                                for option in options))

    def test_model_first_five_case_target_gate_skips_official_evaluation(self):
        try:
            from model.structured import solve
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        from model.tool import SCORE_CASES
        spec={"cases":[{"name":name,"weight":weight} for name,(_,weight) in SCORE_CASES.items()],
              "domains":{},"case_domains":{},"target_score":5000}
        hardware=[{"index":0,"design":{},"area_au":90,"hardware":{}}]
        options=[{"hardware_index":0,"case":name,"lower_bound_cycles":1,
                  "program":"unused", "program_sha256":"0"*64,
                  "status":"unverified"} for name in SCORE_CASES]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"spec.json";write(path,spec)
            with patch("model.structured.prepare_options",return_value=(hardware,options)), \
                    patch("model.structured.estimate_program",return_value={"cycles":1_000_000_000}), \
                    patch("model.structured.predict") as official_predict:
                result=solve(path,root/"solution",max_verifications=1)
            self.assertEqual(result["termination"],"below_prediction_target_unverified")
            self.assertEqual(result["verified_options"],0)
            self.assertTrue(result["conditional_predicted_optimum_within_declared_domain"])
            self.assertTrue((root/"solution/estimation.json").exists())
            self.assertFalse((root/"solution/grade").exists())
            official_predict.assert_not_called()

    def test_model_first_proven_ceiling_skips_estimation_and_grading(self):
        try:
            from model.structured import solve
        except ModuleNotFoundError as error:
            self.skipTest(str(error))
        from model.tool import SCORE_CASES
        spec={"cases":[{"name":name,"weight":weight} for name,(_,weight) in SCORE_CASES.items()],
              "domains":{},"case_domains":{},"target_score":5000}
        hardware=[{"index":0,"design":{},"area_au":90,"hardware":{}}]
        options=[{"hardware_index":0,"case":name,"lower_bound_cycles":1_000_000_000,
                  "program":"unused","program_sha256":"0"*64,
                  "status":"unverified"} for name in SCORE_CASES]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"spec.json";write(path,spec)
            with patch("model.structured.prepare_options",return_value=(hardware,options)), \
                    patch("model.structured.estimate_program") as estimator, \
                    patch("model.structured.predict") as official_predict:
                result=solve(path,root/"solution",max_verifications=1)
            self.assertEqual(result["termination"],"below_proven_score_upper_bound_unverified")
            self.assertLess(result["provable_score_upper_bound"],5000)
            self.assertTrue(result["target_unattainable_within_declared_domain"])
            self.assertIsNone(result["conditional_predicted_score"])
            self.assertIsNone(result["official_score"])
            self.assertTrue((root/"solution/bound-screen.json").exists())
            self.assertFalse((root/"solution/grade").exists())
            estimator.assert_not_called()
            official_predict.assert_not_called()

    def test_estimate_rejects_overcommitted_workgroups(self):
        hardware=json.loads((STARTER/"examples/small-config.json").read_text())["hardware"]
        hardware["rf_kib"]=64
        groups=[{"sm":0,"rf_kib":40,"sh_kib":0,"commands":[]},
                {"sm":0,"rf_kib":40,"sh_kib":0,"commands":[]}]
        with tempfile.TemporaryDirectory() as directory:
            program=Path(directory)/"overcommitted.jsonl"
            program.write_text(json.dumps({"hardware":hardware})+"\n"+
                json.dumps({"record":"wave","groups":groups})+"\n")
            with self.assertRaisesRegex(InfeasibleProgramError,"resident workgroups exceed"):
                estimate_program(program)

    def test_independent_estimate_small_program(self):
        report=estimate_program(STARTER/"examples/small-prefill.jsonl")
        self.assertTrue(report["independent"])
        self.assertEqual(report["waves"],24)
        self.assertLess(abs(report["cycles"]/23658-1),.05)

    def test_weight_panel_reuse_pairs_across_long_gemm_stage(self):
        from model.reuse_b import reuse_b_program
        hardware=json.loads((STARTER/"examples/small-config.json").read_text())["hardware"]
        hardware.update(rf_kib=128,sh_kib=0,sh_banks=0,sh_tc_bw=0)
        def group(index):
            panel=0 if index in (0,17) else index * 256
            return {"sm":0,"rf_kib":2,"sh_kib":0,"commands":[
                {"command":"run","instruction":{"op":"fill","dst":0,"len":16,"value":0.0}},
                {"command":"async","token":0,"instruction":{"op":"load","tensor":1,
                    "global":view(index*16,1,16),"local":view(16,1,16)}},
                {"command":"async","token":1,"instruction":{"op":"load","tensor":2,
                    "global":view(panel,16,16),"local":view(32,16,16)}},
                {"command":"wait","tokens":[0,1]},
                {"command":"run","instruction":{"op":"mma","a":16,"b":32,"c":0,
                    "m":1,"n":16,"k":16}},
                {"command":"run","instruction":{"op":"store","tensor":3,
                    "global":view(index*16,1,16),"local":view(0,1,16)}}]}
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"source.jsonl"
            destination=Path(directory)/"shared.jsonl"
            source.write_text("".join(json.dumps(record)+"\n" for record in [
                {"hardware":hardware,"decode":True},
                *({"record":"wave","groups":[group(index)]} for index in range(18))]))
            stats=reuse_b_program(source,destination)
            self.assertEqual(stats["paired_groups"],1)
            self.assertEqual(stats["removed_b_loads"],1)
            self.assertEqual(stats["source_waves"],18)
            waves=[record for record in program_records(destination) if record.get("record")=="wave"]
            self.assertEqual(sum(len(record["groups"]) for record in waves),17)
            self.assertEqual(sum(command.get("instruction",{}).get("op")=="load" and
                command["instruction"]["tensor"]==2 for wave in waves
                for group in wave["groups"] for command in group["commands"]),17)

    def test_adaptive_merge_reserves_maximum_rf(self):
        from model.merge import merge_program
        hardware=json.loads((STARTER/"examples/small-config.json").read_text())["hardware"]
        hardware.update(sh_kib=0,sh_banks=0,sh_tc_bw=0)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"source.jsonl"
            destination=Path(directory)/"adaptive.jsonl"
            waves=[]
            for offset,rf_kib in ((0,1),(16,2)):
                waves.append({"record":"wave","groups":[{"sm":0,"sh_kib":0,
                    "rf_kib":rf_kib,"commands":[
                    {"command":"async","token":1,"instruction":{"op":"load","tensor":1,
                        "global":view(offset,1,16),"local":view(0,1,16)}},
                    {"command":"wait","tokens":[1]},
                    {"command":"run","instruction":{"op":"vector","kind":"add",
                        "dst":0,"len":16,"a":{"kind":"reg","base":0,"stride":1},
                        "b":{"kind":"imm","value":1.0}}},
                    {"command":"run","instruction":{"op":"store","tensor":2,
                        "global":view(offset,1,16),"local":view(0,1,16)}}]}]})
            source.write_text("".join(json.dumps(record)+"\n" for record in
                [{"hardware":hardware},*waves]))
            stats=merge_program(source,destination,2,"adaptive")
            self.assertEqual(stats["fused_adaptive_waves"],1)
            output=list(program_records(destination))[1]
            self.assertEqual(output["groups"][0]["rf_kib"],2)
            self.assertEqual(len(output["groups"][0]["commands"]),8)

    def test_bulk_dma_diagnostic_preserves_default_estimate(self):
        hardware=json.loads((STARTER/"examples/small-config.json").read_text())["hardware"]
        group={"sm":0,"commands":[{"command":"run","instruction":{
            "op":"load","tensor":1,"global":view(index*16,1,16),
            "local":view(0,1,16)}} for index in range(130)]}
        default=estimate_wave([group],hardware)
        explicit=estimate_wave([group],hardware,bulk_dma_approximation=True)
        diagnostic=estimate_wave([group],hardware,bulk_dma_approximation=False)
        self.assertEqual(default,explicit)
        self.assertGreaterEqual(diagnostic,default)

    def test_compact_reader_expands_template_and_command_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"compact.jsonl"
            command_body={"command":"repeat","var":"c","start":0,"count":2,"step":1,
                "body":[{"command":"run","instruction":{"op":"fill","len":16,
                    "dst":{"add":[{"var":"p"},{"mul":[{"var":"c"},16]}]},"value":0.0}}]}
            records=[{"hardware":{}},{"record":"template","kind":"record","name":"t",
                "params":["p"],"body":[{"record":"repeat","var":"r","start":0,"count":2,
                    "step":1,"body":[{"record":"wave","groups":[{"sm":{"var":"r"},
                        "rf_kib":1,"sh_kib":0,"commands":[command_body]}]}]}]},
                {"record":"call","name":"t","args":[32]},{"record":"commit"}]
            path.write_text("".join(json.dumps(record)+"\n" for record in records))
            expanded=list(program_records(path))
            self.assertEqual([record.get("record") for record in expanded],[None,"wave","wave","commit"])
            self.assertEqual([record["groups"][0]["sm"] for record in expanded[1:3]],[0,1])
            self.assertEqual([command["instruction"]["dst"] for command in expanded[1]["groups"][0]["commands"]],[32,48])

    def test_write_line_reuse_is_not_free(self):
        hardware=json.loads((STARTER/"examples/small-config.json").read_text())["hardware"]
        def group(base):
            return {"sm":0,"commands":[{"command":"run","instruction":{
                "op":"store","tensor":0,"global":view(base,1,32),"local":view(0,1,32)}}]}
        aligned=estimate_wave([group(0)],hardware)
        crossing=estimate_wave([group(1)],hardware)
        self.assertGreater(crossing-aligned,250)

    def test_cross_group_line_conflict_respects_dma_parallelism(self):
        hardware=json.loads((STARTER/"examples/small-config.json").read_text())["hardware"]
        groups=[]
        for base in [0,4]:
            groups.append({"sm":0,"commands":[{"command":"run","instruction":{
                "op":"store","tensor":0,"global":view(base,1,4),"local":view(0,1,4)}}]})
        single=estimate_wave(groups,hardware,details=True)
        hardware.update(dma_engines=2,dma_depth=2)
        parallel=estimate_wave(groups,hardware,details=True)
        self.assertEqual(single["collision"],260)
        self.assertEqual(parallel["collision"],520)

    def test_candidate_estimate_does_not_replace_strict_bound(self):
        spec={"cases":[{"name":"small-prefill","config":"phase-two-starter/examples/small-config.json",
                        "weight":1.0}],"domains":{"hardware.rf_tier":[1]},
              "estimate_candidates":True}
        with tempfile.TemporaryDirectory() as directory:
            candidate=prepare(spec,Path(directory),60)[0]
            expanded=candidate["directory"]/"export-small-prefill/program.jsonl"
            self.assertEqual(candidate["bounds"],[lower_bound_plain(expanded)])
            self.assertEqual(candidate["estimated_cycles"],[23448])
            self.assertGreater(candidate["estimated_cycles"][0],candidate["bounds"][0])

    def test_finite_estimate_search_ranks_without_official_timing(self):
        spec={"cases":[{"name":"small-prefill","config":"phase-two-starter/examples/small-config.json",
                        "weight":1.0,"overrides":{"hardware.rf_tier":3}}],
              "domains":{"hardware.rf_tier":[1,2]},"max_designs":2}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"spec.json";write(path,spec)
            report=search_estimate(path,root/"search",60)
            self.assertEqual(report["domain_designs"],2)
            self.assertEqual(report["best_design"],{"hardware.rf_tier":2})
            self.assertEqual(report["best_predicted_cycles"],{"small-prefill":23406})
            self.assertIsNone(report["best_score_if_eligible"])
            self.assertEqual(report["numerical_correctness"],"not_checked")
            self.assertEqual(json.loads((root/"search/candidate-0001/small-prefill.config.json").read_text())
                             ["hardware"]["rf_tier"],2)
            self.assertTrue((root/"search/best/programs/small-prefill.jsonl").exists())

    def test_hardware_replay_search_preserves_program_body(self):
        spec={"cases":[{"name":"small-prefill","config":"phase-two-starter/examples/small-config.json",
                        "weight":1.0}],"program_base":"phase-two-starter/examples",
              "domains":{"hardware.rf_tier":[1,2]},"max_designs":2,"target_score":5000}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"spec.json";write(path,spec)
            report=search_estimate(path,root/"search",60)
            self.assertEqual(report["domain_designs"],2)
            self.assertFalse(report["target_met"])
            self.assertEqual(report["best_predicted_cycles"],{"small-prefill":23406})
            base=STARTER/"examples/small-prefill.jsonl"
            with base.open("rb") as original, (root/"search/best/programs/small-prefill.jsonl").open("rb") as replay:
                original.readline();replay.readline()
                self.assertEqual(original.read(),replay.read())

    def test_hardware_replay_rejects_software_domains(self):
        spec={"cases":[{"name":"small-prefill","config":"phase-two-starter/examples/small-config.json",
                        "weight":1.0}],"program_base":"phase-two-starter/examples",
              "domains":{"policy.split_k":[1]}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"spec.json";write(path,spec)
            with self.assertRaisesRegex(ValueError,"only supports hardware"):
                search_estimate(path,root/"search",60)

    def test_casewise_search_proves_best_of_cartesian_product(self):
        cases=[{"name":f"small-{index}","config":"phase-two-starter/examples/small-config.json",
                "weight":1/3} for index in range(3)]
        spec={"cases":cases,"domains":{},
            "case_domains":{case["name"]:{"policy.m":[2,4]} for case in cases},
            "max_designs":8}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"spec.json";write(path,spec)
            report=search_estimate(path,root/"search",60)
            self.assertEqual(report["domain_designs"],8)
            self.assertEqual(report["evaluated_designs"],8)
            self.assertEqual(report["estimated_programs"],6)
            candidate=report["candidates"][0]
            for case in cases:
                name=case["name"]
                options=candidate["case_options"][name]
                self.assertEqual(len(options),2)
                self.assertEqual(report["best_predicted_cycles"][name],
                                 min(option["cycles"] for option in options))
                self.assertEqual(report["best_program_sha256"][name],
                                 next(option["program_sha256"] for option in options
                                      if option["cycles"]==report["best_predicted_cycles"][name]))
                self.assertTrue((root/"search/best/programs"/f"{name}.jsonl").exists())
            self.assertIsNone(report["best_score_if_eligible"])

    def test_casewise_search_rejects_case_hardware_changes(self):
        spec={"cases":[{"name":"small-prefill","config":"phase-two-starter/examples/small-config.json",
                        "weight":1.0}],"domains":{},
              "case_domains":{"small-prefill":{"hardware.rf_kib":[64,128]}}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"spec.json";write(path,spec)
            with self.assertRaisesRegex(ValueError,"cannot override shared hardware"):
                search_estimate(path,root/"search",60)

    def test_casewise_search_selects_shared_hardware_first(self):
        cases=[{"name":f"small-{index}","config":"phase-two-starter/examples/small-config.json",
                "weight":0.5} for index in range(2)]
        spec={"cases":cases,"domains":{"hardware.rf_tier":[1,2]},
              "case_domains":{case["name"]:{"policy.m":[2,4]} for case in cases},
              "max_designs":8}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"spec.json";write(path,spec)
            report=search_estimate(path,root/"search",60)
            self.assertEqual(report["domain_designs"],8)
            self.assertEqual(report["evaluated_designs"],8)
            self.assertEqual(report["estimated_programs"],8)
            self.assertEqual(report["best_estimated_log_cost"],
                             min(candidate["predicted_log_cost"] for candidate in report["candidates"]))
            with (root/"search/best/programs/small-0.jsonl").open() as source:
                self.assertEqual(json.loads(next(source))["hardware"]["rf_tier"],
                                 report["best_design"]["hardware.rf_tier"])

    def test_grade_target_does_not_start_evaluator_below_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);search=root/"search";search.mkdir()
            write(search/"search.json",{"best_score_if_eligible":2105.13})
            with self.assertRaisesRegex(ValueError,"below grade target"):
                grade_if_target(search,root/"grade",target=5000)
            self.assertFalse((root/"grade").exists())

    def test_grade_target_rejects_changed_programs_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);search=root/"search";programs=search/"best/programs"
            programs.mkdir(parents=True)
            hashes={name:"0"*64 for name in ("w-p","a-p","w-d1","w-d16","w-d4l")}
            for name in hashes:(programs/f"{name}.jsonl").write_text("{}\n")
            write(search/"search.json",{"best_score_if_eligible":5001,
                                        "best_program_sha256":hashes})
            with self.assertRaisesRegex(ValueError,"program hashes changed"):
                grade_if_target(search,root/"grade")
            self.assertFalse((root/"grade").exists())

    def test_per_wave_rf_port_bound(self):
        hardware=json.loads((STARTER/"examples/small-config.json").read_text())["hardware"]
        hardware["rf_tier"]=1
        hardware["vector"]=32
        instruction=dict(op="fill",dst=0,len=2048,value=0.0)
        wave=dict(record="wave",groups=[dict(sm=0,rf_kib=16,sh_kib=0,
            commands=[dict(command="run",instruction=instruction)])])
        with tempfile.TemporaryDirectory() as directory:
            program=Path(directory)/"waves.jsonl"
            records=[dict(hardware=hardware),wave,wave,dict(record="commit")]
            program.write_text("".join(json.dumps(record)+"\n" for record in records))
            self.assertEqual(lower_bound_plain(program),2*(2048*4//64+1)+1)

    def test_core_work_is_pinned_to_sm(self):
        hardware=json.loads((STARTER/"examples/small-config.json").read_text())["hardware"]
        instruction=dict(op="vector",kind="exp",dst=64,len=64,
            a=dict(kind="reg",base=0,stride=1),b=dict(kind="imm",value=0.0))
        group=lambda sm:dict(sm=sm,rf_kib=1,sh_kib=0,
            commands=[dict(command="run",instruction=instruction)])
        with tempfile.TemporaryDirectory() as directory:
            program=Path(directory)/"groups.jsonl"
            def bound(groups):
                program.write_text(json.dumps(dict(hardware=hardware))+"\n"+
                    json.dumps(dict(record="wave",groups=groups))+"\n")
                return lower_bound_plain(program)
            self.assertEqual(bound([group(0),group(0)]),2*8*12+1)
            self.assertEqual(bound([group(0),group(1)]),8*12+1)

    def test_bound_below_full_small_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            config=STARTER/"examples/small-config.json"
            command([OFFICIAL,"export",config,7,directory/"export"],60,directory/"export-command.json")
            program=directory/"export/program.jsonl"
            bound=lower_bound_plain(program)
            command([OFFICIAL,"evaluate",program,7,directory/"evaluation"],60,directory/"evaluate-command.json")
            result=json.loads((directory/"evaluation/result.json").read_text())
            self.assertEqual(result["status"],"complete")
            self.assertLessEqual(bound,result["evaluation"]["report"]["stats"]["cycles"])

    def test_enhanced_bound_below_full_small_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            config=STARTER/"examples/small-config.json"
            command([OFFICIAL,"export",config,7,directory/"export"],60,directory/"export-command.json")
            program=directory/"export/program.jsonl"
            plain=lower_bound_plain(program)
            enhanced=lower_bound_enhanced(program)
            self.assertGreaterEqual(enhanced,plain)
            command([OFFICIAL,"evaluate",program,7,directory/"evaluation"],60,directory/"evaluate-command.json")
            result=json.loads((directory/"evaluation/result.json").read_text())
            self.assertEqual(result["status"],"complete")
            cycles=result["evaluation"]["report"]["stats"]["cycles"]
            self.assertLessEqual(enhanced,cycles)

class DifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=json.loads((STARTER/"examples/small-config.json").read_text())["hardware"]
        cls.temp=tempfile.TemporaryDirectory();cls.directory=Path(cls.temp.name);cls.counter=0
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def backend(self,mode,payload):
        self.__class__.counter+=1;k=self.counter
        src=self.directory/f"input-{k}.json";out=self.directory/f"output-{k}.json"
        write(src,payload);command([RUNTIME,mode,src,out],30,self.directory/f"command-{k}.json")
        return json.loads(out.read_text())["result"]
    def variants(self):
        rng=random.Random(20261009)
        for _ in range(12):
            h=copy.deepcopy(self.base)
            h.update(sms=2,rf_kib=rng.choice([32,64,128,256]),rf_tier=rng.choice([1,2,3]),
                p=rng.choice([2,4,8]),q=rng.choice([8,16]),tc_k_parallel=rng.choice([1,2,4]),
                sh_kib=64,sh_banks=rng.choice([1,4,16]),shared_ports=rng.choice([1,2]),
                reduction_units=rng.choice([0,1,2]),vector=rng.choice([8,32,64]),sfu=3)
            validate(h);yield h
    def instructions(self):
        reg=lambda base,stride=1:dict(kind="reg",base=base,stride=stride)
        imm=dict(kind="imm",value=.5)
        result=[dict(op="fill",dst=0,len=33,value=0.0)]
        for kind in ["add","fma","square","exp","rsqrt","tanh"]:
            result.append(dict(op="vector",kind=kind,dst=256,len=19,a=reg(0,0),b=imm))
        result.extend([dict(op="reduce",scratch=1024,dst=2048,src=0,len=n,max=False) for n in [1,3,33]])
        result.extend([dict(op="mma",a=0,b=1024,c=2048,m=7,n=13,k=k) for k in [1,3,17]])
        result.extend([dict(op="mma_shared",a=0,b=view(0,17,13,stride),c=2048,m=7,n=13,k=17) for stride in [13,16]])
        result.extend([dict(op=op,shared=view(0,17,1,stride),local=view(4096,17,1))
                       for op in ["shared_read","shared_write"] for stride in [1,16]])
        return result
    def test_independent_area_and_microsteps(self):
        instructions=self.instructions()
        for h in self.variants():
            actual=self.backend("profiles",dict(hardware=h,instructions=instructions))
            self.assertAlmostEqual(area(h),actual["area_au"],places=11)
            self.assertAlmostEqual(base_power(h),actual["base_power_w"],places=12)
            for i,p in zip(instructions,actual["profiles"]):self.assertEqual(profile(i,h),p)
    def test_independent_serial_cycles_and_power(self):
        # Covers tails, Kp, reductions, SFU, bank conflicts and SH->TC timing.
        instructions=self.instructions()
        for h in self.variants():
            groups=[dict(sm=0,rf_kib=32,sh_kib=16,
                         commands=[dict(command="run",instruction=i) for i in instructions])]
            actual=self.backend("wave",dict(hardware=h,groups=groups))["report"]
            expected=serial_wave(instructions,h)
            for field in ["cycles","rf_bytes","sh_bytes","physical_fmas"]:
                self.assertEqual(expected[field],actual["stats"][field],field)
            for field in ["short_power_w","long_power_w","energy_j"]:
                self.assertAlmostEqual(expected[field],actual[field],delta=max(1e-12,abs(actual[field])*1e-11))
    def test_rejects_capacity_and_async_hazard(self):
        h=copy.deepcopy(self.base);h["sms"]=32;h["rf_tier"]=3
        with self.assertRaises(ValueError):validate(h)
        h=copy.deepcopy(self.base);h["sfu"]=129
        with self.assertRaises(ValueError):validate(h)
        # Official static dependency guard must reject un-awaited destination use.
        h=copy.deepcopy(self.base)
        group=dict(sm=0,rf_kib=1,sh_kib=0,commands=[
            dict(command="async",token=1,instruction={"op":"load","tensor":0,"global":view(0,1,16),"local":view(0,1,16)}),
            dict(command="run",instruction=dict(op="fill",dst=0,len=16,value=0.)),
            dict(command="wait",tokens=[1])])
        with self.assertRaises(RuntimeError):self.backend("wave",dict(hardware=h,groups=[group],bases=[0]))

    def test_independent_multiple_groups_fixed_arbitration(self):
        instructions=self.instructions()
        for index,h in enumerate(self.variants()):
            h.update(rf_kib=128,tc_count=1+index%2,resident_groups=4)
            groups=[]
            for gi in range(4):
                # Different group lengths, engine paths and SM placements exercise
                # fixed rotating order and readiness-dependent event jumps.
                subset=instructions[gi:10+gi]+instructions[12:]
                groups.append(dict(sm=gi%2 if index%2 else 0,rf_kib=32,sh_kib=16,
                    commands=[dict(command="run",instruction=i) for i in subset]))
            expected=compute_wave(groups,h)
            actual=self.backend("wave",dict(hardware=h,groups=groups))["report"]
            for field in ("cycles","rf_bytes","sh_bytes","physical_fmas","scheduler_iterations"):
                self.assertEqual(expected[field],actual["stats"][field],field)
            for field in ("short_power_w","long_power_w","energy_j"):
                self.assertAlmostEqual(expected[field],actual[field],delta=max(1e-12,abs(actual[field])*1e-10))

    def test_timing_prediction_is_not_a_numerical_certificate(self):
        source=STARTER/"examples/small-prefill.jsonl"
        records=[json.loads(line) for line in source.read_text().splitlines()]
        changed=False
        for r in records:
            if r.get("record")!="wave":continue
            for g in r["groups"]:
                for c in g["commands"]:
                    i=c.get("instruction",{})
                    if i.get("op")=="fill" and not changed:
                        i["value"]=1000.0;changed=True
        self.assertTrue(changed)
        program=self.directory/"numerically-wrong.jsonl"
        program.write_text("".join(json.dumps(r)+"\n" for r in records))
        prediction=self.directory/"wrong-prediction.json"
        command([RUNTIME,"predict",program,prediction],30,self.directory/"wrong-predict-command.json")
        self.assertEqual(json.loads(prediction.read_text())["result"]["numerical_correctness"],"not_checked")
        out=self.directory/"wrong-official"
        with self.assertRaises(RuntimeError):
            command([OFFICIAL,"evaluate",program,7,out],30,self.directory/"wrong-evaluate-command.json")
        report=json.loads((out/"result.json").read_text())
        self.assertEqual(report["status"],"failed")
        self.assertTrue(report["error"])

    def test_prediction_rejects_cross_group_hbm_race(self):
        header=json.loads((STARTER/"examples/small-prefill.jsonl").read_text().splitlines()[0])
        group=lambda sm:dict(sm=sm,rf_kib=1,sh_kib=0,commands=[
            dict(command="run",instruction=dict(op="fill",dst=0,len=16,value=0.)),
            dict(command="run",instruction={"op":"store","tensor":0,"global":view(0,1,16),"local":view(0,1,16)})])
        records=[header,dict(record="alloc",len=16),dict(record="wave",groups=[group(0),group(1)])]
        program=self.directory/"race.jsonl";program.write_text("".join(json.dumps(r)+"\n" for r in records))
        log=self.directory/"race-command.json"
        with self.assertRaises(RuntimeError):command([RUNTIME,"predict",program,self.directory/"race-output.json"],30,log)
        self.assertIn("HBM race",json.loads(log.read_text())["stderr"])

if __name__=="__main__":unittest.main()
