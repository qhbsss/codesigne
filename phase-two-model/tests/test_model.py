import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

from model.hardware import area,base_power,validate
from model.profiles import profile
from model.serial import serial_wave
from model.compute import compute_wave
from model.tool import RUNTIME,STARTER,OFFICIAL,command,write

def view(base,rows,cols,stride=None):
    return dict(base=base,rows=rows,cols=cols,row_stride=cols if stride is None else stride,col_stride=1)

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
