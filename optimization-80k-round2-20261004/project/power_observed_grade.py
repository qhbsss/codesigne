"""Unchanged grade CLI with timing progress and first-layer power diagnostics."""
import json,runpy,sys,time
from pathlib import Path
from codesign.challenge.pipeline_events import GlobalEventPipeline
from codesign.challenge.power import power_profile_w
from codesign.challenge.abi import build_layout
from codesign.challenge.workload import MODELS
original=GlobalEventPipeline._issue;started=time.monotonic();last={};seen=set()
program=Path(sys.argv[sys.argv.index('--program-p1')+1]).read_text().splitlines()
address=build_layout(MODELS['M1'],'P1').symbols['layer1/ln1_g'].address//4
mark=next(i for i,line in enumerate(program) if line.startswith('LD ') and (args:=json.loads(line[3:]))['src'].get('offset')==address and args['dst']['space'] in ('SH','RF'))
# Index of next-layer first LD equals the one-based line number of preceding barrier.
def observed(self,preview):
    begin,pc,tape,selected=preview;ins=tape.next_command()[1]
    if ins.op=='BARRIER' and ins.line==mark and len(ins.args.get('wgs',[]))==64 and id(self) not in seen:
        power=power_profile_w(self.hw.area_mm2(),self.energy_events)[0]
        print(json.dumps(dict(first_layer_cycles=begin,first_layer_peak_window_power_w=power,elapsed=time.monotonic()-started)),file=sys.stderr,flush=True);seen.add(id(self))
    if ins.op in ('BARRIER','STEP.COMMIT') and (begin-last.get(id(self),-10000)>=10000 or ins.op=='STEP.COMMIT'):
        print(json.dumps(dict(progress=ins.op,cycle=begin,line=ins.line,elapsed=time.monotonic()-started)),file=sys.stderr,flush=True);last[id(self)]=begin
    return original(self,preview)
GlobalEventPipeline._issue=observed
sys.argv=['challenge.py']+sys.argv[1:];runpy.run_path('challenge.py',run_name='__main__')
