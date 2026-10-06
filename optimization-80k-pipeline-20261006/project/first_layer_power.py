"""Observe unchanged public timing; independently locate its exact power window."""
import argparse,json,time
from pathlib import Path
import numpy as np
from codesign.challenge.pipeline_events import GlobalEventPipeline
from codesign.challenge.hardware import Hardware,cost_model
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.isa import iter_parse
from codesign.challenge.runner import required_hbm_words
from codesign.challenge.abi import build_layout
from codesign.challenge.workload import MODELS
from dataclasses import asdict
p=argparse.ArgumentParser();p.add_argument('--hardware',required=True);p.add_argument('--program',required=True);p.add_argument('--report',required=True);a=p.parse_args()
original=GlobalEventPipeline._issue;captured=[];barriers=[]
def observe(self,preview):
    if not captured:captured.append(self)
    begin,pc,tape,selected=preview;ins=tape.next_command()[1]
    if ins.op in ('BARRIER','STEP.COMMIT'):barriers.append({'cycle':begin,'line':ins.line,'op':ins.op})
    return original(self,preview)
GlobalEventPipeline._issue=observe
h=Hardware.from_dict(json.loads(Path(a.hardware).read_text()));started=time.monotonic()
layout=build_layout(MODELS['M1'],'P1');address=layout.symbols['layer1/ln1_g'].address//4
lines=Path(a.program).read_text().splitlines();stop=next(i for i,l in enumerate(lines) if l.startswith('LD ') and (v:=json.loads(l[3:]))['src'].get('offset')==address and v['dst']['space'] in ('SH','RF'))
program='\n'.join(lines[:stop]+[l for l in lines if l.startswith('WG.END ')])+'\n'
timing=asdict(estimate_pipeline(h,iter_parse(program),required_hbm_words(program,layout),None));pipeline=captured[0];events=pipeline.energy_events
changes=np.zeros(pipeline.last_finish+1001)
starts=np.fromiter((e.start for e in events),dtype=np.int64);ends=np.fromiter((e.finish for e in events),dtype=np.int64);rates=np.fromiter((e.dynamic_pj/(e.finish-e.start) for e in events),dtype=np.float64)
np.add.at(changes,starts,rates);np.add.at(changes,ends,-rates)
prefix=np.concatenate(([0.0],np.cumsum(np.cumsum(changes))))
window=cost_model()['power_window_cycles'];powers=(prefix[window:]-prefix[:-window])/window*cost_model()['clock_hz']*1e-12+h.area_mm2()*cost_model()['static_w_per_mm2']
peak=int(np.argmax(powers));assert abs(float(powers[peak])-timing['peak_window_power_w'])<1e-6
selected=[];remaining=powers.copy()
for _ in range(6):
 i=int(np.argmax(remaining));selected.append({'start':i,'finish':i+window,'power_w':float(powers[i]),'barriers_nearby':[b for b in barriers if i-200<=b['cycle']<=i+window+200]});remaining[max(0,i-window):min(len(remaining),i+window)]=-1
report={'kind':'first-p1-layer-exact-power-hotspots','elapsed_seconds':time.monotonic()-started,'timing':timing,'windows':selected};Path(a.report).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'cycles':timing['cycles'],'peak_power_w':timing['peak_window_power_w'],'windows':selected}))
