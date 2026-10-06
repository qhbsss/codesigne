"""Locate the peak window without changing the public scheduling decisions."""
import argparse,json
from pathlib import Path
import numpy as np
from codesign.challenge.pipeline_events import GlobalEventPipeline
from codesign.challenge.hardware import Hardware
from codesign.challenge.runner import estimate_case
p=argparse.ArgumentParser();p.add_argument('--hardware',required=True);p.add_argument('--program',required=True);p.add_argument('--report',required=True);a=p.parse_args()
original=GlobalEventPipeline._issue;observations=[];engine=None
def observed(self,preview):
    global engine
    engine=self
    begin,pc,tape,selected=preview;ins=tape.next_command()[1]
    if ins.op in ('BARRIER','STEP.COMMIT'):observations.append(dict(op=ins.op,cycle=begin,line=ins.line))
    return original(self,preview)
GlobalEventPipeline._issue=observed
h=Hardware.from_dict(json.loads(Path(a.hardware).read_text()));r=estimate_case('M2_D1',h,Path(a.program).read_text())
events=engine.energy_events
last=max(e.finish for e in events);diff=np.zeros(last+1001,dtype=np.float64)
for e in events:
    rate=e.dynamic_pj/(e.finish-e.start);diff[e.start]+=rate;diff[e.finish]-=rate
integral=np.r_[0,np.cumsum(np.cumsum(diff))];energy=integral[1000:]-integral[:-1000];start=int(np.argmax(energy))
output=dict(timing=r,peak_start=start,peak_end=start+1000,peak_w=float(energy[start])*5e-7+h.static_power_w(),nearby=[o for o in observations if start-1000<=o['cycle']<=start+2000],observations=observations)
Path(a.report).write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k not in ('timing','observations')},indent=2))
