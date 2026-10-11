"""Observe public scheduler barriers; delegate all scheduling unchanged."""
import argparse,json,time
from pathlib import Path
from codesign.challenge.pipeline_events import GlobalEventPipeline
from codesign.challenge.hardware import Hardware
from codesign.challenge.runner import estimate_case
p=argparse.ArgumentParser();p.add_argument('--case',default='M2_D1');p.add_argument('--hardware',required=True);p.add_argument('--program',required=True);p.add_argument('--report',required=True);a=p.parse_args()
original=GlobalEventPipeline._issue;observations=[];started=time.monotonic()
def observed(self,preview):
    begin,pc,tape,selected=preview;ins=tape.next_command()[1]
    if ins.op in ('BARRIER','STEP.COMMIT'):
        entry=dict(op=ins.op,cycle=begin,line=ins.line,elapsed=time.monotonic()-started);observations.append(entry);print(json.dumps(entry),flush=True)
    return original(self,preview)
GlobalEventPipeline._issue=observed
hardware=Hardware.from_dict(json.loads(Path(a.hardware).read_text()));timing=estimate_case(a.case,hardware,Path(a.program).read_text())
Path(a.report).write_text(json.dumps(dict(timing=timing,observations=observations),indent=2));print(json.dumps({k:timing[k] for k in ['cycles','peak_window_power_w']}),flush=True)
