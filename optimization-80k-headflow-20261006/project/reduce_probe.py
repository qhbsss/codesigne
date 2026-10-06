"""Timing of complete initialization/normalization/cache/QKV prefix."""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
from codesign.challenge.hardware import Hardware
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.isa import iter_parse
from codesign.challenge.runner import required_hbm_words
from codesign.challenge.abi import build_layout
from codesign.challenge.workload import MODELS
p=argparse.ArgumentParser();p.add_argument('--program',required=True);p.add_argument('--hardware',required=True);p.add_argument('--report',required=True);a=p.parse_args()
lines=Path(a.program).read_text().splitlines()
start=next(i for i,l in enumerate(lines) if l.startswith('VEC ') and (d:=json.loads(l[4:]))['dst'].get('lane')==11 and d['dst']['count']==20)
end=next(i for i in range(start,len(lines)) if lines[i].startswith('BARRIER '))
lines=[l for l in lines if l.startswith('WG.BEGIN ')]+lines[start:end+1]+[l for l in lines if l.startswith('WG.END ')]
stop=len(lines)
program='\n'.join(lines[:stop])+'\n';h=Hardware.from_dict(json.loads(Path(a.hardware).read_text()));t=time.monotonic()
r=asdict(estimate_pipeline(h,iter_parse(program),required_hbm_words(program,build_layout(MODELS['M1'],'P1')),None))
Path(a.report).write_text(json.dumps({'kind':'isolated-p1-reduction-only','timing':r,'elapsed':time.monotonic()-t},indent=2)+'\n');print(json.dumps({k:r[k] for k in ('cycles','peak_window_power_w','hbm_read_bytes')}))
