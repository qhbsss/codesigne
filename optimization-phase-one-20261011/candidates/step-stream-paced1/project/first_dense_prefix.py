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
lines=Path(a.program).read_text().splitlines();depth=0;active=False;stop=None
for i,l in enumerate(lines):
 if l.startswith('FOR '):
  if depth==0 and lines[i+1].startswith('VEC ') and json.loads(lines[i+1][4:])['dst'].get('lane')==14:active=True
  depth+=1
 elif l.startswith('END.FOR '):
  depth-=1
  if depth==0 and active:stop=i+2;break
assert stop and lines[stop-1].startswith('BARRIER ')
program='\n'.join(lines[:stop]+[l for l in lines if l.startswith('WG.END ')])+'\n';h=Hardware.from_dict(json.loads(Path(a.hardware).read_text()));t=time.monotonic()
r=asdict(estimate_pipeline(h,iter_parse(program),required_hbm_words(program,build_layout(MODELS['M1'],'P1')),None))
Path(a.report).write_text(json.dumps({'kind':'exact-initialization-qkv-prefix','timing':r,'elapsed':time.monotonic()-t},indent=2)+'\n');print(json.dumps({k:r[k] for k in ('cycles','peak_window_power_w','hbm_read_bytes')}))
