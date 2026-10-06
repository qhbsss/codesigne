"""Unchanged public timing of an isolated dense row loop; not a full grade."""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
from codesign.challenge.hardware import Hardware
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.isa import iter_parse
from codesign.challenge.runner import required_hbm_words
from codesign.challenge.abi import build_layout
from codesign.challenge.workload import MODELS
p=argparse.ArgumentParser();p.add_argument('--program',required=True);p.add_argument('--hardware',required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--report',required=True);a=p.parse_args()
lines=Path(a.program).read_text().splitlines();begins=[l for l in lines if l.startswith('WG.BEGIN ')];ends=[l for l in lines if l.startswith('WG.END ')];blocks=[];depth=0;start=None
for i,l in enumerate(lines):
 if l.startswith('FOR '):
  d=json.loads(l.split(' ',1)[1])
  if depth==0 and d.get('start')==0 and d.get('step')==1 and lines[i+1].startswith('VEC ') and json.loads(lines[i+1].split(' ',1)[1])['dst'].get('lane') in (10,12,13,14):
   boundary=i-1
   while boundary>=0:
    if lines[boundary].startswith('BARRIER ') and boundary+1<i and lines[boundary+1].startswith('LD '):
     aa=json.loads(lines[boundary+1][3:])
     if aa['dst'].get('lane') in (8,12,14) and aa['src']['space']=='HBM':break
    boundary-=1
   start=boundary+1
  depth+=1
 elif l.startswith('END.FOR '):
  depth-=1
  if depth==0 and start is not None:blocks.append(lines[start:i+1]);start=None
program='\n'.join(begins+blocks[a.index]+ends)+'\n';hw=Hardware.from_dict(json.loads(Path(a.hardware).read_text()));t=time.monotonic()
r=asdict(estimate_pipeline(hw,iter_parse(program),required_hbm_words(program,build_layout(MODELS['M1'],'P1')),None))
out=dict(kind='isolated-dense-public-timing',index=a.index,elapsed=time.monotonic()-t,timing=r)
Path(a.report).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('cycles','peak_window_power_w','hbm_read_bytes','hbm_write_bytes')}))
