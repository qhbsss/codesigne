"""Exact public timing for a prefix, without asserting case output coverage."""
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
from codesign.challenge.hardware import Hardware
from codesign.challenge.abi import build_layout
from codesign.challenge.workload import MODELS
from codesign.challenge.isa import iter_parse
from codesign.challenge.pipeline import estimate_pipeline
from codesign.challenge.runner import required_hbm_words
p=argparse.ArgumentParser();p.add_argument('--program',required=True);p.add_argument('--hardware',required=True);p.add_argument('--barriers',type=int,required=True);p.add_argument('--report',required=True);a=p.parse_args()
lines=Path(a.program).read_text().splitlines();selected=[];count=0
for line in lines:
 selected.append(line)
 if line.startswith('BARRIER '):
  count+=1
  if count==a.barriers:break
for line in lines:
 if line.startswith('WG.END '):selected.append(line)
program='\n'.join(selected)+'\n';started=time.monotonic();h=Hardware.from_dict(json.loads(Path(a.hardware).read_text()));layout=build_layout(MODELS['M1'],'P1')
timing=asdict(estimate_pipeline(h,iter_parse(program),required_hbm_words(program,layout),None));report=dict(kind='exact-prefix-probe',barriers=a.barriers,elapsed=time.monotonic()-started,timing=timing)
Path(a.report).write_text(json.dumps(report,indent=2));print(json.dumps({k:timing[k] for k in ['cycles','peak_window_power_w','hbm_read_bytes','cache_hits','cache_misses']}))
