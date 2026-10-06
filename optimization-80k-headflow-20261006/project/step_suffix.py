"""Exact timing of P1's post-commit suffix with resident RF data assumed ready.

Not a full score. Whole-program functional checks must validate retained data.
"""
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
lines=Path(a.program).read_text().splitlines();begin=next(i for i,l in enumerate(lines) if l.startswith('STEP.COMMIT '))+1
program='\n'.join([l for l in lines[:begin] if l.startswith('WG.BEGIN ')]+lines[begin:])+'\n'
h=Hardware.from_dict(json.loads(Path(a.hardware).read_text()));t=time.monotonic()
r=asdict(estimate_pipeline(h,iter_parse(program),required_hbm_words(program,build_layout(MODELS['M1'],'P1')),None))
Path(a.report).write_text(json.dumps(dict(kind='p1-post-commit-suffix-only',score=None,elapsed=time.monotonic()-t,timing=r),indent=2)+'\n')
print(json.dumps({k:r[k] for k in ('cycles','peak_window_power_w','hbm_read_bytes')}))
