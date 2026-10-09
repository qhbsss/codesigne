"""Pure analytical MILP for a declared compute-only GEMM implementation space.

Unlike the full-program optimizer, no runtime calls supply objective costs.
The independent microstep/serial equations compute each alternative exactly.
"""
import argparse
import copy
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import lil_matrix
from .hardware import area,validate
from .serial import serial_wave
from .tool import STARTER,RUNTIME,command,fresh,write

def commands(k_tile):
    # Same M8*N16*K32 GEMM for all choices, constant operands exactly .5.
    # A/B tiles are explicitly filled, not free reinterpreted strided views.
    m,n,k=8,16,32;items=[dict(op="fill",dst=2048,len=m*n,value=0.0)]
    for kk in range(0,k,k_tile):
        kt=min(k_tile,k-kk)
        items.extend([dict(op="fill",dst=0,len=m*kt,value=.5),
            dict(op="fill",dst=1024,len=kt*n,value=.5),
            dict(op="mma",a=0,b=1024,c=2048,m=m,n=n,k=kt)])
    return items

def solve(out):
    out=Path(out);fresh(out)
    base=json.loads((STARTER/"examples/small-config.json").read_text())["hardware"]
    base.update(sms=2,rf_kib=32,sh_kib=0,sh_banks=0,sh_tc_bw=0,
        sfu=1,vector=32,tc_count=1,resident_groups=1,dma_depth=0,
        hbm_channels=1,hbm_queue=32,sm_noc=16,global_noc=64)
    domains={"rf_kib":[32,64],"rf_tier":[1,2],"p":[4,8],"q":[8,16],
        "tc_k_parallel":[1,2,4],"k_tile":[8,16,32]}
    keys=list(domains);choices=[]
    for values in itertools.product(*domains.values()):
        d=dict(zip(keys,values));h=copy.deepcopy(base)
        for key,value in d.items():
            if key!="k_tile":h[key]=value
        try:validate(h)
        except ValueError:continue
        prediction=serial_wave(commands(d["k_tile"]),h)
        if prediction["short_power_w"]>34 or prediction["long_power_w"]>26:continue
        choices.append(dict(design=d,hardware=h,prediction=prediction,area_au=area(h)))
    n=len(choices);factors=[(k,v) for k,values in domains.items() for v in values]
    p=len(factors);rows=2+len(domains)+p
    A=lil_matrix((rows,n+p));lo=np.zeros(rows);hi=np.zeros(rows)
    A[0,:n]=1;lo[0]=hi[0]=1
    A[1,:n]=[c["area_au"] for c in choices];lo[1]=-np.inf;hi[1]=100
    row=2
    for key in domains:
        for j,(k,v) in enumerate(factors):
            if k==key:A[row,n+j]=1
        lo[row]=hi[row]=1;row+=1
    for j,(key,value) in enumerate(factors):
        A[row,n+j]=1
        for i,c in enumerate(choices):
            if c["design"][key]==value:A[row,i]=-1
        row+=1
    objective=np.zeros(n+p);objective[:n]=[c["prediction"]["cycles"] for c in choices]
    r=milp(objective,integrality=np.ones(n+p),bounds=Bounds(np.zeros(n+p),np.ones(n+p)),
        constraints=LinearConstraint(A.tocsr(),lo,hi),options={"mip_rel_gap":0.0})
    if not r.success:raise RuntimeError(r.message)
    c=choices[int(np.argmax(r.x[:n]))]
    payload=dict(hardware=c["hardware"],groups=[dict(sm=0,rf_kib=16,sh_kib=0,
        commands=[dict(command="run",instruction=i) for i in commands(c["design"]["k_tile"])])])
    write(out/"kernel-wave.json",payload);write(out/"hardware.json",c["hardware"])
    # A posteriori differential test only. Runtime never entered the MILP costs.
    command([RUNTIME,"wave",out/"kernel-wave.json",out/"runtime.json"],60,out/"runtime-command.json")
    official=json.loads((out/"runtime.json").read_text())["result"]["report"]
    differences=[]
    for field in ["cycles","rf_bytes","sh_bytes","physical_fmas"]:
        if c["prediction"][field]!=official["stats"][field]:differences.append(field)
    for field in ["short_power_w","long_power_w","energy_j"]:
        if abs(c["prediction"][field]-official[field])>max(1e-12,abs(official[field])*1e-11):differences.append(field)
    result={"scope":"compute-only M8 N16 K32 constant GEMM, one group; not a Transformer submission",
        "objective_source":"independent mathematical microstep equations; no runtime timings",
        "candidate_count":n,"variables":n+p,"constraints":rows,
        "solver_status":r.message,"mip_gap":float(r.mip_gap),
        "dual_bound_cycles":float(r.mip_dual_bound),"chosen":c,
        "runtime_differences":differences,
        "all_modeled_metrics_match":not differences,
        "numeric_scope":"constant .5 operands, exact real output 8; general input correctness not proved",
        "source_sha256":json.loads((out/"runtime.json").read_text())["source_sha256"]}
    write(out/"solution.json",result)
    if differences:raise RuntimeError("analytical model differs from runtime")
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("output",type=Path);a=p.parse_args()
    print(json.dumps(solve(a.output),indent=2))
if __name__=="__main__":main()
