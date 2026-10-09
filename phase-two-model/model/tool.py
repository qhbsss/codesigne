"""CLI, exact/functional comparison and finite-domain MIP decomposition."""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from .hardware import area, validate
from .profiles import lower_bound_plain

ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/"phase-two-model"
STARTER=ROOT/"phase-two-starter"
RUNTIME=MODEL/"runtime/target/release/phase-two-model-runtime"
OFFICIAL=STARTER/"source/target/release/vnext-concurrent"

def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fresh(path):Path(path).mkdir(parents=True,exist_ok=False)

def command(args,timeout,log):
    """Persist failures/timeouts, never quietly treat execution errors as infeasible."""
    start=time.monotonic()
    try:
        r=subprocess.run([str(x) for x in args],capture_output=True,text=True,timeout=timeout)
        data={"argv":[str(x) for x in args],"seconds":time.monotonic()-start,
              "returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
    except subprocess.TimeoutExpired as e:
        data={"argv":[str(x) for x in args],"seconds":time.monotonic()-start,
              "timeout":True,"stdout":str(e.stdout or ""),"stderr":str(e.stderr or "")}
    write(log,data)
    if data.get("returncode")!=0:raise RuntimeError(f"command failed or timed out; retained {log}")

def predict(program,out,timeout=600):
    program=Path(program).resolve();out=Path(out);fresh(out)
    before=sha(program)
    command([RUNTIME,"predict",program,out/"prediction.json"],timeout,out/"predict-command.json")
    p=json.loads((out/"prediction.json").read_text())
    if before!=sha(program) or p["result"]["program_sha256"]!=before:
        raise RuntimeError("program changed during prediction")
    return p

def compare(program,out,seed=7,timeout=600):
    """Exact shared timing backend vs full official numerical execution."""
    out=Path(out);fresh(out);program=Path(program).resolve();before=sha(program)
    p=predict(program,out/"model",timeout)
    command([OFFICIAL,"evaluate",program,seed,out/"official"],timeout,out/"official-command.json")
    o=json.loads((out/"official/result.json").read_text())
    if o["status"]!="complete":raise RuntimeError("official evaluation incomplete")
    result=compare_reports(p,o,before)
    if before!=sha(program):raise RuntimeError("program changed during comparison")
    write(out/"comparison.json",result)
    return result

def compare_reports(p,o,program_hash):
    if p["source_sha256"]!=o["source_sha256"]:raise RuntimeError("runtime/official source hashes differ")
    a,b=p["result"],o["evaluation"]
    if a["program_sha256"]!=program_hash or b["program_sha256"]!=program_hash:
        raise RuntimeError("program hash mismatch")
    diffs=[]
    def walk(x,y,path):
        if isinstance(x,dict) and isinstance(y,dict):
            if set(x)!=set(y):diffs.append({"path":path,"different_keys":True})
            for k in sorted(set(x)&set(y)):walk(x[k],y[k],path+"."+k)
        elif x!=y:diffs.append({"path":path,"model":x,"official":y})
    walk(a["report"],b["report"],"report")
    for field in ("header","step_cycles","hbm_peak_allocated_bytes","work_units"):
        walk(a[field],b[field],field)
    exact=not diffs
    return {"program_sha256":program_hash,"source_sha256":p["source_sha256"],
        "backend":p["backend"],"independent_runtime_validation":False,
        "all_timing_report_fields_equal":exact,"differences":diffs,
        "cycles":b["report"]["stats"]["cycles"],
        "cycle_gap":a["report"]["stats"]["cycles"]-b["report"]["stats"]["cycles"],
        "comparison":b["comparison"],"official_numerical_pass":True,
        "prediction_seconds":p["host_seconds"],"official_seconds":o["host_seconds"],
        "note":"Timing equality uses the same immutable transition core. FP32 correctness comes only from the separate full evaluator."}

def set_field(config,key,value):
    parts=key.split(".");obj=config
    for p in parts[:-1]:obj=obj[p]
    if parts[-1] not in obj:raise ValueError(f"unknown design field {key}")
    obj[parts[-1]]=value

def prepare(spec,out,timeout):
    """Generate unpaid design alternatives and analytical bounds BEFORE timing.

    Tuple lifting is exact for a finite declared domain. No premeasured scores
    enter the initial MILP; each factor remains a named discrete decision.
    """
    cases=spec["cases"];keys=list(spec["domains"]);domains=list(spec["domains"].values())
    if any(not d for d in domains):raise ValueError("empty domain")
    if any(len({json.dumps(v) for v in d})!=len(d) for d in domains):raise ValueError("duplicate domain values")
    total=math.prod(map(len,domains))
    if total>spec.get("max_designs",256):raise ValueError("declared finite space exceeds max_designs; decompose it")
    candidates=[];rejections=[]
    for idx,values in enumerate(itertools.product(*domains)):
        configs=[]
        try:
            for case in cases:
                config=copy.deepcopy(json.loads((ROOT/case["config"]).read_text()))
                for k,v in zip(keys,values):set_field(config,k,v)
                validate(config["hardware"])
                if not 1<=config.get("parallel_groups",1)<=config["hardware"]["resident_groups"]:
                    raise ValueError("parallel_groups exceeds residency")
                configs.append(config)
            if any(c["hardware"]!=configs[0]["hardware"] for c in configs):
                raise ValueError("five-case shared hardware constraint")
        except ValueError as e:
            rejections.append({"design":dict(zip(keys,values)),"reason":str(e),"kind":"analytic_infeasibility"});continue
        directory=out/f"design-{idx:04d}";fresh(directory)
        write(directory/"hardware.json",configs[0]["hardware"])
        paths=[];bounds=[]
        for case,c in zip(cases,configs):
            name=case["name"];cp=directory/f"{name}.config.json";write(cp,c)
            command([OFFICIAL,"export",cp,0,directory/f"export-{name}"],timeout,directory/f"export-{name}.command.json")
            plain=directory/f"export-{name}/program.jsonl"
            bounds.append(lower_bound_plain(plain))
            compact=directory/f"{name}.jsonl"
            command(["python3",STARTER/"tools/compact_program.py",plain,compact],timeout,directory/f"compact-{name}.command.json")
            paths.append(compact)
        lb=sum(case["weight"]*math.log(bound) for case,bound in zip(cases,bounds))
        candidates.append({"directory":directory,"paths":paths,"design":dict(zip(keys,values)),
            "area_au":area(configs[0]["hardware"]),"bounds":bounds,"lb":lb,"state":"unmeasured"})
    write(out/"analytic-rejections.json",rejections)
    return candidates

def solve_master(candidates,domains,solver_seconds):
    import numpy as np
    from scipy.optimize import Bounds,LinearConstraint,milp
    from scipy.sparse import lil_matrix
    # x(field,value) + tuple selectors q: exact extended finite-domain formulation.
    factors=[(k,v) for k,values in domains.items() for v in values]
    n=len(candidates);p=len(factors)
    objective=np.zeros(n+p)
    objective[:n]=[c["lb"] for c in candidates]
    lower=np.zeros(n+p);upper=np.ones(n+p)
    for i,c in enumerate(candidates):
        if c["state"]=="infeasible":upper[i]=0
    rows=2+len(domains)+p;A=lil_matrix((rows,n+p));lo=np.zeros(rows);hi=np.zeros(rows)
    A[0,:n]=1;lo[0]=hi[0]=1
    A[1,:n]=[c["area_au"] for c in candidates];lo[1]=-np.inf;hi[1]=100
    row=2
    for key in domains:
        for f,(k,v) in enumerate(factors):
            if k==key:A[row,n+f]=1
        lo[row]=hi[row]=1;row+=1
    for f,(key,value) in enumerate(factors):
        A[row,n+f]=1
        for i,c in enumerate(candidates):
            if c["design"][key]==value:A[row,i]=-1
        row+=1
    r=milp(objective,integrality=np.ones(n+p),bounds=Bounds(lower,upper),
        constraints=LinearConstraint(A.tocsr(),lo,hi),
        options={"time_limit":solver_seconds,"mip_rel_gap":0.0})
    info={"status":int(r.status),"message":r.message,"variables":n+p,"constraints":rows,
        "mip_gap":float(r.mip_gap) if getattr(r,"mip_gap",None) is not None else None,
        "dual_bound":float(r.mip_dual_bound) if getattr(r,"mip_dual_bound",None) is not None else None}
    if r.x is None:return None,info
    return int(np.argmax(r.x[:n])),info

def optimize(spec_path,out,timeout=600,max_evaluations=32):
    out=Path(out);fresh(out);start=time.monotonic()
    spec=json.loads(Path(spec_path).read_text());write(out/"spec.json",spec)
    if not spec["cases"] or any(c["weight"]<=0 for c in spec["cases"]):raise ValueError("positive case weights required")
    if not math.isclose(sum(c["weight"] for c in spec["cases"]),1):raise ValueError("weights must sum to one")
    candidates=prepare(spec,out,timeout)
    if not candidates:raise ValueError("no legal designs")
    incumbent=None;best=math.inf;iterations=[];termination="evaluation_budget";dual=None
    for it in range(max_evaluations+1):
        index,info=solve_master(candidates,spec["domains"],spec.get("solver_seconds",30))
        dual=info["dual_bound"]
        if index is None:
            termination="no_feasible_design" if info["status"]==2 else "solver_incomplete";break
        if info["status"]!=0:termination="solver_incomplete";break
        if incumbent is not None and dual is not None and dual>=best-1e-9:
            termination="optimal_within_declared_space_and_tolerance";break
        if it==max_evaluations:break
        c=candidates[index];entry={"iteration":it,"design":c["design"],"master":info,
            "analytic_lower_bound_log_cost":c["lb"]};iterations.append(entry)
        write(out/"iterations.json",iterations)
        exact=[];snapshots=[];source=None
        for name,path in zip([x["name"] for x in spec["cases"]],c["paths"]):
            p=predict(path,c["directory"]/f"prediction-{name}",timeout)
            if source is None:source=p["source_sha256"]
            if source!=p["source_sha256"]:raise RuntimeError("source changed during optimization")
            v=p["result"];cycles=v["report"]["stats"]["cycles"]
            if v["lower_bound_cycles"]>cycles:raise RuntimeError("invalid runtime lower bound")
            exact.append(cycles);snapshots.append(p)
        if any(lb>t for lb,t in zip(c["bounds"],exact)):raise RuntimeError("independent lower bound exceeds exact runtime")
        cost=sum(case["weight"]*math.log(t) for case,t in zip(spec["cases"],exact))
        c["lb"]=cost;c["state"]="timed";entry["exact_cycles"]=exact;entry["exact_log_cost"]=cost
        if any(not p["result"]["report"]["power_pass"] for p in snapshots):
            c["state"]="infeasible";entry["rejection"]="exact_power_constraint";continue
        if cost<best-1e-9:
            passed=True
            for case,path,p in zip(spec["cases"],c["paths"],snapshots):
                name=case["name"];official=c["directory"]/f"verify-{name}"
                try:
                    command([OFFICIAL,"evaluate",path,spec.get("seed",7),official],timeout,c["directory"]/f"verify-{name}.command.json")
                except RuntimeError:
                    report_path=official/"result.json"
                    # Only a known semantic rejection is an infeasibility cut.
                    if report_path.exists():
                        error=json.loads(report_path.read_text()).get("error") or ""
                        if any(x in error for x in ("mismatch","nonfinite","uninitialized","power constraint")):
                            passed=False;entry["rejection"]=error;break
                    raise
                o=json.loads((official/"result.json").read_text())
                diff=compare_reports(p,o,sha(path));write(c["directory"]/f"comparison-{name}.json",diff)
                if not diff["all_timing_report_fields_equal"]:raise RuntimeError("runtime/full evaluation discrepancy")
            if not passed:c["state"]="infeasible";continue
            incumbent=index;best=cost;entry["new_incumbent"]=True
        write(out/"iterations.json",iterations)
    result={"scope":"finite_declared_generator_space; not unrestricted ISA optimality",
        "timing_backend":"shared_official_transition_core; not independent simulator",
        "termination":termination,"evaluated_designs":len(iterations),"legal_designs":len(candidates),
        "model_log_lower_bound":dual,"incumbent_log_cost":best if incumbent is not None else None,
        "log_gap":max(0,best-dual) if incumbent is not None and dual is not None else None,
        "numerical_scope":f"official attention_stress public seed {spec.get('seed',7)}; no universal proof",
        "host_seconds":time.monotonic()-start,"iterations":iterations,
        "has_official_five_case_score":False}
    if incumbent is not None:
        c=candidates[incumbent];fresh(out/"best");fresh(out/"best/programs")
        shutil.copyfile(c["directory"]/"hardware.json",out/"best/hardware.json")
        for case,path in zip(spec["cases"],c["paths"]):
            shutil.copyfile(path,out/"best/programs"/f"{case['name']}.jsonl")
            shutil.copyfile(c["directory"]/f"{case['name']}.config.json",out/"best"/f"{case['name']}.config.json")
        result["best_design"]=c["design"]
        result["best_program_hashes"]={case["name"]:sha(path) for case,path in zip(spec["cases"],c["paths"])}
        result["source_sha256"]=json.loads((c["directory"]/f"prediction-{spec['cases'][0]['name']}/prediction.json").read_text())["source_sha256"]
        result["official_evidence_directory"]=str(c["directory"])
    write(out/"iterations.json",iterations);write(out/"solution.json",result)
    return result

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="mode",required=True)
    for mode in ("predict","compare","optimize"):
        p=sub.add_parser(mode);p.add_argument("input",type=Path);p.add_argument("output",type=Path)
        p.add_argument("--timeout",type=int,default=600)
        if mode=="compare":p.add_argument("--seed",type=int,default=7)
        if mode=="optimize":p.add_argument("--max-evaluations",type=int,default=32)
    a=ap.parse_args()
    try:
        if a.mode=="predict":r=predict(a.input,a.output,a.timeout)
        elif a.mode=="compare":r=compare(a.input,a.output,a.seed,a.timeout)
        else:r=optimize(a.input,a.output,a.timeout,a.max_evaluations)
    except Exception as error:
        if a.output.exists() and not (a.output/"failure.json").exists():
            write(a.output/"failure.json",{"status":"incomplete","error":str(error),
                "mode":a.mode,"input":str(a.input),"has_official_five_case_score":False})
        raise
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
