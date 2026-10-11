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
from .estimate import InfeasibleProgramError, estimate_program

ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/"phase-two-model"
STARTER=ROOT/"phase-two-starter"
RUNTIME=MODEL/"runtime/target/release/phase-two-model-runtime"
OFFICIAL=STARTER/"source/target/release/vnext-concurrent"
SCORE_CASES={"w-p":(187689274,1/4),"a-p":(374856358,1/4),
    "w-d1":(27068433,1/6),"w-d16":(42358913,1/6),"w-d4l":(70439292,1/6)}

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

def estimate(program,out):
    out=Path(out);fresh(out)
    report=estimate_program(program)
    report.update(program_sha256=sha(program),numerical_correctness="not_checked",
        optimality_bound=False,official_score=None)
    write(out/"estimate.json",report)
    return report

def estimate_suite(directory,out,grade=None):
    out=Path(out);fresh(out)
    cases={}
    for name,(reference,weight) in SCORE_CASES.items():
        path=Path(directory)/f"{name}.jsonl"
        result=estimate_program(path)
        result["program_sha256"]=sha(path)
        result["reference_cycles"]=reference
        result["weight"]=weight
        cases[name]=result
    log_gain=sum(entry["weight"]*math.log(entry["reference_cycles"]/entry["cycles"])
        for entry in cases.values())
    result={"cases":cases,"score_if_all_eligible":1000*math.exp(log_gain),
        "independent":True,"optimality_bound":False,"numerical_correctness":"not_checked",
        "official_score":None}
    if grade is not None:
        official=json.loads(Path(grade).read_text())
        if official["status"]!="complete":raise ValueError("grade did not complete")
        if set(official["cases"])!=set(cases):raise ValueError("grade cases differ")
        if any(official["cases"][name]["program_sha256"]!=entry["program_sha256"]
               for name,entry in cases.items()):raise ValueError("grade program hashes differ")
        result["official_score"]=official["score"]
        result["score_error_percent"]=(result["score_if_all_eligible"]/official["score"]-1)*100
        for name,entry in cases.items():
            actual=official["cases"][name]["cycles"]
            entry["official_cycles"]=actual
            entry["cycle_error_percent"]=(entry["cycles"]/actual-1)*100
    write(out/"estimate.json",result)
    return result

def set_field(config,key,value):
    parts=key.split(".");obj=config
    for p in parts[:-1]:obj=obj[p]
    if parts[-1] not in obj:raise ValueError(f"unknown design field {key}")
    obj[parts[-1]]=value

def search_estimate_casewise(spec,out,search_started,timeout):
    """Optimize independent per-case generator choices for each shared hardware."""
    cases=spec["cases"]
    domains=spec["domains"]
    case_domains=spec["case_domains"]
    if spec.get("program_base"):
        raise ValueError("case_domains requires export; program_base only supports hardware fields")
    if any(not key.startswith("hardware.") for key in domains):
        raise ValueError("case_domains requires shared hardware domains only")
    if set(case_domains)-{case["name"] for case in cases}:
        raise ValueError("case_domains contains unknown cases")
    for name, local in case_domains.items():
        if any(not (key.startswith("policy.") or key in ("parallel_groups","async_prefetch"))
               for key in local):
            raise ValueError(f"case {name} cannot override shared hardware or workload")
        if any(not values or len({json.dumps(value) for value in values})!=len(values)
               for values in local.values()):
            raise ValueError(f"case {name} has empty or duplicate domain values")
    option_counts={case["name"]:math.prod(map(len,case_domains.get(case["name"],{}).values()))
                   for case in cases}
    hardware_count=math.prod(map(len,domains.values()))
    total=hardware_count*math.prod(option_counts.values())
    if total>spec.get("max_designs",256):
        raise ValueError("declared finite space exceeds max_designs")
    base_hardware=None
    if "hardware_base_program" in spec:
        with (ROOT/spec["hardware_base_program"]).open() as source:
            base_hardware=json.loads(next(source))["hardware"]
        validate(base_hardware)
    candidates=[];rejections=[];estimated_programs=0;feasible_designs=0
    for index,values in enumerate(itertools.product(*domains.values())):
        hardware_design=dict(zip(domains,values))
        directory=out/f"candidate-{index:04d}"
        selected={};options_by_case={};common_hardware=None
        for case in cases:
            name=case["name"]
            local=case_domains.get(name,{})
            options=[]
            for option_index,option_values in enumerate(itertools.product(*local.values())):
                local_design=dict(zip(local,option_values))
                design={**hardware_design,**{f"{name}.{key}":value
                    for key,value in local_design.items()}}
                try:
                    config=copy.deepcopy(json.loads((ROOT/case["config"]).read_text()))
                    if base_hardware is not None:
                        config["hardware"]=copy.deepcopy(base_hardware)
                    for key,value in case.get("overrides",{}).items():
                        set_field(config,key,value)
                    for key,value in hardware_design.items():
                        set_field(config,key,value)
                    for key,value in local_design.items():
                        set_field(config,key,value)
                    validate(config["hardware"])
                    if not 1<=config.get("parallel_groups",1)<=config["hardware"]["resident_groups"]:
                        raise ValueError("parallel_groups exceeds residency")
                    if common_hardware is None:
                        common_hardware=config["hardware"]
                    if config["hardware"]!=common_hardware:
                        raise ValueError("shared hardware constraint")
                except ValueError as error:
                    rejections.append({"design":design,"case":name,"reason":str(error)})
                    continue
                option_dir=directory/"choices"/name/f"option-{option_index:04d}"
                option_dir.mkdir(parents=True,exist_ok=False)
                source=option_dir/"config.json";write(source,config)
                exported=option_dir/"export"
                try:
                    command([OFFICIAL,"export",source,spec.get("seed",7),exported],timeout,
                        option_dir/"export-command.json")
                except RuntimeError:
                    log=json.loads((option_dir/"export-command.json").read_text())
                    rejections.append({"design":design,"case":name,"reason":f"export failed: {log.get('stderr','').strip()[:200]}"})
                    continue
                program=option_dir/"program.jsonl"
                try:
                    command(["python3",STARTER/"tools/compact_program.py",
                        exported/"program.jsonl",program],timeout,option_dir/"compact-command.json")
                except RuntimeError:
                    rejections.append({"design":design,"case":name,"reason":"compact failed"})
                    continue
                (exported/"program.jsonl").unlink()
                started=time.monotonic()
                try:
                    cycles=estimate_program(program)["cycles"]
                except InfeasibleProgramError as error:
                    rejections.append({"design":design,"case":name,"reason":str(error)})
                    continue
                estimated_programs+=1
                options.append({"design":local_design,"cycles":cycles,"program_sha256":sha(program),
                    "program":str(program),"estimate_seconds":time.monotonic()-started})
                write(out/"progress.json",{"completed_hardware":index,"current_hardware":hardware_design,
                    "case":name,"estimated_programs":estimated_programs})
            options_by_case[name]=options
            if not options:
                break
            selected[name]=min(options,key=lambda option:option["cycles"])
        if len(selected)!=len(cases):
            rejections.append({"design":hardware_design,"reason":"no feasible case option"})
            continue
        feasible_designs+=math.prod(len(options_by_case[case["name"]]) for case in cases)
        fresh(directory/"programs")
        write(directory/"hardware.json",common_hardware)
        for name,choice in selected.items():
            shutil.copyfile(choice["program"],directory/"programs"/f"{name}.jsonl")
        predicted={name:choice["cycles"] for name,choice in selected.items()}
        cost=sum(case["weight"]*math.log(predicted[case["name"]]) for case in cases)
        candidates.append({"design":{**hardware_design,**{f"{name}.{key}":value
                for name,choice in selected.items() for key,value in choice["design"].items()}},
            "directory":str(directory),"predicted_cycles":predicted,
            "program_sha256":{name:choice["program_sha256"] for name,choice in selected.items()},
            "predicted_log_cost":cost,"area_au":area(common_hardware),
            "case_options":options_by_case})
        print(f"estimated hardware {index+1}/{hardware_count}: {hardware_design} log_cost={cost:.6f}",flush=True)
    write(out/"rejections.json",rejections)
    if not candidates:raise ValueError("no legal candidate; see rejections.json")
    best=min(candidates,key=lambda candidate:candidate["predicted_log_cost"])
    fresh(out/"best");fresh(out/"best/programs")
    for case in cases:
        name=case["name"]
        program=out/"best/programs"/f"{name}.jsonl"
        shutil.copyfile(Path(best["directory"])/"programs"/f"{name}.jsonl",program)
        if sha(program)!=best["program_sha256"][name]:
            raise RuntimeError(f"selected program changed during search: {name}")
    shutil.copyfile(Path(best["directory"])/"hardware.json",out/"best/hardware.json")
    score=None
    if set(case["name"] for case in cases)==set(SCORE_CASES) and all(
        math.isclose(case["weight"],SCORE_CASES[case["name"]][1]) for case in cases):
        score=1000*math.exp(sum(case["weight"]*math.log(
            SCORE_CASES[case["name"]][0]/best["predicted_cycles"][case["name"]]) for case in cases))
    result={"scope":"exhaustive independent estimates over declared case-separable generator domain",
        "global_optimum_proven":False,"actual_optimum_proven":False,"official_score":None,
        "numerical_correctness":"not_checked","domain_designs":total,
        "evaluated_designs":feasible_designs,"estimated_programs":estimated_programs,
        "rejections":rejections,"candidates":candidates,"best_design":best["design"],
        "best_predicted_cycles":best["predicted_cycles"],
        "best_program_sha256":best["program_sha256"],
        "best_estimated_log_cost":best["predicted_log_cost"],
        "best_score_if_eligible":score,
        "estimator_sha256":sha(Path(__file__).parent/"estimate.py"),
        "target_score":spec.get("target_score"),
        "target_met":score is not None and score>=spec.get("target_score",math.inf),
        "search_wall_seconds":time.monotonic()-search_started}
    write(out/"search.json",result)
    return result

def search_estimate(spec_path,out,timeout=600):
    """Exhaustively rank a declared finite domain without the timing backend."""
    search_started=time.monotonic()
    out=Path(out);fresh(out)
    spec=json.loads(Path(spec_path).read_text());write(out/"spec.json",spec)
    cases=spec["cases"];domains=spec["domains"];keys=list(domains)
    if not cases or any(case["weight"]<=0 for case in cases):raise ValueError("positive case weights required")
    if not math.isclose(sum(case["weight"] for case in cases),1):raise ValueError("weights must sum to one")
    if len({case["name"] for case in cases})!=len(cases):raise ValueError("duplicate case")
    if any(not values or len({json.dumps(v) for v in values})!=len(values)
           for values in domains.values()):raise ValueError("empty or duplicate domain values")
    if "case_domains" in spec:
        return search_estimate_casewise(spec,out,search_started,timeout)
    total=math.prod(map(len,domains.values()))
    if total>spec.get("max_designs",256):raise ValueError("declared finite space exceeds max_designs")
    base_hardware=None
    if "hardware_base_program" in spec:
        with (ROOT/spec["hardware_base_program"]).open() as source:
            base_hardware=json.loads(next(source))["hardware"]
        validate(base_hardware)
    program_base=spec.get("program_base")
    base_program_hashes={}
    if program_base:
        if any(not key.startswith("hardware.") for key in keys) or any(
            case.get("overrides") for case in cases):
            raise ValueError("program_base only supports hardware design fields")
        for case in cases:
            path=ROOT/program_base/f'{case["name"]}.jsonl'
            with path.open() as source:header=json.loads(next(source))
            config=json.loads((ROOT/case["config"]).read_text())
            if any(header[field]!=config[field] for field in ("shape","batch","decode","hardware")):
                raise ValueError(f'program_base mismatch for {case["name"]}')
            base_program_hashes[case["name"]]=sha(path)
    candidates=[];rejections=[]
    for index,values in enumerate(itertools.product(*domains.values())):
        design=dict(zip(keys,values));configs=[]
        try:
            for case in cases:
                config=copy.deepcopy(json.loads((ROOT/case["config"]).read_text()))
                if base_hardware is not None:config["hardware"]=copy.deepcopy(base_hardware)
                for key,value in case.get("overrides",{}).items():set_field(config,key,value)
                for key,value in design.items():set_field(config,key,value)
                validate(config["hardware"])
                if not 1<=config.get("parallel_groups",1)<=config["hardware"]["resident_groups"]:
                    raise ValueError("parallel_groups exceeds residency")
                configs.append(config)
            if any(config["hardware"]!=configs[0]["hardware"] for config in configs):
                raise ValueError("shared hardware constraint")
        except ValueError as error:
            rejections.append({"design":design,"reason":str(error)});continue
        directory=out/f"candidate-{index:04d}";fresh(directory);fresh(directory/"programs")
        write(directory/"hardware.json",configs[0]["hardware"])
        predicted={};hashes={};predict_seconds={}
        for case,config in zip(cases,configs):
            case_started=time.monotonic()
            name=case["name"];source=directory/f"{name}.config.json";write(source,config)
            program=directory/"programs"/f"{name}.jsonl"
            if program_base:
                base_path=ROOT/program_base/f"{name}.jsonl"
                if sha(base_path)!=base_program_hashes[name]:
                    raise RuntimeError(f"base program changed during search: {name}")
                with base_path.open("rb") as reader, program.open("xb") as writer:
                    header=json.loads(reader.readline());header["hardware"]=config["hardware"]
                    writer.write(json.dumps(header,separators=(",",":")).encode()+b"\n")
                    shutil.copyfileobj(reader,writer)
            else:
                exported=directory/f"export-{name}"
                command([OFFICIAL,"export",source,spec.get("seed",7),exported],timeout,
                    directory/f"export-{name}.command.json")
                expanded=exported/"program.jsonl"
                command(["python3",STARTER/"tools/compact_program.py",expanded,program],timeout,
                    directory/f"compact-{name}.command.json")
                expanded.unlink()
            try:
                predicted[name]=estimate_program(program)["cycles"]
            except InfeasibleProgramError as error:
                rejections.append({"design":design,"case":name,"reason":str(error)})
                break
            predict_seconds[name]=time.monotonic()-case_started
            hashes[name]=sha(program)
            write(out/"progress.json",{"completed_designs":len(candidates),"current_design":design,
                "completed_cases":predicted,"case_wall_seconds":predict_seconds})
        if len(predicted)!=len(cases):continue
        cost=sum(case["weight"]*math.log(predicted[case["name"]]) for case in cases)
        candidates.append({"design":design,"directory":str(directory),"predicted_cycles":predicted,
            "program_sha256":hashes,"predicted_log_cost":cost,"area_au":area(configs[0]["hardware"]),
            "case_wall_seconds":predict_seconds})
        write(out/"progress.json",{"completed_designs":len(candidates),"current_design":None})
        print(f"estimated design {index+1}/{total}: {design} log_cost={cost:.6f}",flush=True)
    write(out/"rejections.json",rejections)
    if not candidates:raise ValueError("no legal candidate; see rejections.json")
    best=min(candidates,key=lambda candidate:candidate["predicted_log_cost"])
    fresh(out/"best");fresh(out/"best/programs")
    for case in cases:
        name=case["name"]
        shutil.copyfile(Path(best["directory"])/"programs"/f"{name}.jsonl",
            out/"best/programs"/f"{name}.jsonl")
    shutil.copyfile(Path(best["directory"])/"hardware.json",out/"best/hardware.json")
    score=None
    if set(case["name"] for case in cases)==set(SCORE_CASES) and all(
        math.isclose(case["weight"],SCORE_CASES[case["name"]][1]) for case in cases):
        score=1000*math.exp(sum(case["weight"]*math.log(
            SCORE_CASES[case["name"]][0]/best["predicted_cycles"][case["name"]]) for case in cases))
    scope=("finite header-replay program domain" if program_base else "finite generator domain")
    result={"scope":f"exhaustive independent estimates over declared {scope}",
        "global_optimum_proven":False,"actual_optimum_proven":False,"official_score":None,
        "numerical_correctness":"not_checked","domain_designs":total,
        "evaluated_designs":len(candidates),"rejections":rejections,"candidates":candidates,
        "best_design":best["design"],"best_predicted_cycles":best["predicted_cycles"],
        "best_program_sha256":best["program_sha256"],"best_estimated_log_cost":best["predicted_log_cost"],
        "best_score_if_eligible":score,"estimator_sha256":sha(Path(__file__).parent/"estimate.py"),
        "program_base":program_base,"base_program_sha256":base_program_hashes,
        "target_score":spec.get("target_score"),
        "target_met":score is not None and score>=spec.get("target_score",math.inf),
        "search_wall_seconds":time.monotonic()-search_started}
    write(out/"search.json",result)
    return result

def grade_if_target(search_dir,out,target=5000,seed=7,timeout=600):
    search_dir=Path(search_dir);out=Path(out)
    report=json.loads((search_dir/"search.json").read_text())
    predicted=report.get("best_score_if_eligible")
    if predicted is None or not math.isfinite(predicted) or predicted<target:
        raise ValueError(f"conditional predicted score {predicted} below grade target {target}")
    hashes=report["best_program_sha256"]
    if set(hashes)!=set(SCORE_CASES):raise ValueError("grade requires exactly five official cases")
    programs=search_dir/"best/programs"
    if any(sha(programs/f"{name}.jsonl")!=digest for name,digest in hashes.items()):
        raise ValueError("selected program hashes changed since independent prediction")
    if out.exists():raise FileExistsError(out)
    command(["python3",STARTER/"tools/grade_v09.py",programs,out,
             "--binary",OFFICIAL,"--seed",seed,"--jobs",2,"--cpu-budget",4,
             "--total-memory-gib",8,"--timeout",timeout],timeout*3+60,
            out.with_name(out.name+"-grade-command.json"))
    return json.loads((out/"grade.json").read_text())

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
        estimates=[estimate_program(path)["cycles"] for path in paths] if spec.get("estimate_candidates",False) else None
        lb=sum(case["weight"]*math.log(bound) for case,bound in zip(cases,bounds))
        candidates.append({"directory":directory,"paths":paths,"design":dict(zip(keys,values)),
            "area_au":area(configs[0]["hardware"]),"bounds":bounds,"lb":lb,
            "estimated_cycles":estimates,
            "estimated_log_cost":sum(case["weight"]*math.log(t) for case,t in zip(cases,estimates)) if estimates else None,
            "state":"unmeasured"})
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
        if spec.get("estimate_candidates",False):
            viable=[j for j,candidate in enumerate(candidates)
                if candidate["state"]=="unmeasured" and candidate["lb"]<best-1e-9]
            if viable:
                index=min(viable,key=lambda j:candidates[j]["estimated_log_cost"])
        c=candidates[index];entry={"iteration":it,"design":c["design"],"master":info,
            "analytic_lower_bound_log_cost":c["lb"]};iterations.append(entry)
        if c["estimated_cycles"]:
            entry["independent_estimated_cycles"]=c["estimated_cycles"]
            entry["independent_estimated_log_cost"]=c["estimated_log_cost"]
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
    for mode in ("predict","compare","optimize","model-first","estimate","estimate-suite","search-estimate","grade-if-target"):
        p=sub.add_parser(mode);p.add_argument("input",type=Path);p.add_argument("output",type=Path)
        p.add_argument("--timeout",type=int,default=600)
        if mode=="compare":p.add_argument("--seed",type=int,default=7)
        if mode=="optimize":p.add_argument("--max-evaluations",type=int,default=32)
        if mode=="model-first":
            p.add_argument("--max-verifications",type=int,default=32)
            p.add_argument("--resume",action="store_true")
        if mode=="estimate-suite":p.add_argument("--grade",type=Path)
        if mode=="grade-if-target":
            p.add_argument("--target-score",type=float,default=5000)
            p.add_argument("--seed",type=int,default=7)
    a=ap.parse_args()
    try:
        if a.mode=="predict":r=predict(a.input,a.output,a.timeout)
        elif a.mode=="compare":r=compare(a.input,a.output,a.seed,a.timeout)
        elif a.mode=="estimate":r=estimate(a.input,a.output)
        elif a.mode=="estimate-suite":r=estimate_suite(a.input,a.output,a.grade)
        elif a.mode=="search-estimate":r=search_estimate(a.input,a.output,a.timeout)
        elif a.mode=="model-first":
            from .structured import solve
            r=solve(a.input,a.output,a.timeout,a.max_verifications,resume=a.resume)
        elif a.mode=="grade-if-target":r=grade_if_target(a.input,a.output,a.target_score,a.seed,a.timeout)
        else:r=optimize(a.input,a.output,a.timeout,a.max_evaluations)
    except Exception as error:
        if a.output.exists() and not (a.output/"failure.json").exists():
            write(a.output/"failure.json",{"status":"incomplete","error":str(error),
                "mode":a.mode,"input":str(a.input),"has_official_five_case_score":False})
        raise
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
