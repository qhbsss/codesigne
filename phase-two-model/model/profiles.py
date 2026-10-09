"""Independent Python translation of paid computation microstep equations.

This is not a runtime scheduler. Its resource work gives safe lower bounds,
while the exact backend determines arbitration, communication and power.
"""
from .hardware import validate

def step(**kw):
    s={k:0 for k in ("rf_read_bytes","rf_write_bytes","sh_read_bytes","sh_write_bytes",
        "sh_read_cycles","sh_write_cycles","sh_tc_bytes","core_duration","physical_fmas")}
    s.update(core_energy_pj=0.0,sh_read_service=[],sh_write_service=[]);s.update(kw);return s

def bank_activity(words,h,write=False):
    if not h["sh_banks"]: raise ValueError("SH disabled")
    counts=[0]*h["sh_banks"]
    for w in (words if write else list(dict.fromkeys(words))): counts[w%len(counts)]+=1
    ports=1 if write else h["shared_ports"]
    return [sum(min(max(n-c*ports,0),ports)*4 for n in counts)
            for c in range((max(counts)+ports-1)//ports)]

def profile(i,h):
    op=i["op"];steps=[];engine="Vector"
    if op in ("fill","vector","reduce") and not 1<=i["len"]<=8192:
        raise ValueError("compute length")
    if op=="fill":
        steps=[step(rf_write_bytes=4*min(h["vector"],i["len"]-s))
               for s in range(0,i["len"],h["vector"])]
    elif op=="vector":
        kind=i["kind"];unary=kind in ("exp","tanh","rsqrt","square")
        sfu=kind in ("exp","tanh","rsqrt");engine="Sfu" if sfu else "Vector"
        width=h["sfu"] if sfu else h["vector"]
        for start in range(0,i["len"],width):
            count=min(width,i["len"]-start)
            def rb(arg): return 0 if arg["kind"]=="imm" else 4 if arg["stride"]==0 else 4*count
            steps.append(step(rf_read_bytes=rb(i["a"])+(0 if unary else rb(i["b"]))+(4*count if kind=="fma" else 0),
                rf_write_bytes=4*count,core_duration=12 if sfu else 4,
                core_energy_pj=count*{"exp":12,"tanh":12,"rsqrt":8,"fma":4}.get(kind,1.5)))
    elif op=="reduce":
        engine="Reduction" if h["reduction_units"] else "Vector";n=i["len"]
        if n==1:steps.append(step(rf_read_bytes=4,rf_write_bytes=4))
        while n>1:
            out=(n+1)//2
            for start in range(0,out,h["vector"]):
                count=min(out-start,h["vector"]);inputs=min(n-2*start,2*count)
                steps.append(step(rf_read_bytes=4*inputs,rf_write_bytes=4*count,
                    core_duration=1 if h["reduction_units"] else 4,core_energy_pj=(inputs//2)*1.5))
            n=out
    elif op in ("shared_read","shared_write"):
        engine="Copy";view=i["shared"];write=op=="shared_write"
        total=view["rows"]*view["cols"]
        for start in range(0,total,16):
            count=min(16,total-start)
            words=[view["base"]+(k//view["cols"])*view["row_stride"]+(k%view["cols"])*view["col_stride"] for k in range(start,start+count)]
            service=bank_activity(words,h,write);kw=dict(core_duration=6 if start==0 else 0,
                core_energy_pj=count*.5+(20 if start==0 else 0))
            if write:kw.update(rf_read_bytes=4*count,sh_write_bytes=sum(service),sh_write_cycles=len(service),sh_write_service=service)
            else:kw.update(rf_write_bytes=4*count,sh_read_bytes=sum(service),sh_read_cycles=len(service),sh_read_service=service)
            steps.append(step(**kw))
    elif op in ("mma","mma_shared"):
        if not h["tc_count"]:raise ValueError("TC disabled")
        engine="Tc";m,n,k=i["m"],i["n"],i["k"];p,q,kp=h["p"],h["q"],h["tc_k_parallel"]
        if not (1<=m<=64 and 1<=n<=64 and 1<=k<=256):raise ValueError("MMA limits")
        if max(m*k,k*n,m*n)>16384:raise ValueError("MMA operand size")
        if op=="mma_shared" and not h["sh_tc_bw"]:raise ValueError("SH-TC disabled")
        for row in range(0,m,p):
            for col in range(0,n,q):
                mm,nn=min(p,m-row),min(q,n-col)
                steps.append(step(rf_read_bytes=4*mm*nn))
                for kk in range(0,k,kp):
                    active=min(kp,k-kk);physical=p*q*kp
                    st=step(rf_read_bytes=4*mm*active,core_duration=1+kp.bit_length()-1,
                        physical_fmas=physical,core_energy_pj=physical*3+p*q*(kp-1)*1.5)
                    if op=="mma_shared":
                        v=i["b"];words=[v["base"]+t*v["row_stride"]+j*v["col_stride"] for t in range(kk,kk+active) for j in range(col,col+nn)]
                        service=[]
                        for s in range(0,len(words),16):service.extend(bank_activity(words[s:s+16],h))
                        st.update(sh_read_service=service,sh_read_bytes=sum(service),sh_read_cycles=len(service),sh_tc_bytes=sum(service))
                    else:st["rf_read_bytes"]+=4*nn*active
                    steps.append(st)
                steps.append(step(rf_write_bytes=4*mm*nn,core_duration=p+q-2))
    else: raise ValueError("DMA belongs to memory transition backend")
    totals={k:sum(s[k] for s in steps) for k in ("rf_read_bytes","rf_write_bytes","sh_read_bytes","sh_write_bytes",
        "sh_read_cycles","sh_write_cycles","sh_tc_bytes","core_duration","core_energy_pj","physical_fmas")}
    return dict(engine=engine,steps=steps,**totals)

def lower_bound_plain(path):
    """A conservative lower bound from expanded paid primitives, not timings.

    Core occupancy excludes operand service/delays. Per-SM mandatory issue work
    is accumulated per wave; one final fence per wave/commit is unavoidable.
    """
    import json
    with open(path) as f:
        header=json.loads(next(f));h=header["hardware"];validate(h)
        core={k:0 for k in ("Tc","Vector","Sfu","Reduction")};issue=0
        for line in f:
            r=json.loads(line)
            if r["record"]=="wave":
                issues=[0]*h["sms"]
                for g in r["groups"]:
                    issues[g["sm"]]+=len(g["commands"])
                    for cmd in g["commands"]:
                        if cmd["command"]=="wait":continue
                        i=cmd["instruction"]
                        if i["op"] in ("load","store","load_shared","store_shared"):continue
                        p=profile(i,h)
                        if p["engine"] in core:core[p["engine"]]+=p["core_duration"]
                issue+=max(issues,default=0)+1
            elif r["record"]=="commit":issue+=1
            elif r["record"] not in ("input","alloc","release"):raise ValueError("bound requires expanded exporter output")
    caps={"Tc":h["sms"]*h["tc_count"],"Vector":h["sms"],"Sfu":h["sms"],"Reduction":h["sms"]*h["reduction_units"]}
    return max([issue,1]+[(v+caps[k]-1)//caps[k] for k,v in core.items() if v and caps[k]])
