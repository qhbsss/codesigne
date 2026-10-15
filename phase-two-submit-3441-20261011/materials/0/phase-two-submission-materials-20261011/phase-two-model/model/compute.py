"""Independent fixed-arbitration transition model for compute-only waves.

Models several groups/SMs, shared RF/SH ports and multiple engines. No DMA,
tokens, HBM, cache or NoC. Explicitly rejects those unsupported commands.
"""
from math import log2
from .hardware import validate,rf_latency,sh_latency,rf_energy
from .profiles import profile
from .serial import energy_report

def compute_wave(groups,h):
    validate(h);n=h["sms"]
    if len(groups)>min(256,n*h["resident_groups"]):raise ValueError("group capacity")
    for s in range(n):
        on=[g for g in groups if g["sm"]==s]
        if (len(on)>h["resident_groups"] or sum(g["rf_kib"] for g in on)>h["rf_kib"]
                or sum(g["sh_kib"] for g in on)>h["sh_kib"]):raise ValueError("local capacity")
    for g in groups:
        if not 0<=g["sm"]<n or g["rf_kib"]<1 or g["sh_kib"]%4:raise ValueError("invalid group")
        if any(c["command"]!="run" or c["instruction"]["op"] in ("load","store","load_shared","store_shared") for c in g["commands"]):
            raise ValueError("independent compute model does not support DMA/wait")
    profiles=[[profile(c["instruction"],h) for c in g["commands"]] for g in groups]
    capacity={"Vector":1,"Sfu":1,"Reduction":h["reduction_units"],"Tc":h["tc_count"],"Copy":h["dma_engines"]}
    busy=[{k:[False]*v for k,v in capacity.items()} for _ in range(n)]
    ports=[dict(read=0,write=0,sh_read=0,sh_write=0,interface=0,issue=0) for _ in range(n)]
    states=[dict(pc=0,ready=0,run=None) for _ in groups]
    t=round_=0;events=[];rf_bytes=sh_bytes=fmas=0;iterations=0
    rb=[128,256,512][h["rf_tier"]-1];wb=rb//2
    sh_e=.8*(1+.05*log2(h["sh_kib"]/64)) if h["sh_kib"] else 0
    def add(start,duration,energy):
        if duration:events.append((start,duration,energy))
    def charge(start,count,bw,energy):
        full,rem=divmod(count,bw)
        if full:add(start,full,full*bw*energy)
        if rem:add(start+full,1,rem*energy)
    while True:
        iterations+=1
        if iterations>200000000:raise ValueError("iteration budget")
        for off in range(len(groups)):
            gi=(off+round_)%len(groups);g=groups[gi];s=states[gi];p=ports[g["sm"]]
            if s["ready"]>t:continue
            while s["run"] is not None:
                run=s["run"];prof=profiles[gi][s["pc"]]
                if run["step"]==len(prof["steps"]):
                    busy[g["sm"]][prof["engine"]][run["engine"]]=False
                    s.update(pc=s["pc"]+1,ready=t,run=None);break
                st=prof["steps"][run["step"]];progress=False
                if run["phase"]==0:
                    ready=max(p["read"] if st["rf_read_bytes"] else 0,
                        p["sh_read"] if st["sh_read_bytes"] else 0,
                        p["interface"] if st["sh_tc_bytes"] else 0)
                    if ready<=t:
                        delay=0
                        if st["rf_read_bytes"]:
                            count=st["rf_read_bytes"];service=(count+rb-1)//rb
                            p["read"]=t+service;delay=max(delay,service+rf_latency(h))
                            charge(t,count,rb,rf_energy(h));rf_bytes+=count
                        if st["sh_read_bytes"]:
                            service=max(st["sh_read_cycles"],1);p["sh_read"]=t+service
                            delay=max(delay,service+sh_latency(h));sh_bytes+=st["sh_read_bytes"]
                            for j,count in enumerate(st["sh_read_service"]):add(t+j,1,count*sh_e*(1.15 if h["shared_ports"]==2 else 1))
                        if st["sh_tc_bytes"]:
                            start=t+st["sh_read_cycles"]+sh_latency(h)
                            service=(st["sh_tc_bytes"]+h["sh_tc_bw"]-1)//h["sh_tc_bw"]
                            p["interface"]=start+service;delay=max(delay,start+service+2-t)
                            charge(start,st["sh_tc_bytes"],h["sh_tc_bw"],.2)
                        s["ready"]=t+delay;progress=True
                elif run["phase"]==1:
                    if st["core_duration"]:
                        add(t,st["core_duration"],st["core_energy_pj"]);s["ready"]=t+st["core_duration"]
                    progress=True
                elif run["phase"]==2:
                    if (not st["rf_write_bytes"] or p["write"]<=t) and (not st["sh_write_bytes"] or p["sh_write"]<=t):
                        delay=0
                        if st["rf_write_bytes"]:
                            count=st["rf_write_bytes"];service=(count+wb-1)//wb
                            p["write"]=t+service;delay=max(delay,service+rf_latency(h));rf_bytes+=count
                            charge(t,count,wb,rf_energy(h))
                        if st["sh_write_bytes"]:
                            service=max(st["sh_write_cycles"],1);p["sh_write"]=t+service
                            delay=max(delay,service+sh_latency(h));sh_bytes+=st["sh_write_bytes"]
                            for j,count in enumerate(st["sh_write_service"]):add(t+j,1,count*sh_e)
                        s["ready"]=t+delay;progress=True
                else:run["step"]+=1;run["phase"]=0;continue
                if progress:run["phase"]+=1
                if not progress or s["ready"]>t:break
            if s["run"] is not None or s["pc"]==len(profiles[gi]):continue
            if p["issue"]>t:continue
            prof=profiles[gi][s["pc"]];engines=busy[g["sm"]][prof["engine"]]
            if False not in engines:continue
            engine=engines.index(False);engines[engine]=True
            fmas+=prof["physical_fmas"];s.update(run=dict(step=0,phase=0,engine=engine),ready=t+1)
            p["issue"]=t+1;add(t,1,8)
        round_=(round_+1)%max(len(groups),1)
        if all(s["pc"]==len(profiles[i]) and s["run"] is None for i,s in enumerate(states)):break
        next_=None
        def future(value):
            nonlocal next_
            value=max(value,t+1);next_=value if next_ is None else min(next_,value)
        for gi,s in enumerate(states):
            if s["pc"]==len(profiles[gi]) and s["run"] is None:continue
            if s["ready"]>t:future(s["ready"]);continue
            sm=groups[gi]["sm"];p=ports[sm];prof=profiles[gi][s["pc"]]
            if s["run"] is not None:
                run=s["run"];st=prof["steps"][run["step"]]
                if run["phase"]==0:ready=max(p["read"] if st["rf_read_bytes"] else 0,p["sh_read"] if st["sh_read_bytes"] else 0,p["interface"] if st["sh_tc_bytes"] else 0)
                elif run["phase"]==2:ready=max(p["write"] if st["rf_write_bytes"] else 0,p["sh_write"] if st["sh_write_bytes"] else 0)
                else:ready=t
                future(ready)
            elif False in busy[sm][prof["engine"]]:future(p["issue"])
        if next_ is None:raise ValueError("deadlock")
        t=next_
    add(t,1,h["sms"]*8);t+=1
    return dict(cycles=t,rf_bytes=rf_bytes,sh_bytes=sh_bytes,physical_fmas=fmas,
        scheduler_iterations=iterations,**energy_report(events,t,h))
