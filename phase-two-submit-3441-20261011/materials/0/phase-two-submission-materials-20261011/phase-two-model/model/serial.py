"""Independent exact equations for one group, compute-only waves.

Excluded: DMA, multiple groups, runtime cache and network. Those are explicitly
delegated to the shared transition backend, never approximated here.
"""
from math import log2
from .hardware import base_power,rf_energy,rf_latency,sh_latency,validate
from .profiles import profile

def energy_report(events,cycles,h):
    """Independent continuous integral over integer-cycle constant services."""
    from bisect import bisect_right
    changes={}
    for start,duration,energy in events:
        changes[start]=changes.get(start,0)+energy/duration
        end=start+duration;changes[end]=changes.get(end,0)-energy/duration
    times=sorted(changes);rates=[];prefix=[];energy=0.;rate=0.;last=0
    for t in times:
        energy+=rate*(t-last);prefix.append(energy);rate+=changes[t];rates.append(rate);last=t
    def integral(t):
        j=bisect_right(times,t)-1
        return 0. if j<0 else prefix[j]+rates[j]*(t-times[j])
    def peak(width):
        return max([0.]+[integral(t)-integral(t-width) for t in set(times+[t+width for t in times])])
    base=base_power(h)
    return dict(short_power_w=base+peak(100)/(100*2000),
        long_power_w=base+peak(10000)/(10000*2000),
        energy_j=sum(e for _,_,e in events)*1e-12+base*cycles*2e-9)

def serial_wave(instructions,h):
    validate(h)
    events=[];t=0;rf_bytes=sh_bytes=fmas=0
    rb=[128,256,512][h["rf_tier"]-1];wb=rb//2
    def add(start,duration,energy):
        if duration:events.append((start,duration,energy))
    def charge(start,count,bw,energy):
        full,rem=divmod(count,bw)
        if full:add(start,full,full*bw*energy)
        if rem:add(start+full,1,rem*energy)
    sh_e=.8*(1+.05*log2(h["sh_kib"]/64)) if h["sh_kib"] else 0
    for i in instructions:
        p=profile(i,h);fmas+=p["physical_fmas"]
        add(t,1,8);t+=1
        for s in p["steps"]:
            delay=0
            if s["rf_read_bytes"]:
                count=s["rf_read_bytes"];service=(count+rb-1)//rb
                charge(t,count,rb,rf_energy(h));delay=max(delay,service+rf_latency(h));rf_bytes+=count
            if s["sh_read_bytes"]:
                service=s["sh_read_cycles"]
                for j,count in enumerate(s["sh_read_service"]):add(t+j,1,count*sh_e*(1.15 if h["shared_ports"]==2 else 1))
                delay=max(delay,service+sh_latency(h));sh_bytes+=s["sh_read_bytes"]
            if s["sh_tc_bytes"]:
                start=t+s["sh_read_cycles"]+sh_latency(h)
                service=(s["sh_tc_bytes"]+h["sh_tc_bw"]-1)//h["sh_tc_bw"]
                charge(start,s["sh_tc_bytes"],h["sh_tc_bw"],.2);delay=max(delay,start+service+2-t)
            t+=delay
            add(t,s["core_duration"],s["core_energy_pj"]);t+=s["core_duration"]
            delay=0
            if s["rf_write_bytes"]:
                count=s["rf_write_bytes"];service=(count+wb-1)//wb
                charge(t,count,wb,rf_energy(h));delay=max(delay,service+rf_latency(h));rf_bytes+=count
            if s["sh_write_bytes"]:
                for j,count in enumerate(s["sh_write_service"]):add(t+j,1,count*sh_e)
                delay=max(delay,s["sh_write_cycles"]+sh_latency(h));sh_bytes+=s["sh_write_bytes"]
            t+=delay
    add(t,1,8*h["sms"]);t+=1
    return {"cycles":t,"rf_bytes":rf_bytes,"sh_bytes":sh_bytes,"physical_fmas":fmas,
            **energy_report(events,t,h)}
