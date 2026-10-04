"""Uniform 16-SM weight packets with explicit ISA pacing and private RF targets."""
import json
from .compiler import hbm,imm,add,mul,sub,emit_barrier

def floor(x,n):return sub({'ceildiv':[add(x,1),n]},1)

def background_cache(lines,sync,tasks,name,pause):
    result={i:(32,[(part,part*4+ki,0,ki*64,64) for part in range(2) for ki in range(4)]) for i,w,t,col in tasks}
    for cohort in range(0,len(tasks),16):
        active=tasks[cohort:cohort+16];ident=name+'packet'+str(cohort);pg={'var':ident}
        bank=floor(pg,8);part=floor(pg,32)
        kk=add(mul({'mod':[bank,4]},64),mul({'mod':[pg,8]},8))
        lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=64,step=1),separators=(',',':')))
        for i,w,t,col in active:w.loops.append(ident)
        for i,w,t,col in active:
            w.ld(hbm(add(t.offset,mul(kk,t.shape[1]),col,mul(part,16)),128,[8,16],[t.shape[1],1]),w._rf(bank,128,mul({'mod':[pg,8]},128)))
        emit_barrier(lines,sync)
        delay=name+'pause'+str(cohort)
        lines.append('FOR '+json.dumps(dict(var=delay,start=0,stop=pause,step=1),separators=(',',':')))
        for i,w,t,col in active:w.loops.append(delay)
        for i,w,t,col in active:w.vec('add',[imm(0),imm(0)],w._rf(14,1,1000))
        for i,w,t,col in active:w.loops.pop()
        lines.append('END.FOR {}')
        for i,w,t,col in active:w.loops.pop()
        lines.append('END.FOR {}')
        emit_barrier(lines,sync)
    return result
