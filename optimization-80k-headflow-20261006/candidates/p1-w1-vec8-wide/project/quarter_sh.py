"""N64 W1: twelve RF weight banks, four SH spill banks, M8 rows."""
import json
from math import sqrt,pi
from .compiler import hbm,imm,add,mul,sub,emit_barrier

def cache(w,t,col):
    for part in range(4):
        for ki in range(4):
            bank=part*4+ki;src=hbm(t.offset+ki*64*1024+col+part*16,1024,[64,16],[1024,1])
            w.ld(src,w._rf(bank,1024) if part<3 else w._sh(ki*1024,1024))

def dense(lines,allworkers,tasks,rows,name):
    ident=name+'tile';tile={'var':ident};row=mul(tile,32)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=4,step=1),separators=(',',':')))
    for w,a,out,col in tasks:w.loops.append(ident)
    for w,a,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,2048))
    for ki in range(8):
        for w,a,out,col in tasks:
            w.ld(hbm(add(a.offset,mul(row,256),ki*32),1024,[32,32],[256,1]),w._rf(12,1024))
            w.ld(w._sh(ki*512,512),w._rf(13,512))
        for w,a,out,col in tasks:
            for part in range(4):
                acc=w._rf(14,512,part*16);acc.update(shape=[32,16],strides=[64,1])
                w.emit('MMA.ACC',a=w._rf(12,1024),b=w._rf(part*4+ki//2,512,(ki%2)*512) if part<3 else w._rf(13,512),acc=acc,m=32,n=16,k=32,event=None)
    for w,a,out,col in tasks:
        acc=w._rf(14,2048);temp=w._rf(12,2048)
        w.ld(hbm(w.w1_bias_source,64),w._rf(13,64,576))
        w.vec('add',[imm(1),imm(0)],w._rf(13,32,512))
        w.emit('MMA.ACC',a=w._rf(13,32,512),b=w._rf(13,64,576),acc=acc,m=32,n=64,k=1,event=None)
        w.vec('mul',[acc,acc],temp);w.vec('fma',[temp,imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],temp)
        w.vec('mul',[temp,acc],temp);w.sfu('tanh',temp,temp)
        w.vec('mul',[acc,imm(.5)],acc);w.vec('fma',[acc,temp,acc],acc)
        w.st(acc,hbm(add(out.offset,mul(row,1024),col),2048,[32,64],[1024,1]))
    emit_barrier(lines,[w for w,a,out,col in tasks])
    for w,a,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')


def fill_w2(lines,owners,t,compute_lines):
    import re
    refs={}
    for line in compute_lines:
        if line.startswith('MMA.ACC '):
            d=json.loads(line[8:])
            if d['k']==32:refs.setdefault(d['acc']['wg'],[]).append(d['event'])
    assert len(refs)==16 and all(len(es)==32 for es in refs.values())
    variables={tuple(re.findall(r'\{([^}]+)\}',e)) for es in refs.values() for e in es};assert len(variables)==1
    (ident,)=variables.pop();tile={'var':ident}
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=4,step=1),separators=(',',':')))
    for w in owners:w.loops.append(ident)
    for extra in range(2):
        lines.append('BARRIER '+json.dumps(dict(wgs=[w.wg for w in owners],events=[es[15 if extra==0 else 31] for es in refs.values()]),separators=(',',':')))
        bank=add(mul(tile,2),extra);part=sub({'ceildiv':[add(bank,1),4]},1);kk=mul({'mod':[bank,4]},64)
        for i,w in enumerate(owners):
            kg,ng=divmod(i,8)
            w.ld(hbm(add(t.offset,kg*256*256,mul(kk,256),ng*32,mul(part,16)),1024,[64,16],[256,1]),w._rf(bank,1024))
    for w in owners:w.loops.pop()
    lines.append('END.FOR {}')
    return {i:(32,[(p,p*4+k,0,k*64,64) for p in range(2) for k in range(4)]) for i in range(32)}
