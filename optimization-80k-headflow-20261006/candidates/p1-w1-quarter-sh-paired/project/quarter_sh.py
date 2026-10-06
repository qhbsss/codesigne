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
    for w,a,out,col in tasks:w.ld(hbm(a.offset,512,[8,64],[256,1]),w._rf(12,512))
    emit_barrier(lines,allworkers)
    ident=name+'tile';tile={'var':ident}
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=rows//16,step=1),separators=(',',':')))
    for w,a,out,col in tasks:w.loops.append(ident)
    for slot in range(2):
        row=add(mul(tile,16),slot*8)
        for w,a,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,512,slot*512))
        for ki in range(4):
            nextrow=row if ki<3 else {'min':[add(row,8),rows-8]}
            for w,a,out,col in tasks:
                w.ld(hbm(add(a.offset,mul(nextrow,256),((ki+1)%4)*64),512,[8,64],[256,1]),w._rf(12,512,((ki+1)%2)*512))
                w.ld(w._sh(ki*1024,1024),w._rf(13,1024))
            for w,a,out,col in tasks:
                for part in range(4):
                    acc=w._rf(14,128,slot*512+part*16);acc.update(shape=[8,16],strides=[64,1])
                    w.emit('MMA.ACC',a=w._rf(12,512,(ki%2)*512),b=w._rf(part*4+ki if part<3 else 13,1024),acc=acc,m=8,n=16,k=64,event=None)
        if slot==1:
            for w,a,out,col in tasks:
                acc=w._rf(14,512);temp=w._rf(15,512)
                w.vec('mul',[acc,imm(.5)],acc);w.vec('fma',[acc,temp,acc],acc)
                w.st(acc,hbm(add(out.offset,mul(mul(tile,16),1024),col),512,[8,64],[1024,1]))
        for w,a,out,col in tasks:
            acc=w._rf(14,512,slot*512);temp=w._rf(15,512)
            w.vec('add',[imm(1),imm(0)],w._rf(15,8,512))
            w.emit('MMA.ACC',a=w._rf(15,8,512),b=w._rf(15,64,576),acc=acc,m=8,n=64,k=1,event=None)
            w.vec('mul',[acc,acc],temp);w.vec('fma',[temp,imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],temp)
            w.vec('mul',[temp,acc],temp);w.sfu('tanh',temp,temp)
    for w,a,out,col in tasks:
        acc=w._rf(14,512,512);temp=w._rf(15,512)
        w.vec('mul',[acc,imm(.5)],acc);w.vec('fma',[acc,temp,acc],acc)
        w.st(acc,hbm(add(out.offset,mul(add(mul(tile,16),8),1024),col),512,[8,64],[1024,1]))
    emit_barrier(lines,[w for w,a,out,col in tasks])
    for w,a,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')


def fill_w2(lines,owners,t,compute_lines):
    import re
    refs={}
    for line in compute_lines:
        if line.startswith('MMA.ACC '):
            d=json.loads(line[8:])
            if d['k']==64:refs[d['acc']['wg']]=d['event']
    assert len(refs)==16
    variables={tuple(re.findall(r'\{([^}]+)\}',e)) for e in refs.values()};assert len(variables)==1
    (ident,)=variables.pop();bank={'var':ident}
    part=sub({'ceildiv':[add(bank,1),4]},1);kk=mul({'mod':[bank,4]},64)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=8,step=1),separators=(',',':')))
    lines.append('BARRIER '+json.dumps(dict(wgs=[w.wg for w in owners],events=list(refs.values())),separators=(',',':')))
    for w in owners:w.loops.append(ident)
    for i,w in enumerate(owners):
        kg,ng=divmod(i,8)
        w.ld(hbm(add(t.offset,kg*256*256,mul(kk,256),ng*32,mul(part,16)),1024,[64,16],[256,1]),w._rf(bank,1024))
    for w in owners:w.loops.pop()
    lines.append('END.FOR {}')
    return {i:(32,[(p,p*4+k,0,k*64,64) for p in range(2) for k in range(4)]) for i in range(32)}
