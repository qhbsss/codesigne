"""Four independent M8 accumulators; 32-row GELU and two W2 banks per block."""
import json,re
from math import sqrt,pi
from .compiler import hbm,imm,add,mul,sub,emit_barrier
from .wide_qkv import dense as original

def dense(lines,allworkers,tasks,rows,k,n,name,**kwargs):
    if not kwargs.get('gelu',False):return original(lines,allworkers,tasks,rows,k,n,name,**kwargs)
    assert rows==128 and k==256 and n==1024 and len(tasks)==32
    def load(w,a,tile,ki,bank):
        w.ld(hbm(add(a.offset,mul(tile,32*(a.row_stride or k)),ki),1024,[32,32],[a.row_stride or k,1]),w._rf(bank,1024))
    for w,a,b,out,col in tasks:load(w,a,0,0,8)
    emit_barrier(lines,allworkers)
    ident=name+'tile';tile={'var':ident}
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=4,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,1024))
    for ki in range(8):
        nexttile={'min':[add(tile,1),3]} if ki==7 else tile
        for w,a,b,out,col in tasks:load(w,a,nexttile,((ki+1)%8)*32,8+(ki+1)%2)
        for w,a,b,out,col in tasks:
            for panel in range(2):
                _,bank,off,_,depth=next(c for c in b[1] if c[0]==panel and c[3]==(ki//2)*64)
                acc=w._rf(14,512,panel*16);acc.update(shape=[32,16],strides=[32,1])
                w.emit('MMA.ACC',a=w._rf(8+ki%2,1024),b=w._rf(bank,512,off+(ki%2)*512),acc=acc,m=32,n=16,k=32,event=None)
    for w,a,b,out,col in tasks:
        acc=w._rf(14,1024);temp=w._rf(15,1024)
        w.vec('add',[imm(1),imm(0)],w._rf(15,32))
        w.emit('MMA.ACC',a=w._rf(15,32),b=w._rf(13,32),acc=acc,m=32,n=32,k=1,event=None)
        w.vec('mul',[acc,acc],temp);w.vec('fma',[temp,imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],temp)
        w.vec('mul',[temp,acc],temp);w.sfu('tanh',temp,temp)
        w.vec('mul',[acc,imm(.5)],acc);w.vec('fma',[acc,temp,acc],acc)
        w.st(acc,hbm(add(out.offset,mul(tile,32*n),col),1024,[32,32],[n,1]))
    emit_barrier(lines,[w for w,a,b,out,col in tasks])
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')

def fill_w2(lines,owners,t,compute_lines):
    refs={}
    for line in compute_lines:
        if line.startswith('MMA.ACC '):
            d=json.loads(line[8:])
            if d['k']==32:refs.setdefault(d['acc']['wg'],[]).append(d['event'])
    assert len(refs)==32 and all(len(items)==16 for items in refs.values())
    variables={tuple(re.findall(r'\{([^}]+)\}',e)) for items in refs.values() for e in items};assert len(variables)==1
    (ident,)=variables.pop();tile={'var':ident}
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=4,step=1),separators=(',',':')))
    for w in owners:w.loops.append(ident)
    for extra in range(2):
        lines.append('BARRIER '+json.dumps(dict(wgs=[w.wg for w in owners],events=[items[7 if extra==0 else 15] for items in refs.values()]),separators=(',',':')))
        bank=add(mul(tile,2),extra);panel=sub({'ceildiv':[add(bank,1),4]},1);kk=mul({'mod':[bank,4]},64)
        for i,w in enumerate(owners):
            kg,ng=divmod(i,8)
            w.ld(hbm(add(t.offset,kg*256*256,mul(kk,256),ng*32,mul(panel,16)),1024,[64,16],[256,1]),w._rf(bank,1024))
    for w in owners:w.loops.pop()
    lines.append('END.FOR {}')
    return {i:(32,[(p,p*4+kk,0,kk*64,64) for p in range(2) for kk in range(4)]) for i in range(32)}
