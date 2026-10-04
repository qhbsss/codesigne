"""M32 RF-resident W1, six-bank streaming A and a dead A bank for GELU."""
import json
from math import sqrt,pi
from .compiler import hbm,imm,add,mul,sub,emit_barrier
from .wide_qkv import dense as original

def dense(lines,allworkers,tasks,rows,k,n,name,**kwargs):
    if not kwargs.get('gelu'):return original(lines,allworkers,tasks,rows,k,n,name,**kwargs)
    assert k==256 and rows%32==0
    for w,a,b,out,col in tasks:
        for ki in range(4):
            w.ld(hbm(a.offset+ki*32,1024,[32,32],[a.row_stride or k,1]),w._rf(8+ki,1024))
    emit_barrier(lines,allworkers)
    ident=name+'rf32tile';tile={'var':ident};row=mul(tile,32);base=mul(tile,8)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=rows//32,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,1024))
    for ki in range(8):
        current=add(8,{'mod':[add(base,ki),6]})
        for w,a,b,out,col in tasks:
            for part in range(2):
                _,lane,off,_,_=next(c for c in b[1] if c[0]==part and c[3]==(ki//2)*64)
                acc=w._rf(14,512,part*16);acc.update(shape=[32,16],strides=[32,1])
                w.emit('MMA.ACC',a=w._rf(current,1024),b=w._rf(lane,512,off+(ki%2)*512),acc=acc,m=32,n=16,k=32,event=None)
            future=add(base,ki+4)
            # No speculative overwrite beyond the final row.
            with w.loop(name+'pf'+str(ki),0,{'min':[1,{'max':[0,sub(rows//32*8,future)]}]}) as unused:
                rr=mul(sub({'ceildiv':[add(future,1),8]},1),32);kk=mul({'mod':[future,8]},32)
                w.ld(hbm(add(a.offset,mul(rr,a.row_stride or k),kk),1024,[32,32],[a.row_stride or k,1]),w._rf(add(8,{'mod':[future,6]}),1024))
    for w,a,b,out,col in tasks:
        acc=w._rf(14,1024);temp=w._rf(15,1024)
        half=w._rf(add(8,{'mod':[add(base,12),6]}),1024)
        w.ld(w._sh(0,32),w._rf(15,32))
        w.vec('add',[imm(1),imm(0)],w._rf(15,32,64))
        w.emit('MMA.ACC',a=w._rf(15,32,64),b=w._rf(15,32),acc=acc,m=32,n=32,k=1,event=None)
        w.vec('mul',[acc,acc],temp)
        w.vec('fma',[temp,imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],temp)
        w.vec('mul',[temp,acc],temp);w.sfu('tanh',temp,temp)
        w.vec('mul',[acc,imm(.5)],half);w.vec('fma',[half,temp,half],acc)
        w.st(acc,hbm(add(out.offset,mul(row,n),col),1024,[32,32],[n,1]))
    emit_barrier(lines,allworkers)
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
