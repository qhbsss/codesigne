"""M32 K32 FFN, four chunks ahead in a six-bank ring."""
import json
from math import sqrt,pi
from .wide_qkv import dense as fallback
from .compiler import hbm,imm,add,mul,emit_barrier

def dense(lines,allworkers,tasks,rows,k,n,name,**kwargs):
    if kwargs.get('n_tile',32)!=32:return fallback(lines,allworkers,tasks,rows,k,n,name,**kwargs)
    assert rows==128 and k==256
    for w,a,b,out,col in tasks:
        for ki in range(4):w.ld(hbm(a.offset+ki*32,1024,[32,32],[a.row_stride or k,1]),w._rf(8+ki,1024))
    emit_barrier(lines,allworkers)
    ident=name+'stream32tile';tile={'var':ident};row=mul(tile,32)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=4,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,1024))
    for ki in range(8):
        idx=add(mul(tile,8),ki);target={'min':[add(idx,4),31]}
        targetrow=mul(add({'ceildiv':[add(target,1),8]},-1),32);targetk=mul({'mod':[target,8]},32)
        current=add(8,{'mod':[idx,6]});following=add(8,{'mod':[add(idx,4),6]})
        for w,a,b,out,col in tasks:w.ld(hbm(add(a.offset,mul(targetrow,a.row_stride or k),targetk),1024,[32,32],[a.row_stride or k,1]),w._rf(following,1024))
        for w,a,b,out,col in tasks:
            for part in range(2):
                acc=w._rf(14,512,part*16);acc.update(shape=[32,16],strides=[32,1])
                w.emit('MMA.ACC',a=w._rf(current,1024),b=w._rf(part*4+ki//2,512,(ki%2)*512),acc=acc,m=32,n=16,k=32,event=None)
    for w,a,b,out,col in tasks:
        if kwargs.get('gelu'):
            w.ld(w._sh(0,32),w._rf(15,32));w.vec('add',[imm(1),imm(0)],w._rf(15,32,32))
            w.emit('MMA.ACC',a=w._rf(15,32,32),b=w._rf(15,32),acc=w._rf(14,1024),m=32,n=32,k=1,event=None)
            for part in range(2):
                acc=w._rf(14,512,part*512);temp=w._rf(15,512);half=w._rf(15,512,512)
                w.vec('mul',[acc,acc],temp);w.vec('fma',[temp,imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],temp)
                w.vec('mul',[temp,acc],temp);w.sfu('tanh',temp,temp);w.vec('mul',[acc,imm(.5)],half);w.vec('fma',[half,temp,half],acc)
        w.st(w._rf(14,1024),hbm(add(out.offset,mul(row,n),col),1024,[32,32],[n,1]))
    emit_barrier(lines,allworkers)
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
