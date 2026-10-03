"""N64 with 3/4 of weights in RF and one packed tail in SH."""
import json
from math import sqrt,pi
from .wide_qkv import dense as fallback
from .compiler import hbm,imm,add,mul,emit_barrier

def dense(lines,allworkers,tasks,rows,k,n,name,**kwargs):
    if kwargs.get('n_tile')!=64:return fallback(lines,allworkers,tasks,rows,k,n,name,**kwargs)
    for w,a,b,out,col in tasks:
        for ki in range(3):w.ld(hbm(a.offset+ki*64,512,[8,64],[a.row_stride or k,1]),w._rf(12+ki//2,512,(ki%2)*512))
    emit_barrier(lines,allworkers)
    ident=name+'hybrid64tile';tile={'var':ident};row=mul(tile,8)
    nextrow=mul({'min':[add(tile,1),rows//8-1]},8)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=rows//8,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,512))
    for ki in range(3):
        for w,a,b,out,col in tasks:
            for part in range(4):
                acc=w._rf(14,128,part*16);acc.update(shape=[8,16],strides=[64,1])
                w.emit('MMA.ACC',a=w._rf(12+ki//2,512,(ki%2)*512),b=w._rf(part*3+ki,1024),acc=acc,m=8,n=16,k=64,event=None)
            if ki<2:w.ld(hbm(add(a.offset,mul(nextrow,a.row_stride or k),ki*64),512,[8,64],[a.row_stride or k,1]),w._rf(12,512,ki*512))
    for w,a,b,out,col in tasks:w.ld(hbm(add(a.offset,mul(row,a.row_stride or k),192),512,[8,64],[a.row_stride or k,1]),w._rf(15,512))
    for part in range(4):
        for w,a,b,out,col in tasks:
            w.ld(w._sh(part*1024,1024),w._rf(13,1024))
            acc=w._rf(14,128,part*16);acc.update(shape=[8,16],strides=[64,1])
            w.emit('MMA.ACC',a=w._rf(15,512),b=w._rf(13,1024),acc=acc,m=8,n=16,k=64,event=None)
    for w,a,b,out,col in tasks:
        w.ld(hbm(add(a.offset,mul(nextrow,a.row_stride or k),128),512,[8,64],[a.row_stride or k,1]),w._rf(13,512))
        acc=w._rf(14,512)
        if kwargs.get('gelu'):
            w.ld(w._sh(4096,64),w._rf(15,64));w.vec('add',[imm(1),imm(0)],w._rf(15,8,64))
            w.emit('MMA.ACC',a=w._rf(15,8,64),b=w._rf(15,64),acc=acc,m=8,n=64,k=1,event=None)
            temp=w._rf(15,512);half=w._rf(15,512,512)
            w.vec('mul',[acc,acc],temp);w.vec('fma',[temp,imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],temp)
            w.vec('mul',[temp,acc],temp);w.sfu('tanh',temp,temp);w.vec('mul',[acc,imm(.5)],half);w.vec('fma',[half,temp,half],acc)
        w.st(acc,hbm(add(out.offset,mul(row,n),col),512,[8,64],[n,1]))
    emit_barrier(lines,allworkers)
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
