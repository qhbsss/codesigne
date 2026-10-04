"""M16/N64, RF K192 + SH K64, rotating two tail-weight banks."""
import json
from math import sqrt,pi
from .wide_qkv import dense as fallback
from .compiler import hbm,imm,add,mul,emit_barrier

def dense(lines,allworkers,tasks,rows,k,n,name,**kwargs):
    if kwargs.get('n_tile')!=64:return fallback(lines,allworkers,tasks,rows,k,n,name,**kwargs)
    for w,a,b,out,col in tasks:
        for ki in range(2):w.ld(hbm(a.offset+ki*64,1024,[16,64],[a.row_stride or k,1]),w._rf(12+ki,1024))
    emit_barrier(lines,allworkers)
    ident=name+'m16hybrid';tile={'var':ident};row=mul(tile,16)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=rows//16,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,1024))
    def mac(w,ki,part,abank,bbank):
        acc=w._rf(14,256,part*16);acc.update(shape=[16,16],strides=[64,1])
        w.emit('MMA.ACC',a=w._rf(abank,1024),b=w._rf(bbank,1024),acc=acc,m=16,n=16,k=64,event=None)
    for ki in range(3):
        for w,a,b,out,col in tasks:
            for part in range(4):mac(w,ki,part,12+ki%2,part*3+ki)
            if ki==0:w.ld(hbm(add(a.offset,mul(row,a.row_stride or k),128),1024,[16,64],[a.row_stride or k,1]),w._rf(12,1024))
    for w,a,b,out,col in tasks:
        w.ld(hbm(add(a.offset,mul(row,a.row_stride or k),192),1024,[16,64],[a.row_stride or k,1]),w._rf(15,1024))
        w.ld(w._sh(0,1024),w._rf(12,1024))
        w.ld(w._sh(1024,1024),w._rf(13,1024))
    for part in range(4):
        for w,a,b,out,col in tasks:
            mac(w,3,part,15,12+part%2)
            if part<2:w.ld(w._sh((part+2)*1024,1024),w._rf(12+part%2,1024))
    for w,a,b,out,col in tasks:
        acc=w._rf(14,1024);temp=w._rf(12,1024);half=w._rf(13,1024)
        w.ld(w._sh(4096,64),w._rf(15,64));w.vec('add',[imm(1),imm(0)],w._rf(15,16,64))
        w.emit('MMA.ACC',a=w._rf(15,16,64),b=w._rf(15,64),acc=acc,m=16,n=64,k=1,event=None)
        w.vec('mul',[acc,acc],temp);w.vec('fma',[temp,imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],temp)
        w.vec('mul',[temp,acc],temp);w.sfu('tanh',temp,temp);w.vec('mul',[acc,imm(.5)],half);w.vec('fma',[half,temp,half],acc)
        w.st(acc,hbm(add(out.offset,mul(row,n),col),1024,[16,64],[n,1]))
        nextrow=mul({'min':[add(tile,1),rows//16-1]},16)
        for ki in range(2):w.ld(hbm(add(a.offset,mul(nextrow,a.row_stride or k),ki*64),1024,[16,64],[a.row_stride or k,1]),w._rf(12+ki,1024))
    emit_barrier(lines,allworkers)
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
