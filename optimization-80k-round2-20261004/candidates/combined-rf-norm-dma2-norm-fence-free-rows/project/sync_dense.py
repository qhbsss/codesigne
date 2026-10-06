"""Preload four A banks, then rotate six banks to overlap the next row tile."""
import json
from math import sqrt,pi
from .compiler import hbm,imm,add,mul,emit_barrier

def dense(lines,allworkers,tasks,rows,k,n,name,gelu=False,n_tile=32,k_tile=16,residual=None):
    for w,a,b,out,col in tasks:
        for ki in range(4):
            w.ld(hbm(a.offset+ki*64,1024,[16,64],[a.row_stride or k,1]),w._rf(8+ki,1024))
    emit_barrier(lines,allworkers)
    ident=name+'tile';tile={'var':ident};row=mul(tile,16)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=rows//16,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,16*n_tile*(2 if n_tile==16 else 1)))
    for ki in range(0,k,64):
        current=add(8,{'mod':[add(mul(tile,4),ki//64),6]})
        following=add(8,{'mod':[add(mul(add(tile,1),4),ki//64),6]})
        nextrow=mul({'min':[add(tile,1),rows//16-1]},16)
        for w,a,b,out,col in tasks:
            w.ld(hbm(add(a.offset,mul(nextrow,a.row_stride or k),ki),1024,[16,64],[a.row_stride or k,1]),w._rf(following,1024))
        for w,a,b,out,col in tasks:
            for part in range(max(1,n_tile//16)):
                if n_tile==32:_,lane,off,_,dk=next(c for c in b[1] if c[0]==part and c[3]==ki)
                else:lane,off,_,dk=next(c for c in b[1] if c[2]==ki)
                mn=min(16,n_tile);acc=w._rf(14,16*mn,part*16+(256*((ki//64)%2) if n_tile==16 else 0));acc.update(shape=[16,mn],strides=[n_tile,1])
                w.emit('MMA.ACC',a=w._rf(current,1024),b=w._rf(lane,64*mn,off),acc=acc,m=16,n=mn,k=64,event=None)
    for w,a,b,out,col in tasks:
        acc=w._rf(14,16*n_tile)
        if n_tile==16:w.vec('add',[acc,w._rf(14,16*n_tile,256)],acc)
        t=lambda count,off=0:w._rf(15,count,off)
        if gelu:
            w.vec('add',[imm(1),imm(0)],w._rf(15,16))
            w.emit('MMA.ACC',a=w._rf(15,16),b=w._rf(14,32,512),acc=acc,m=16,n=32,k=1,event=None)
            w.vec('mul',[acc,acc],t(512));w.vec('fma',[t(512),imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],t(512))
            w.vec('mul',[t(512),acc],t(512));w.sfu('tanh',t(512),t(512))
            w.vec('mul',[acc,imm(.5)],t(512,512));w.vec('fma',[t(512,512),t(512),t(512,512)],acc)
        if residual is not None:
            w.ld(hbm(add(residual.offset,mul(row,n),col),16*n_tile,[16,n_tile],[n,1]),t(16*n_tile))
            w.vec('add',[acc,t(16*n_tile)],acc)
        w.st(acc,hbm(add(out.offset,mul(row,n),col),16*n_tile,[16,n_tile],[n,1]))
    # Disjoint output cells and per-WG RF hazards make a row barrier unnecessary.
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
