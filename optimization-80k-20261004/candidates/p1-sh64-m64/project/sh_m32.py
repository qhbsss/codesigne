"""N64/M32 full SH weights, compact N16 tiles and double A/B prefetch."""
import json
from math import sqrt,pi
from .wide_qkv import dense as fallback
from .compiler import hbm,imm,add,mul,emit_barrier

def dense(lines,allworkers,tasks,rows,k,n,name,**kwargs):
    if kwargs.get('n_tile')!=64:return fallback(lines,allworkers,tasks,rows,k,n,name,**kwargs)
    for w,a,b,out,col in tasks:w.ld(hbm(a.offset,1024,[32,32],[a.row_stride or k,1]),w._rf(12,1024))
    emit_barrier(lines,allworkers)
    ident=name+'sh32';tile={'var':ident};row=mul(tile,32)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=rows//32,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:
        w.vec('add',[imm(0),imm(0)],w._rf(10,1024));w.vec('add',[imm(0),imm(0)],w._rf(11,1024))
    def preload(w,a,ki):
        for part in range(4):w.ld(w._sh((ki*4+part)*512,512),w._rf(2*(ki%2)+part//2,512,(part%2)*512))
        w.ld(hbm(add(a.offset,mul(row,a.row_stride or k),ki*32),1024,[32,32],[a.row_stride or k,1]),w._rf(12+ki%2,1024))
    for w,a,b,out,col in tasks:preload(w,a,0);preload(w,a,1)
    for ki in range(8):
        for w,a,b,out,col in tasks:
            for mh in range(2):
                for part in range(4):
                    av=w._rf(12+ki%2,512,mh*512)
                    bv=w._rf(2*(ki%2)+part//2,512,(part%2)*512)
                    acc=w._rf(10+mh,256,part*16);acc.update(shape=[16,16],strides=[64,1])
                    w.emit('MMA.ACC',a=av,b=bv,acc=acc,m=16,n=16,k=32,event=None)
            if ki<6:preload(w,a,ki+2)
    for w,a,b,out,col in tasks:
        w.vec('add',[imm(1),imm(0)],w._rf(4,16))
        av=[];tv=[];hv=[]
        for mh in range(2):
            for nh in range(2):
                acc=w._rf(10+mh,512,nh*32);acc.update(shape=[16,32],strides=[64,1]);av.append(acc)
                tv.append(w._rf(6+mh,512,nh*512));hv.append(w._rf(8+mh,512,nh*512))
                w.emit('MMA.ACC',a=w._rf(4,16),b=w._rf(14,32,512+nh*32),acc=acc,m=16,n=32,k=1,event=None)
        for i in range(4):w.vec('mul',[av[i],av[i]],tv[i])
        for i in range(4):w.vec('fma',[tv[i],imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],tv[i])
        for i in range(4):w.vec('mul',[tv[i],av[i]],tv[i])
        for i in range(4):w.sfu('tanh',tv[i],tv[i])
        for i in range(4):w.vec('mul',[av[i],imm(.5)],hv[i])
        for i in range(4):w.vec('fma',[hv[i],tv[i],hv[i]],av[i])
        for mh in range(2):w.st(w._rf(10+mh,1024),hbm(add(out.offset,mul(add(row,mh*16),n),col),1024,[16,64],[n,1]))
    emit_barrier(lines,allworkers)
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
