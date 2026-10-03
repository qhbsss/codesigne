"""N64 FFN: full weights in SH, ping-pong RF panels and six-bank A ring."""
import json
from math import sqrt,pi
from .wide_qkv import dense as fallback
from .compiler import hbm,imm,add,mul,emit_barrier

def dense(lines,allworkers,tasks,rows,k,n,name,**kwargs):
    if kwargs.get('n_tile')!=64:
        return fallback(lines,allworkers,tasks,rows,k,n,name,**kwargs)
    for w,a,b,out,col in tasks:
        for ki in range(4):w.ld(hbm(a.offset+ki*64,1024,[16,64],[a.row_stride or k,1]),w._rf(8+ki,1024))
        for part in range(4):w.ld(w._sh(b[1]+part*16,1024,[64,16],[64,1]),w._rf(part,1024))
    emit_barrier(lines,allworkers)
    ident=name+'wide64tile';tile={'var':ident};row=mul(tile,16)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=rows//16,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,1024))
    for ki in range(4):
        current=add(8,{'mod':[add(mul(tile,4),ki),6]});following=add(8,{'mod':[add(mul(add(tile,1),4),ki),6]})
        nextrow=mul({'min':[add(tile,1),rows//16-1]},16)
        for w,a,b,out,col in tasks:
            w.ld(hbm(add(a.offset,mul(nextrow,a.row_stride or k),ki*64),1024,[16,64],[a.row_stride or k,1]),w._rf(following,1024))
            nk=(ki+1)%4
            for part in range(4):w.ld(w._sh(b[1]+nk*64*64+part*16,1024,[64,16],[64,1]),w._rf((1-ki%2)*4+part,1024))
        for w,a,b,out,col in tasks:
            for part in range(4):
                acc=w._rf(14,256,part*16);acc.update(shape=[16,16],strides=[64,1])
                w.emit('MMA.ACC',a=w._rf(current,1024),b=w._rf((ki%2)*4+part,1024),acc=acc,m=16,n=16,k=64,event=None)
    for w,a,b,out,col in tasks:
        if kwargs.get('gelu'):
            w.ld(w._sh(0,64),w._rf(15,64));w.vec('add',[imm(1),imm(0)],w._rf(15,16,64))
            w.emit('MMA.ACC',a=w._rf(15,16,64),b=w._rf(15,64),acc=w._rf(14,1024),m=16,n=64,k=1,event=None)
            for part in range(2):
                acc=w._rf(14,512,part*512);temp=w._rf(15,512);half=w._rf(15,512,512)
                w.vec('mul',[acc,acc],temp);w.vec('fma',[temp,imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],temp)
                w.vec('mul',[temp,acc],temp);w.sfu('tanh',temp,temp);w.vec('mul',[acc,imm(.5)],half);w.vec('fma',[half,temp,half],acc)
        w.st(w._rf(14,1024),hbm(add(out.offset,mul(row,n),col),1024,[16,64],[n,1]))
    emit_barrier(lines,allworkers)
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
