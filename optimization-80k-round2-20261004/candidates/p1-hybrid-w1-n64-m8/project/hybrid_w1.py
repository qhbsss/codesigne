"""N64 W1: thirteen RF weight banks, three SH spill banks, M8 rows."""
import json
from math import sqrt,pi
from .compiler import hbm,imm,add,mul,emit_barrier

def cache(w,t,col):
    for part in range(4):
        for ki in range(4):
            bank=part*4+ki;src=hbm(t.offset+ki*64*1024+col+part*16,1024,[64,16],[1024,1])
            w.ld(src,w._rf(bank,1024) if bank<13 else w._sh((bank-13)*1024,1024))

def dense(lines,allworkers,tasks,rows,name):
    for w,a,out,col in tasks:w.ld(hbm(a.offset,512,[8,64],[256,1]),w._rf(14,512))
    emit_barrier(lines,allworkers)
    ident=name+'tile';tile={'var':ident};row=mul(tile,8)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=rows//8,step=1),separators=(',',':')))
    for w,a,out,col in tasks:w.loops.append(ident)
    for w,a,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(13,512))
    for ki in range(4):
        nextrow=row if ki<3 else mul({'min':[add(tile,1),rows//8-1]},8)
        for w,a,out,col in tasks:
            w.ld(hbm(add(a.offset,mul(nextrow,256),((ki+1)%4)*64),512,[8,64],[256,1]),w._rf(14,512,((ki+1)%2)*512))
            if ki>=1:w.ld(w._sh((ki-1)*1024,1024),w._rf(15,1024))
        for w,a,out,col in tasks:
            for part in range(4):
                bank=part*4+ki;acc=w._rf(13,128,part*16);acc.update(shape=[8,16],strides=[64,1])
                w.emit('MMA.ACC',a=w._rf(14,512,(ki%2)*512),b=w._rf(bank if bank<13 else 15,1024),acc=acc,m=8,n=16,k=64,event=None)
    for w,a,out,col in tasks:
        acc=w._rf(13,512);temp=w._rf(15,512);half=w._rf(15,512,512)
        w.vec('add',[imm(1),imm(0)],w._rf(15,8))
        w.emit('MMA.ACC',a=w._rf(15,8),b=w._rf(13,64,768),acc=acc,m=8,n=64,k=1,event=None)
        w.vec('mul',[acc,acc],temp);w.vec('fma',[temp,imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],temp)
        w.vec('mul',[temp,acc],temp);w.sfu('tanh',temp,temp);w.vec('mul',[acc,imm(.5)],half);w.vec('fma',[half,temp,half],acc)
        w.st(acc,hbm(add(out.offset,mul(row,1024),col),512,[8,64],[1024,1]))
    emit_barrier(lines,[w for w,a,out,col in tasks])
    for w,a,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
