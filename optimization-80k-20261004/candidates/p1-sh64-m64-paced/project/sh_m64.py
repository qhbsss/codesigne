"""M64 with eight B banks, four in-place A banks and four accumulators."""
import json
from math import sqrt,pi
from .wide_qkv import dense as fallback
from .compiler import hbm,imm,add,mul,emit_barrier

def dense(lines,allworkers,tasks,rows,k,n,name,**kwargs):
    if kwargs.get('n_tile')!=64:return fallback(lines,allworkers,tasks,rows,k,n,name,**kwargs)
    for w,a,b,out,col in tasks:
        for mh in range(4):w.ld(hbm(a.offset+mh*16*(a.row_stride or k),1024,[16,64],[a.row_stride or k,1]),w._rf(8+mh,1024))
    emit_barrier(lines,allworkers)
    ident=name+'sh64';tile={'var':ident};row=mul(tile,64)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=rows//64,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:
        for mh in range(4):w.vec('add',[imm(0),imm(0)],w._rf(12+mh,1024))
    def load_part(w,ki,part):
        w.ld(w._sh((ki*8+part)*512,1024,[2,512],[2048,1]),w._rf(4*(ki%2)+part,1024))
    def load_b(w,ki):
        for part in range(4):load_part(w,ki,part)
    for w,a,b,out,col in tasks:load_b(w,0);load_b(w,1)
    for ki in range(4):
        for w,a,b,out,col in tasks:
            for mh in range(4):
                for part in range(4):
                    acc=w._rf(12+mh,256,part*16);acc.update(shape=[16,16],strides=[64,1])
                    w.emit('MMA.ACC',a=w._rf(8+mh,1024),b=w._rf(4*(ki%2)+part,1024),acc=acc,m=16,n=16,k=64,event=None)
                    if mh==3 and ki<2:load_part(w,ki+2,part)
                nextrow=row if ki<3 else mul({'min':[add(tile,1),rows//64-1]},64)
                nextk=(ki+1)*64 if ki<3 else 0
                w.ld(hbm(add(a.offset,mul(add(nextrow,mh*16),a.row_stride or k),nextk),1024,[16,64],[a.row_stride or k,1]),w._rf(8+mh,1024))
    for w,a,b,out,col in tasks:
        w.ld(hbm(w.w1_bias_hbm,64),w._rf(0,64));w.vec('add',[imm(1),imm(0)],w._rf(1,16))
        av=[];tv=[];hv=[]
        for mh in range(4):
            for nh in range(2):
                acc=w._rf(12+mh,512,nh*32);acc.update(shape=[16,32],strides=[64,1]);av.append(acc)
                tv.append(w._rf(mh,512,nh*512));hv.append(w._rf(4+mh,512,nh*512))
                w.emit('MMA.ACC',a=w._rf(1,16),b=w._rf(0,32,nh*32),acc=acc,m=16,n=32,k=1,event=None)
        for i in range(8):w.vec('mul',[av[i],av[i]],tv[i])
        for i in range(8):w.vec('fma',[tv[i],imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],tv[i])
        for i in range(8):w.vec('mul',[tv[i],av[i]],tv[i])
        for i in range(8):w.sfu('tanh',tv[i],tv[i])
        for i in range(8):w.vec('mul',[av[i],imm(.5)],hv[i])
        for i in range(8):w.vec('fma',[hv[i],tv[i],hv[i]],av[i])
        for mh in range(4):
            w.st(w._rf(12+mh,1024),hbm(add(out.offset,mul(add(row,mh*16),n),col),1024,[16,64],[n,1]))
            if mh<3:
                event=json.loads(w.lines[-1].split(' ',1)[1])['event']
                w.emit('WAIT',wg=w.wg,events=[event])
    emit_barrier(lines,allworkers)
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
