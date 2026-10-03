"""Trade K partitions for N width: half A replication, twice partial outputs."""
import json
from .sync_dense import dense as fallback
from .compiler import hbm,imm,add,mul,emit_barrier

def dense(lines,allworkers,tasks,rows,k,n,name,**kwargs):
    if k!=128:return fallback(lines,allworkers,tasks,rows,k,n,name,**kwargs)
    assert kwargs['n_tile']==64
    for w,a,b,out,col in tasks:
        for ki in range(2):w.ld(hbm(a.offset+ki*64,1024,[16,64],[a.row_stride or k,1]),w._rf(8+ki,1024))
    emit_barrier(lines,allworkers)
    ident=name+'k128tile';tile={'var':ident};row=mul(tile,16)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=rows//16,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,1024))
    for ki in range(2):
        current=add(8,{'mod':[add(mul(tile,2),ki),4]});following=add(8,{'mod':[add(mul(add(tile,1),2),ki),4]})
        nextrow=mul({'min':[add(tile,1),rows//16-1]},16)
        for w,a,b,out,col in tasks:w.ld(hbm(add(a.offset,mul(nextrow,a.row_stride or k),ki*64),1024,[16,64],[a.row_stride or k,1]),w._rf(following,1024))
        for w,a,b,out,col in tasks:
            for part in range(4):
                acc=w._rf(14,256,part*16);acc.update(shape=[16,16],strides=[64,1])
                w.emit('MMA.ACC',a=w._rf(current,1024),b=w._rf(part*2+ki,1024),acc=acc,m=16,n=16,k=64,event=None)
    for w,a,b,out,col in tasks:w.st(w._rf(14,1024),hbm(add(out.offset,mul(row,n),col),1024,[16,64],[n,1]))
    emit_barrier(lines,allworkers)
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
