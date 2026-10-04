"""Balance QKV across 16 SMs, reusing each input for three N16 panels."""
import json
from .sync_dense import dense as original_dense
from .compiler import hbm,imm,add,mul,emit_barrier

def dense(lines,allworkers,tasks,rows,k,n,name,**kwargs):
    if kwargs.get('n_tile')!=48:
        return original_dense(lines,allworkers,tasks,rows,k,n,name,**kwargs)
    assert k==256 and rows%8==0
    for w,a,b,out,col in tasks:
        w.lines.extend(getattr(w, 'pending_prompt_weights', {}).pop(id(b), []))
    for w,a,b,out,col in tasks:
        for ki in range(4):
            w.ld(hbm(a.offset+ki*64,512,[8,64],[a.row_stride or k,1]),w._rf(12+ki//2,512,(ki%2)*512))
    ident=name+'wide48tile';tile={'var':ident};row=mul(tile,8)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=rows//8,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,384))
    for ki in range(4):
        for w,a,b,out,col in tasks:
            for part in range(3):
                _,lane,off,_,dk=next(c for c in b[1] if c[0]==part and c[3]==ki*64)
                acc=w._rf(14,128,part*16);acc.update(shape=[8,16],strides=[48,1])
                w.emit('MMA.ACC',a=w._rf(12+ki//2,512,(ki%2)*512),b=w._rf(lane,1024,off),acc=acc,m=8,n=16,k=64,event=None)
            # Read-after-write hazards protect each slice; prefetch replaces only consumed A.
            nextrow=mul({'min':[add(tile,1),rows//8-1]},8)
            w.ld(hbm(add(a.offset,mul(nextrow,a.row_stride or k),ki*64),512,[8,64],[a.row_stride or k,1]),w._rf(12+ki//2,512,(ki%2)*512))
    for w,a,b,out,col in tasks:
        w.st(w._rf(14,384),hbm(add(out.offset,mul(row,n),col),384,[8,48],[n,1]))
    # RF hazards protect reuse; the caller fences the complete dense stage.
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
