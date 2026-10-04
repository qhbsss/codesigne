"""Two-row FFN: keep activation in RF, then reuse expired W1 banks for W2."""
from .compiler import Tensor,hbm,imm,emit_barrier

def emit_ffn(lines,workers,shared,ln,res,weights,y,name):
    from .step_cached import matrix
    d=256;f=1024;partial=shared.alloc(32,2,d)
    order=[off+base for off in range(2) for base in range(0,32,2)]
    for j,i in enumerate(order):
        w=workers[i];lo=i*32
        wb=w.cache(weights['w1'],lo,lo+32,64)
        w.cache_vector(weights['b1'],lo,32,1536)
        matrix(w,ln,wb,Tensor(512,(2,32),space='RF',lane=15),d,f,lo,gelu=True)
        for part in range(8):
            w.ld(hbm(weights['w2'].offset+lo*d+part*32,1024,[32,32],[d,1]),w._rf(part,1024))
        w.vec('add',[imm(0),imm(0)],w._rf(14,512))
        for part in range(8):
            acc=w._rf(14,64,part*32);acc.update(shape=[2,32],strides=[d,1])
            w.emit('MMA.ACC',a=w._rf(15,64,512),b=w._rf(part,1024),acc=acc,m=2,n=32,k=32,event=None)
        w.st(w._rf(14,512),hbm(partial.offset+i*2*d,512))
        if j%8==7:emit_barrier(lines,workers)
    emit_barrier(lines,workers)
    for i,w in enumerate(workers[:16]):
        col=i*16
        w.ld(hbm(partial.offset+col,1024,[64,16],[d,1]),w._rf(15,1024))
        w.vec('add',[imm(1),imm(0)],w._rf(11,32))
        w.vec('add',[imm(0),imm(0)],w._rf(14,32))
        w.emit('MMA.ACC',a=w._rf(11,32),b=w._rf(15,1024),acc=w._rf(14,32),m=1,n=32,k=32,event=None)
        w.ld(hbm(res.offset+col,32,[2,16],[d,1]),w._rf(15,32))
        w.vec('add',[w._rf(14,32),w._rf(15,32)],w._rf(14,32))
        w.ld(hbm(weights['b2'].offset+col,16),w._rf(15,16,64))
        w.vec('add',[imm(1),imm(0)],w._rf(15,2,96))
        w.emit('MMA.ACC',a=w._rf(15,2,96),b=w._rf(15,16,64),acc=w._rf(14,32),m=2,n=16,k=1,event=None)
        w.st(w._rf(14,32),hbm(y.offset+col,32,[2,16],[d,1]))
    emit_barrier(lines,workers)
