"""Thirty-two K-partitioned RF-local FFNs, packed one-load reduction."""
from .compiler import Tensor,hbm,imm,emit_barrier

def emit_ffn(lines,workers,shared,res,weights,y,d,f,name):
    partial=shared.alloc(32,d)
    for i,w in enumerate(workers):
        t=weights[i];ln=Tensor(128,(1,d),space='RF',lane=15)
        w.layernorm_persistent(res,t['ln2_g'],t['ln2_b'],ln,d)
        lo=i*(f//32);hi=lo+f//32
        act=Tensor(96,(1,f),space='RF',lane=15)
        w.gemm(ln,t['w1'],act,1,d,f,name+'f1',n_lo=lo,n_hi=hi,epilogue='bias_gelu',bias=t['b1'])
        for key in ('w2_lo','w2_hi'):
            w.lines.extend(w.pending_weights.pop(t[key].rf_chunks,[]))
        r=lambda count,off=0:w._rf(15,count,off)
        w.vec('add',[imm(0),imm(0)],r(d,768))
        for half,key in enumerate(('w2_lo','w2_hi')):
            chunk=t[key].rf_chunks[0]
            bank,off,_,_=chunk
            w.emit('MMA.ACC',a=r(f//32,96),b=w._rf(bank,(f//32)*(d//2),off),acc=r(d//2,768+half*d//2),m=1,n=d//2,k=f//32,event=None)
        w.st(r(d,768),hbm(partial.offset+i*d,d))
    emit_barrier(lines,workers)
    for i,w in enumerate(workers[:16]):
        col=i*(d//16);n=d//16;r=lambda count,off=0:w._rf(15,count,off)
        w.ld(hbm(partial.offset+col,32*n,[32,n],[d,1]),r(32*n,64))
        w.vec('add',[imm(1),imm(0)],r(32))
        w.vec('add',[imm(0),imm(0)],r(n,512))
        w.emit('MMA.ACC',a=r(32),b=r(32*n,64),acc=r(n,512),m=1,n=n,k=32,event=None)
        w.ld(hbm(res.offset+col,n),r(n,32));w.ld(hbm(weights[i]['b2'].offset+col,n),r(n,40))
        w.vec('add',[r(n,512),r(n,32)],r(n,48));w.vec('add',[r(n,48),r(n,40)],r(n,48))
        w.st(r(n,48),hbm(y.offset+col,n))
    emit_barrier(lines,workers)
