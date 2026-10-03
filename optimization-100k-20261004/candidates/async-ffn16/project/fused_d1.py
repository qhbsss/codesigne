"""Sixteen FFNs per layer on disjoint workers, packed one-load reduction."""
from .compiler import Tensor,hbm,imm,emit_barrier

def emit_ffn(lines,workers,shared,res,weights,y,d,f,name):
    partial=shared.alloc(16,d)
    for i,w in enumerate(workers):
        t=weights[i]
        if 'ffn_slot' not in t:continue
        slot=t['ffn_slot'];ln=Tensor(128,(1,d),space='RF',lane=15)
        w.layernorm_persistent(res,t['ln2_g'],t['ln2_b'],ln,d)
        lo=slot*(f//16);hi=lo+f//16
        act=Tensor(96,(1,f),space='RF',lane=15)
        w.gemm(ln,t['w1'],act,1,d,f,name+'f1',n_lo=lo,n_hi=hi,epilogue='bias_gelu',bias=t['b1'])
        w.gemm(Tensor(96,(1,f//16),space='RF',lane=15),t['w2'],Tensor(partial.offset+slot*d,(1,d)),1,f//16,d,name+'f2')
    emit_barrier(lines,workers)
    for i,w in enumerate(workers[:16]):
        col=i*(d//16);n=d//16;r=lambda count,off=0:w._rf(15,count,off)
        w.ld(hbm(partial.offset+col,16*n,[16,n],[d,1]),r(16*n,64))
        w.vec('add',[imm(1),imm(0)],r(16))
        w.vec('add',[imm(0),imm(0)],r(n,512))
        w.emit('MMA.ACC',a=r(16),b=r(16*n,64),acc=r(n,512),m=1,n=n,k=16,event=None)
        w.ld(hbm(res.offset+col,n),r(n,32));w.ld(hbm(weights[i]['b2'].offset+col,n),r(n,40))
        w.vec('add',[r(n,512),r(n,32)],r(n,48));w.vec('add',[r(n,48),r(n,40)],r(n,48))
        w.st(r(n,48),hbm(y.offset+col,n))
    emit_barrier(lines,workers)
