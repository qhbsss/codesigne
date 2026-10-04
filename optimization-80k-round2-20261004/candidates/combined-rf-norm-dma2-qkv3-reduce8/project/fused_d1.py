"""Thirty-two K-partitioned RF-local FFNs, packed one-load reduction."""
from .compiler import Tensor,hbm,imm,emit_barrier

def emit_ffn(lines,workers,shared,res,weights,y,d,f,name):
    partial=shared.alloc(32,d)
    first_cohort_done=[]
    for i,w in enumerate(workers):
        t=weights[i];ln=Tensor(128,(1,d),space='RF',lane=15)
        lo=i*(f//32);hi=lo+f//32
        act=Tensor(96,(1,f),space='RF',lane=15)
        order=[(0,'w2_lo'),(1,'w2_hi')] if i%2==0 else [(1,'w2_hi'),(0,'w2_lo')]
        w.ffn_prefetch=[t[key] for half,key in order]
        w.ffn_wait_events=list(first_cohort_done) if i>=16 and any(t[key].rf_chunks in w.pending_weights for half,key in order) else []
        w.affine_gemm(res,t['ln2_g'],t['ln2_b'],t['w1'],act,1,d,f,name+'f1',n_lo=lo,n_hi=hi,epilogue='bias_gelu',bias=t['b1'])
        r=lambda count,off=0:w._rf(15,count,off)
        w.vec('add',[imm(0),imm(0)],r(d,768))
        for half,key in order:
            w.lines.extend(w.pending_weights.pop(t[key].rf_chunks,[]))
            chunk=t[key].rf_chunks[0]
            bank,off,_,_=chunk
            w.emit('MMA.ACC',a=r(f//32,96),b=w._rf(bank,(f//32)*(d//2),off),acc=r(d//2,768+half*d//2),m=1,n=d//2,k=f//32,event=None)
        w.st(r(d,768),hbm(partial.offset+i*d,d))
        if i<16:
            import json
            first_cohort_done.append(json.loads(lines[-1][3:])['event'])
    emit_barrier(lines,workers)
    for i,w in enumerate(workers[:8]):
        col=i*(d//8);n=d//8;r=lambda count,off=0:w._rf(15,count,off)
        w.ld(hbm(partial.offset+col,32*n,[32,n],[d,1]),r(32*n,64))
        w.vec('add',[imm(1),imm(0)],r(32))
        w.vec('add',[imm(0),imm(0)],r(n,600))
        w.emit('MMA.ACC',a=r(32),b=r(32*n,64),acc=r(n,600),m=1,n=n,k=32,event=None)
        base=weights[i]['ln2_b'].offset
        w.vec('add',[r(n,600),w._rf(14,n,base+64)],r(n,48))
        w.vec('add',[r(n,48),w._rf(14,n,base+48)],r(n,48))
        w.st(r(n,48),hbm(y.offset+col,n))
    emit_barrier(lines,workers)
