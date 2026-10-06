"""Four independent MACs share one LayerNorm and one 32-element GELU."""
from .compiler import hbm,imm

def execute(w,x,t):
    k=128;r=lambda count,off=0:w._rf(15,count,off)
    w.ld(hbm(x.offset,k),r(k))
    w.vec('mul',[r(k),w._rf(14,k,0)],r(k,128))
    w.reduce('sum',r(k),r(1,700));w.vec('mul',[r(k),r(k)],r(k,256))
    for off in [768,800,832,864]:w.vec('add',[imm(0),imm(0)],r(16,off))
    for ci in range(2):
        for part in range(2):
            bank,off,start,depth=t['w1parts'][part].rf_chunks[ci]
            w.emit('MMA.ACC',a=r(depth,128+start),b=w._rf(bank,depth*16,off),acc=r(16,768+ci*64+part*32),m=1,n=16,k=depth,event=None)
    w.reduce('sum',r(k,256),r(1,701))
    w.vec('mul',[r(1,700),imm(1/k)],r(1,700));w.vec('mul',[r(1,700),r(1,700)],r(1,702))
    w.vec('sub',[imm(1e-5),r(1,702)],r(1,702));w.vec('fma',[r(1,701),imm(1/k),r(1,702)],r(1,701))
    w.vec('max',[r(1,701),imm(1e-5)],r(1,701));w.sfu('rsqrt',r(1,701),r(1,701))
    for part in range(2):
        base=w.affine_constants[t['w1parts'][part].rf_chunks];acc=r(16,768+part*32)
        w.vec('add',[acc,r(16,832+part*32)],acc)
        w.vec('fma',[w._rf(14,16,base),r(1,700),acc],acc)
        w.vec('fma',[acc,r(1,701),w._rf(14,16,base+16)],r(16,32+part*16))
    w._gelu_persistent(32,32)
