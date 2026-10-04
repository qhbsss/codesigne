"""Overlap LayerNorm statistics with its following GEMV.

LN(x)W = rsqrt(var+eps) * ((x*gamma)W-mu*(gamma W)) + beta W.
The two fixed projections are computed from the actual runtime parameters.
"""
from .compiler import hbm, imm

def affine_gemm(w,x,gamma,beta,b,out,m,k,n,name,n_lo=0,n_hi=None,epilogue=None,bias=None):
    assert m==1 and k==128 and gamma.space==beta.space=='RF'
    n_hi=n if n_hi is None else n_hi
    width=n_hi-n_lo
    assert width==16 and b.rf_chunks
    r=lambda count,off=0:w._rf(15,count,off)
    w.lines.extend(w.pending_weights.pop(b.rf_chunks,[]))
    if not hasattr(w,'affine_constants'):w.affine_constants={}
    key=b.rf_chunks
    if key not in w.affine_constants:
        base=len(w.affine_constants)*32
        w.affine_constants[key]=base
        w.vec('add',[imm(0),imm(0)],r(width,768))
        w.vec('add',[imm(0),imm(0)],r(width,896))
        for lane,off,start,depth in b.rf_chunks:
            bv=w._rf(lane,depth*width,off)
            w.emit('MMA.ACC',a=w._rf(gamma.lane,depth,gamma.offset+start),b=bv,acc=r(width,768),m=1,n=width,k=depth,event=None)
            w.emit('MMA.ACC',a=w._rf(beta.lane,depth,beta.offset+start),b=bv,acc=r(width,896),m=1,n=width,k=depth,event=None)
        w.st(r(width,768),w._sh(base,width))
        w.st(r(width,896),w._sh(base+16,width))
    base=w.affine_constants[key]
    w.ld(w._sh(base,width),r(width,512))
    w.ld(w._sh(base+16,width),r(width,544))
    if bias is not None:w.ld(hbm(bias.offset+n_lo,width),r(width,576))
    w.ld(hbm(x.offset,k),r(k))
    w.vec('add',[imm(0),imm(0)],r(width,768))
    w.vec('add',[imm(0),imm(0)],r(width,896))
    w.vec('mul',[r(k),w._rf(gamma.lane,k,gamma.offset)],r(k,128))
    w.reduce('sum',r(k),r(1,700))
    w.vec('mul',[r(k),r(k)],r(k,256))
    for ci,(lane,off,start,depth) in enumerate(b.rf_chunks):
        w.emit('MMA.ACC',a=r(depth,128+start),b=w._rf(lane,depth*width,off),acc=r(width,768 if ci%2==0 else 896),m=1,n=width,k=depth,event=None)
    w.reduce('sum',r(k,256),r(1,701))
    w.vec('mul',[r(1,700),imm(1/k)],r(1,700))
    w.vec('mul',[r(1,701),imm(1/k)],r(1,701))
    w.vec('mul',[r(1,700),r(1,700)],r(1,702))
    w.vec('sub',[r(1,701),r(1,702)],r(1,701))
    w.vec('max',[r(1,701),imm(0)],r(1,701))
    w.vec('add',[r(1,701),imm(1e-5)],r(1,701))
    w.sfu('rsqrt',r(1,701),r(1,701))
    w.vec('add',[r(width,768),r(width,896)],r(width,768))
    w.vec('mul',[r(1,700),r(1,701)],r(1,702))
    w.vec('mul',[r(1,702),imm(-1)],r(1,702))
    w.vec('fma',[r(width,512),r(1,702),r(width,544)],r(width,544))
    w.vec('fma',[r(width,768),r(1,701),r(width,544)],r(width,768))
    if epilogue=='bias_gelu':
        w.vec('add',[r(width,768),r(width,576)],r(width,32))
        w._gelu_persistent(width,32)
        stored=r(width,96)
    else:
        assert epilogue is None
        stored=r(width,768)
    if out.space!='RF':w.st(stored,hbm(out.offset+n_lo,width))
