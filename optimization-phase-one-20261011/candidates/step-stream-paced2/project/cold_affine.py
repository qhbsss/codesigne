"""First use: pack gamma, beta and runtime x*gamma into one M3 GEMM."""
import json
from .compiler import hbm,imm

def execute(w,x,gamma,beta,b,out,k,n,col,width,epilogue,bias):
    assert k==128 and width==16 and len(b.rf_chunks)==2
    r=lambda count,off=0:w._rf(15,count,off)
    base=beta.offset;w.affine_constants[b.rf_chunks]=base
    w.ld(hbm(x.offset,k),r(k))
    w.lines.extend(w.pending_weights.pop(b.rf_chunks,[]))
    w.vec('add',[w._rf(gamma.lane,k,gamma.offset),imm(0)],r(k,128))
    w.vec('add',[w._rf(beta.lane,k,beta.offset),imm(0)],r(k,256))
    w.vec('mul',[r(k),r(k,128)],r(k,384))
    if bias is not None and hasattr(w,'reducer_column'):
        w.vec('add',[r(8,w.reducer_column),imm(0)],w._rf(14,8,base+56))
    w.reduce('sum',r(k),r(1,700));w.vec('mul',[r(k),r(k)],r(k,512))
    for ci in range(2):w.vec('add',[imm(0),imm(0)],r(3*width,768+ci*64))
    for ci,(lane,off,start,depth) in enumerate(b.rf_chunks):
        a=r(3*depth,128+start);a.update(shape=[3,depth],strides=[128,1])
        acc=r(3*width,768+ci*64);acc.update(shape=[3,width],strides=[width,1])
        w.emit('MMA.ACC',a=a,b=w._rf(lane,depth*width,off),acc=acc,m=3,n=width,k=depth,event=None)
    w.reduce('sum',r(k,512),r(1,701))
    w.vec('mul',[r(1,700),imm(1/k)],r(1,700));w.vec('mul',[r(1,700),r(1,700)],r(1,702))
    w.vec('sub',[imm(1e-5),r(1,702)],r(1,702));w.vec('fma',[r(1,701),imm(1/k),r(1,702)],r(1,701))
    w.vec('max',[r(1,701),imm(1e-5)],r(1,701));w.sfu('rsqrt',r(1,701),r(1,701))
    w.vec('add',[r(width,768),r(width,832)],r(width,900));w.vec('mul',[r(width,900),imm(-1)],w._rf(14,width,base))
    w.vec('add',[r(width,784),r(width,848)],r(width,900))
    if bias is not None:
        w.ld(hbm(bias.offset+col,width),r(width,928));w.vec('add',[r(width,900),r(width,928)],w._rf(14,width,base+16))
        if hasattr(w,'reducer_column'):w.ld(hbm(w.reducer_bias_source[base],8),w._rf(14,8,base+48))
    else:w.vec('add',[r(width,900),imm(0)],w._rf(14,width,base+16))
    w.vec('add',[r(width,800),r(width,864)],r(width,768))
    w.vec('fma',[w._rf(14,width,base),r(1,700),r(width,768)],r(width,768))
    w.vec('fma',[r(width,768),r(1,701),w._rf(14,width,base+16)],r(width,32 if epilogue=='bias_gelu' else 768))
    if epilogue=='bias_gelu':
        if getattr(w,'ffn_wait_events',[]):w.lines.append('BARRIER '+json.dumps(dict(wgs=[w.wg],events=w.ffn_wait_events),separators=(',',':')))
        for weight in getattr(w,'ffn_prefetch',[]):w.lines.extend(w.pending_weights.pop(weight.rf_chunks,[]))
        w._gelu_persistent(width,32);stored=r(width,96)
    else:assert epilogue is None;stored=r(width,768)
    if out.space!='RF' and not (bias is None and n==3*k and col>=k):w.st(stored,hbm(out.offset+col,width))
