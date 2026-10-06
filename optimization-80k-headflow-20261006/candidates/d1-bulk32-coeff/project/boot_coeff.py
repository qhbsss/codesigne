"""Project actual gamma/beta from RF-resident weights before the online loop."""
from .compiler import hbm,imm

def prepare(w,gamma,beta,b,bias=None,col=0):
    k=128;n=16;r=lambda count,off=0:w._rf(15,count,off);base=beta.offset
    if not hasattr(w,'affine_constants'):w.affine_constants={}
    w.affine_constants[b.rf_chunks]=base
    w.vec('add',[w._rf(gamma.lane,k,gamma.offset),imm(0)],r(k))
    w.vec('add',[w._rf(beta.lane,k,beta.offset),imm(0)],r(k,128))
    for ci,(bank,off,start,depth) in enumerate(b.rf_chunks):
        acc=r(32,512+ci*64);w.vec('add',[imm(0),imm(0)],acc)
        av=r(2*depth,start);av.update(shape=[2,depth],strides=[128,1])
        w.emit('MMA.ACC',a=av,b=w._rf(bank,depth*n,off),acc=acc,m=2,n=n,k=depth,event=None)
    w.vec('add',[r(n,512),r(n,576)],r(n,640))
    w.vec('mul',[r(n,640),imm(-1)],w._rf(14,n,base))
    w.vec('add',[r(n,528),r(n,592)],r(n,640))
    if bias is not None:
        w.ld(hbm(bias.offset+col,n),r(n,672))
        w.vec('add',[r(n,640),r(n,672)],w._rf(14,n,base+16))
        if hasattr(w,'reducer_column'):w.ld(hbm(w.reducer_bias_source[base],8),w._rf(14,8,base+48))
    else:w.vec('add',[r(n,640),imm(0)],w._rf(14,n,base+16))
