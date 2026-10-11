"""P1 first new positions use all 16 SMs; W2 split along K and N."""
from .compiler import Tensor, columns,emit_barrier

def emit_p1_fused_step_layer(lines,workers,shared,x,weights,d,f,h,hd,name,past,k_out,v_out):
    rows=2
    assert len(workers)==32
    qkv=shared.alloc(rows,3*d);ctx=shared.alloc(rows,d);res=shared.alloc(rows,d)
    act=shared.alloc(rows,f);y=shared.alloc(rows,d);ln1=shared.alloc(rows,d);ln2=shared.alloc(rows,d)
    workers[0].layernorm(x,weights['ln1_g'],weights['ln1_b'],ln1,rows,d,name+'ln1')
    emit_barrier(lines,workers)
    for i,w in enumerate(workers[:24]):
        w.gemm(ln1,weights['wqkv'],qkv,rows,d,3*d,name+'qkv'+str(i),n_lo=i*32,n_hi=(i+1)*32)
    emit_barrier(lines,workers)
    for head,w in enumerate(workers[:8]):
        for member in range(2):
            w.attention(Tensor(qkv.offset+member*3*d,(1,3*d)),Tensor(ctx.offset+member*d,(1,d)),
                        k_out[member],v_out[member],1,past,d,h,hd,name+'a'+str(member),head_lo=head,head_hi=head+1)
    emit_barrier(lines,workers)
    for i,w in enumerate(workers[:16]):
        lo,hi=columns(d,16,i)
        w.gemm(ctx,weights['wo'],res,rows,d,d,name+'wo',n_lo=lo,n_hi=hi,epilogue='residual',residual=x)
    emit_barrier(lines,workers)
    workers[0].layernorm(res,weights['ln2_g'],weights['ln2_b'],ln2,rows,d,name+'ln2')
    emit_barrier(lines,workers)
    for i,w in enumerate(workers):
        lo,hi=columns(f,32,i)
        w.gemm(ln2,weights['w1'],act,rows,d,f,name+'w1',n_lo=lo,n_hi=hi,epilogue='bias_gelu',bias=weights['b1'])
    emit_barrier(lines,workers)
    part=shared.alloc(32,rows,d)
    for i,w in enumerate(workers):
        kg,ng=divmod(i,8);klo=kg*(f//4);nlo=ng*(d//8)
        w.gemm(Tensor(act.offset+klo,(rows,f//4),row_stride=f),Tensor(weights['w2'].offset+klo*d,(f//4,d)),
               Tensor(part.offset+i*rows*d,(rows,d)),rows,f//4,d,name+'w2',n_lo=nlo,n_hi=nlo+d//8)
    emit_barrier(lines,workers)
    for i,w in enumerate(workers[:8]):
        w.reduce_w2_four_n8(part,y,res,weights['b2'],rows,d,i,0,rows,name+'red')
    emit_barrier(lines,workers)
    return y
