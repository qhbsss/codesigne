"""Preserve banks0..7 through the new-token stages; consume layer0/2 W2 once."""
from contextlib import contextmanager
from .compiler import hbm,Tensor,emit_barrier
from .step_cached import matrix

@contextmanager
def high_workspace(w):
    original=w._rf
    w._rf=lambda lane,*args,**kw:original(lane+8,*args,**kw)
    try:yield
    finally:w._rf=original

def cache_high(w,t,col):
    chunks=[]
    for part in range(4):
        w.ld(hbm(t.offset+part*64*t.shape[1]+col,1024,[64,16],[t.shape[1],1]),w._rf(8+part,1024))
        chunks.append((8+part,0,part*64,64))
    return 16,chunks

def load_columns(lines,workers,t,count):
    result={}
    order=[off+base for off in range(4) for base in range(0,count,4)]
    for j,i in enumerate(order):
        result[i]=cache_high(workers[i],t,i*16)
        if j%8==7:emit_barrier(lines,workers)
    emit_barrier(lines,workers)
    return result

def emit_p1_fused_step_layer(lines,workers,shared,x,weights,d,f,h,hd,name,past,k_out,v_out,retained_w2=None):
    assert len(workers)==64
    rows=2;qkv=shared.alloc(rows,3*d);ctx=shared.alloc(rows,d);res=shared.alloc(rows,d)
    act=shared.alloc(rows,f);y=shared.alloc(rows,d);ln1=shared.alloc(rows,d);ln2=shared.alloc(rows,d)
    for member,w in enumerate(workers[:2]):
        with high_workspace(w):
            w.layernorm(x,weights['ln1_g'],weights['ln1_b'],ln1,rows,d,name+'ln1',row_lo=member,row_hi=member+1)
    emit_barrier(lines,workers)
    cached=load_columns(lines,workers,weights['wqkv'],48)
    for i,w in enumerate(workers[:48]):matrix(w,ln1,cached[i],qkv,d,3*d,i*16)
    emit_barrier(lines,workers)
    for i,w in enumerate(workers[:16]):
        member,head=divmod(i,8)
        with high_workspace(w):
            w.attention(Tensor(qkv.offset+member*3*d,(1,3*d)),Tensor(ctx.offset+member*d,(1,d)),k_out[member],v_out[member],1,past,d,h,hd,name+'a'+str(member),head_lo=head,head_hi=head+1)
    emit_barrier(lines,workers)
    cached=load_columns(lines,workers,weights['wo'],16)
    for i,w in enumerate(workers[:16]):matrix(w,ctx,cached[i],res,d,d,i*16,residual=x)
    emit_barrier(lines,workers)
    for member,w in enumerate(workers[:2]):
        with high_workspace(w):
            w.layernorm(res,weights['ln2_g'],weights['ln2_b'],ln2,rows,d,name+'ln2',row_lo=member,row_hi=member+1)
    emit_barrier(lines,workers)
    cached=load_columns(lines,workers,weights['w1'],64)
    for i,w in enumerate(workers):
        w.cache_vector(weights['b1'],i*16,16,1536)
        matrix(w,ln2,cached[i],act,d,f,i*16,gelu=True)
    emit_barrier(lines,workers)
    part=shared.alloc(32,rows,d)
    if retained_w2 is None:
        # Layer0's retained banks are dead now. Layer2 stays in WG0..31.
        owners=workers[32:];cached={}
        for j,i in enumerate([off+base for off in range(2) for base in range(0,32,2)]):
            kg,ng=divmod(i,8)
            cached[i]=owners[i].cache(Tensor(weights['w2'].offset+kg*256*d,(256,d)),ng*32,(ng+1)*32,64)
            if j%4==3:emit_barrier(lines,workers)
    else:
        owners,saved=retained_w2;cached={i:saved[i]['w2'] for i in range(32)}
    emit_barrier(lines,workers)
    for i,w in enumerate(owners):
        kg,ng=divmod(i,8)
        matrix(w,Tensor(act.offset+kg*256,(rows,256),row_stride=f),cached[i],Tensor(part.offset+i*rows*d,(rows,d)),256,d,ng*32)
    emit_barrier(lines,workers)
    for i,w in enumerate(workers[:8]):
        with high_workspace(w):
            w.reduce_w2_four_n8(part,y,res,weights['b2'],rows,d,i,0,rows,name+'red')
    emit_barrier(lines,workers)
    return y
