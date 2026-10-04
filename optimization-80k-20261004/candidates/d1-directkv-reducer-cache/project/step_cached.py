"""Two prompt-new positions: full stage weights in RF, bounded boot waves."""
from math import sqrt,pi
from .compiler import hbm,imm,emit_barrier,Tensor,columns

def matrix(w,a,b,out,k,n,col,gelu=False,residual=None):
    width,chunks=b;acc=w._rf(14,2*width)
    w.vec('add',[imm(0),imm(0)],acc)
    for ki in range(0,k,64):
        w.ld(hbm(a.offset+ki,128,[2,64],[a.row_stride or k,1]),w._rf(15,128))
        for part in range(max(1,width//16)):
            if width==32:_,lane,off,_,_=next(c for c in chunks if c[0]==part and c[3]==ki)
            else:lane,off,_,_=next(c for c in chunks if c[2]==ki)
            cc=w._rf(14,32,part*16);cc.update(shape=[2,16],strides=[width,1])
            w.emit('MMA.ACC',a=w._rf(15,128),b=w._rf(lane,1024,off),acc=cc,m=2,n=16,k=64,event=None)
    if gelu:
        w.vec('add',[imm(1),imm(0)],w._rf(11,2))
        w.emit('MMA.ACC',a=w._rf(11,2),b=w._rf(14,width,512),acc=acc,m=2,n=width,k=1,event=None)
        t=w._rf(15,2*width,128);half=w._rf(15,2*width,256)
        w.vec('mul',[acc,acc],t);w.vec('fma',[t,imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],t)
        w.vec('mul',[t,acc],t);w.sfu('tanh',t,t);w.vec('mul',[acc,imm(.5)],half);w.vec('fma',[half,t,half],acc)
    if residual is not None:
        w.ld(hbm(residual.offset+col,2*width,[2,width],[n,1]),w._rf(15,2*width,128))
        w.vec('add',[acc,w._rf(15,2*width,128)],acc)
    w.st(acc,hbm(out.offset+col,2*width,[2,width],[n,1]))

def emit_p1_fused_step_layer(lines,workers,shared,x,weights,d,f,h,hd,name,past,k_out,v_out,retained_w2=None):
    rows=2
    qkv=shared.alloc(rows,3*d);ctx=shared.alloc(rows,d);res=shared.alloc(rows,d)
    act=shared.alloc(rows,f);y=shared.alloc(rows,d);ln1=shared.alloc(rows,d);ln2=shared.alloc(rows,d)
    workers[0].layernorm(x,weights['ln1_g'],weights['ln1_b'],ln1,rows,d,name+'ln1');emit_barrier(lines,workers)
    cached={}
    for j,i in enumerate([off+base for off in range(2) for base in range(0,24,2)]):
        cached[i]=workers[i].cache(weights['wqkv'],i*32,(i+1)*32,64)
        if j%4==3:emit_barrier(lines,workers)
    emit_barrier(lines,workers)
    for i,w in enumerate(workers[:24]):matrix(w,ln1,cached[i],qkv,d,3*d,i*32)
    emit_barrier(lines,workers)
    for head,w in enumerate(workers[:8]):
        for member in range(2):
            w.attention(Tensor(qkv.offset+member*3*d,(1,3*d)),Tensor(ctx.offset+member*d,(1,d)),k_out[member],v_out[member],1,past,d,h,hd,name+'a'+str(member),head_lo=head,head_hi=head+1)
    emit_barrier(lines,workers)
    cached={}
    for j,i in enumerate([off+base for off in range(4) for base in range(0,16,4)]):
        cached[i]=workers[i].cache(weights['wo'],i*16,(i+1)*16,64)
        if j%4==3:emit_barrier(lines,workers)
    emit_barrier(lines,workers)
    for i,w in enumerate(workers[:16]):matrix(w,ctx,cached[i],res,d,d,i*16,residual=x)
    emit_barrier(lines,workers)
    workers[0].layernorm(res,weights['ln2_g'],weights['ln2_b'],ln2,rows,d,name+'ln2');emit_barrier(lines,workers)
    cached={}
    for j,i in enumerate([off+base for off in range(2) for base in range(0,32,2)]):
        w=workers[i];cached[i]=w.cache(weights['w1'],i*32,(i+1)*32,64);w.cache_vector(weights['b1'],i*32,32,1536)
        if j%7==6:emit_barrier(lines,workers)
    emit_barrier(lines,workers)
    for i,w in enumerate(workers):matrix(w,ln2,cached[i],act,d,f,i*32,gelu=True)
    emit_barrier(lines,workers)
    part=shared.alloc(32,rows,d);cached={}
    if retained_w2 is None:
        w2workers=workers;syncworkers=workers
        for j,i in enumerate([off+base for off in range(2) for base in range(0,32,2)]):
            kg,ng=divmod(i,8);cached[i]=workers[i].cache(Tensor(weights['w2'].offset+kg*256*d,(256,d)),ng*32,(ng+1)*32,64)
            if j%4==3:emit_barrier(lines,workers)
        emit_barrier(lines,workers)
    else:
        w2workers,saved=retained_w2;syncworkers=workers+w2workers
        cached={i:saved[i]['w2'] for i in range(32)}
        emit_barrier(lines,syncworkers)
    for i,w in enumerate(w2workers):
        kg,ng=divmod(i,8);matrix(w,Tensor(act.offset+kg*256,(rows,256),row_stride=f),cached[i],Tensor(part.offset+i*rows*d,(rows,d)),256,d,ng*32)
    emit_barrier(lines,syncworkers)
    for i,w in enumerate(workers[:8]):w.reduce_w2_four_n8(part,y,res,weights['b2'],rows,d,i,0,rows,name+'red')
    emit_barrier(lines,workers)
    return y
