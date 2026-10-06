"""Tiny two-row suffix: stream N64 weights; keep dormant final W1 caches."""
from math import sqrt,pi
from .compiler import hbm,imm,emit_barrier,Tensor
from .step_cached import matrix

def streamed(lines,allworkers,tasks,k,n,width,gelu=False,residual=None,store=True):
    # Each task is (worker, input, actual weight, output, weight column).
    def load(w,weight,col,ki,panel):
        bank=((ki//64)%2)*4+panel
        w.ld(hbm(weight.offset+ki*n+col+panel*16,1024,[64,16],[n,1]),w._rf(bank,1024))
    for j,(w,a,weight,out,col) in enumerate(tasks):
        for panel in range(width//16):load(w,weight,col,0,panel)
        if j%4==3:emit_barrier(lines,allworkers)
    emit_barrier(lines,allworkers)
    for w,a,weight,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,2*width))
    for ki in range(0,k,64):
        for w,a,weight,out,col in tasks:w.ld(hbm(a.offset+ki,128,[2,64],[a.row_stride or k,1]),w._rf(15,128))
        for panel in range(width//16):
            for j,(w,a,weight,out,col) in enumerate(tasks):
                acc=w._rf(14,32,panel*16);acc.update(shape=[2,16],strides=[width,1])
                w.emit('MMA.ACC',a=w._rf(15,128),b=w._rf(((ki//64)%2)*4+panel,1024),acc=acc,m=2,n=16,k=64,event=None)
                if ki+64<k:
                    load(w,weight,col,ki+64,panel)
                    if j%4==3:emit_barrier(lines,allworkers)
    for w,a,weight,out,col in tasks:
        acc=w._rf(14,2*width)
        if gelu:
            bias=getattr(w,'stream_bias')
            w.ld(hbm(bias.offset+col,width),w._rf(14,width,512))
            w.vec('add',[imm(1),imm(0)],w._rf(11,2))
            w.emit('MMA.ACC',a=w._rf(11,2),b=w._rf(14,width,512),acc=acc,m=2,n=width,k=1,event=None)
            t=w._rf(15,2*width,128);half=w._rf(15,2*width,256)
            w.vec('mul',[acc,acc],t);w.vec('fma',[t,imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],t)
            w.vec('mul',[t,acc],t);w.sfu('tanh',t,t);w.vec('mul',[acc,imm(.5)],half);w.vec('fma',[half,t,half],acc)
        if residual is not None:
            w.ld(hbm(residual.offset+col,2*width,[2,width],[n,1]),w._rf(15,2*width,128))
            w.vec('add',[acc,w._rf(15,2*width,128)],acc)
        if store:w.st(acc,hbm(out.offset+col,2*width,[2,width],[n,1]))

def emit_p1_fused_step_layer(lines,workers,shared,x,weights,d,f,h,hd,name,past,k_out,v_out,retained_w2=None,retained_w1=None):
    rows=2
    qkv=shared.alloc(rows,3*d);ctx=shared.alloc(rows,d);res=shared.alloc(rows,d)
    act=shared.alloc(rows,f);y=shared.alloc(rows,d);ln1=shared.alloc(rows,d);ln2=shared.alloc(rows,d)
    for member,w in enumerate(workers[:2]):w.layernorm(x,weights['ln1_g'],weights['ln1_b'],ln1,rows,d,name+'ln1',row_lo=member,row_hi=member+1)
    emit_barrier(lines,workers)
    streamed(lines,workers,[(w,ln1,weights['wqkv'],qkv,i*64) for i,w in enumerate(workers[:12])],d,3*d,64)
    emit_barrier(lines,workers)
    for index,w in enumerate(workers[:16]):
        member,head=divmod(index,8)
        w.attention(Tensor(qkv.offset+member*3*d,(1,3*d)),Tensor(ctx.offset+member*d,(1,d)),k_out[member],v_out[member],1,past,d,h,hd,name+'a'+str(member),head_lo=head,head_hi=head+1)
    emit_barrier(lines,workers)
    streamed(lines,workers,[(w,ctx,weights['wo'],res,i*64) for i,w in enumerate(workers[:4])],d,d,64,residual=x)
    emit_barrier(lines,workers)
    for member,w in enumerate(workers[:2]):w.layernorm(res,weights['ln2_g'],weights['ln2_b'],ln2,rows,d,name+'ln2',row_lo=member,row_hi=member+1)
    emit_barrier(lines,workers)
    used=workers[:16] if retained_w1 is None else workers[:8]
    for w in used:w.stream_bias=weights['b1']
    streamed(lines,workers,[(w,ln2,weights['w1'],act,i*64) for i,w in enumerate(used)],d,f,64,gelu=True)
    if retained_w1 is not None:
        for i,w in enumerate(workers[16:32],16):
            w.cache_vector(weights['b1'],i*32,32,1536)
            matrix(w,ln2,retained_w1[i]['w1'],act,d,f,i*32,gelu=True)
    emit_barrier(lines,workers)
    part=shared.alloc(32,rows,d)
    if retained_w2 is None:
        tasks=[]
        for i,w in enumerate(workers[:16]):
            kg,ng=divmod(i,4)
            tasks.append((w,Tensor(act.offset+kg*256,(rows,256),row_stride=f),Tensor(weights['w2'].offset+kg*256*d,(256,d)),None,ng*64))
        streamed(lines,workers,tasks,256,d,64,store=False)
        for i,w in enumerate(workers[:16]):
            kg,ng=divmod(i,4)
            for half in range(2):
                src=w._rf(14,64,half*32);src.update(shape=[2,32],strides=[64,1])
                task=kg*8+ng*2+half;col=ng*64+half*32
                w.st(src,hbm(part.offset+task*rows*d+col,64,[2,32],[d,1]))
        syncworkers=workers
    else:
        w2workers,saved=retained_w2;syncworkers=workers+w2workers
        emit_barrier(lines,syncworkers)
        for i,w in enumerate(w2workers):
            kg,ng=divmod(i,8)
            matrix(w,Tensor(act.offset+kg*256,(rows,256),row_stride=f),saved[i]['w2'],Tensor(part.offset+i*rows*d,(rows,d)),256,d,ng*32)
    emit_barrier(lines,syncworkers)
    for i,w in enumerate(workers[:8]):w.reduce_w2_four_n8(part,y,res,weights['b2'],rows,d,i,0,rows,name+'red')
    emit_barrier(lines,workers)
    return y
