"""Treat both prompts as one 128-row dense matrix; attention remains per batch."""
from .compiler import Builder,Scratch,Tensor,columns,emit_barrier
from .schedule import _open_workers,_weights,_commit,emit_p1_fused_step_layer
from codesign.challenge.abi import build_layout
from codesign.challenge.workload import MODELS
from .compiler import hbm

def generate_m1_p1():
    model=MODELS['M1'];d,f,h,hd=model.width,model.ffn,model.heads,model.head_width
    layout=build_layout(model,'P1');lines=[];origin=layout.symbols['scratch'].address//4
    workers,shared=_open_workers(layout,lines,'mp',origin,[i%16 for i in range(32)],shared_bytes=0)
    probe=workers[0];x=probe.symbol('input/prompt');output=probe.symbol('output/hidden');rows=128
    for layer in range(model.layers):
        w=_weights(probe,f'layer{layer}/');name=f'mp{layer}';qkv=shared.alloc(rows,3*d);ctx=shared.alloc(rows,d)
        res=shared.alloc(rows,d);ln1=shared.alloc(rows,d);ln2=shared.alloc(rows,d);act=shared.alloc(rows,f);y=shared.alloc(rows,d)
        kout=probe.symbol(f'layer{layer}/new_k');vout=probe.symbol(f'layer{layer}/new_v')
        for i,b in enumerate(workers):
            lo,hi=columns(rows,32,i);b.layernorm(x,w['ln1_g'],w['ln1_b'],ln1,rows,d,name+'ln1',lo,hi)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[:16]):
            b.gemm_n_tile=48
            b.gemm(ln1,w['wqkv'],qkv,rows,d,3*d,name+'qkv',n_lo=i*48,n_hi=(i+1)*48)
            b.gemm_n_tile=64
        emit_barrier(lines,workers)
        for i,b in enumerate(workers):
            member,local=divmod(i,16);head,half=divmod(local,2)
            q=Tensor(qkv.offset+member*64*3*d,(64,3*d));k=Tensor(kout.offset+member*h*65*hd,kout.shape);v=Tensor(vout.offset+member*h*65*hd,vout.shape)
            b.export_prompt_kv_rows(q,k,v,half*32,(half+1)*32,d,head,hd,name+'export')
        emit_barrier(lines,workers)
        for i,b in enumerate(workers):
            member,local=divmod(i,16);head,half=divmod(local,2)
            q=Tensor(qkv.offset+member*64*3*d,(64,3*d));c=Tensor(ctx.offset+member*64*d,(64,d))
            k=Tensor(kout.offset+member*h*65*hd,kout.shape);v=Tensor(vout.offset+member*h*65*hd,vout.shape)
            b.attention_prompt_slice(q,c,k,v,64,d,hd,name+'att',head,half*2,half*2+2)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[:16]):
            b.gemm(ctx,w['wo'],res,rows,d,d,name+'wo',n_lo=i*16,n_hi=(i+1)*16,epilogue='residual',residual=x)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers):
            lo,hi=columns(rows,32,i);b.layernorm(res,w['ln2_g'],w['ln2_b'],ln2,rows,d,name+'ln2',lo,hi)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers):
            b.gemm(ln2,w['w1'],act,rows,d,f,name+'w1',n_lo=i*32,n_hi=(i+1)*32,epilogue='bias_gelu',bias=w['b1'])
        emit_barrier(lines,workers)
        part=shared.alloc(32,rows,d)
        for i,b in enumerate(workers):
            kg,ng=divmod(i,4);klo=kg*128;nlo=ng*64
            b.gemm(Tensor(act.offset+klo,(rows,128),row_stride=f),Tensor(w['w2'].offset+klo*d,(128,d)),
                   Tensor(part.offset+i*rows*d,(rows,d)),rows,128,d,name+'w2',n_lo=nlo,n_hi=nlo+64)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers):
            rg,ng=divmod(i,4);lo,hi=columns(rows,8,rg);col=ng*64
            b.ld(hbm(w['b2'].offset+col,64),b._rf(6,64))
            for row in range(lo,hi,8):
                n=8*64
                b.ld(hbm(part.offset+ng*rows*d+row*d+col,n,[8,64],[d,1]),b._rf(2,n))
                for kg in range(1,8):
                    lane=kg%2;b.ld(hbm(part.offset+(kg*4+ng)*rows*d+row*d+col,n,[8,64],[d,1]),b._rf(lane,n))
                    b.vec('add',[b._rf(2,n),b._rf(lane,n)],b._rf(2,n))
                b.ld(hbm(res.offset+row*d+col,n,[8,64],[d,1]),b._rf(5,n))
                b.vec('add',[b._rf(2,n),b._rf(5,n)],b._rf(2,n))
                for rr in range(8):b.vec('add',[b._rf(2,64,rr*64),b._rf(6,64)],b._rf(7,64,rr*64))
                b.st(b._rf(7,n),hbm(y.offset+row*d+col,n,[8,64],[d,1]))
        emit_barrier(lines,workers);x=y
    for member in range(2):
        for i,b in enumerate(workers):
            lo,hi=columns(d,32,i);b.store_slice(Tensor(x.offset+member*64*d,(64,d)),output.offset+member*65*d,64,d,lo,hi,'out')
    _commit(lines,0)
    x=probe.symbol('input/step0')
    for layer in range(model.layers):
        k=probe.symbol(f'layer{layer}/new_k');v=probe.symbol(f'layer{layer}/new_v')
        x=emit_p1_fused_step_layer(lines,workers,shared,x,_weights(probe,f'layer{layer}/'),d,f,h,hd,f'ms{layer}',64,
                                 [Tensor(k.offset+m*h*65*hd,k.shape) for m in range(2)],
                                 [Tensor(v.offset+m*h*65*hd,v.shape) for m in range(2)])
    for member in range(2):
        for i,b in enumerate(workers):
            lo,hi=columns(d,32,i);b.store_slice(Tensor(x.offset+member*d,(1,d)),output.offset+(member*65+64)*d,1,d,lo,hi,'stepout')
    for b in workers:b.finish()
    return '\n'.join(lines)+'\n',layout,shared.cursor
