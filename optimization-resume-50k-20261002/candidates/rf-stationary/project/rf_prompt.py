"""Current-layer RF weights, synchronized 64-WG dense kernels."""
from math import sqrt,pi
from .compiler import Builder,Tensor,hbm,imm,add,mul,minimum,sub,columns,emit_barrier
from .schedule import _open_workers,_weights,_commit
from codesign.challenge.workload import MODELS
from codesign.challenge.abi import build_layout

class RFBuilder(Builder):
    def reset_cache(self):self.used=[0]*7
    def cache(self,t,lo,hi,depth):
        chunks=[];width=hi-lo
        for k in range(0,t.shape[0],depth):
            dk=min(depth,t.shape[0]-k);n=dk*width
            lane=next(i for i in range(7) if self.used[i]+n<=([2048]*6+[1024])[i]);off=self.used[lane];self.used[lane]+=n
            self.ld(hbm(t.offset+k*t.shape[1]+lo,n,[dk,width],[t.shape[1],1]),self._rf(lane,n,off))
            chunks.append((lane,off,k,dk))
        return (width,chunks)
    def cache_vector(self,t,lo,n,off):self.ld(hbm(t.offset+lo,n),self._rf(6,n,off))
    def norm(self,x,y,d,gamma,beta,name,lo,hi):
        r=lambda n,o=0:self._rf(7,n,o)
        with self.loop(name,lo,hi) as row:
            self.ld(hbm(add(x.offset,mul(row,d)),d),r(d))
            self.reduce('sum',r(d),r(1,1800));self.vec('mul',[r(1,1800),imm(1/d)],r(1,1800))
            self.vec('sub',[r(d),r(1,1800)],r(d,256));self.vec('mul',[r(d,256),r(d,256)],r(d,512))
            self.reduce('sum',r(d,512),r(1,1800));self.vec('mul',[r(1,1800),imm(1/d)],r(1,1800))
            self.vec('add',[r(1,1800),imm(1e-5)],r(1,1800));self.sfu('rsqrt',r(1,1800),r(1,1800))
            self.vec('mul',[r(d,256),r(1,1800)],r(d,256))
            self.vec('mul',[r(d,256),self._rf(6,d,gamma)],r(d,256));self.vec('add',[r(d,256),self._rf(6,d,beta)],r(d,256))
            self.st(r(d,256),hbm(add(y.offset,mul(row,d)),d))
    def gemm_cached(self,a,b,out,rows,k,n,col,name,m_tile=32,epilogue=None,residual=None,bias_off=1536):
        width,chunks=b;stride=a.row_stride or k
        r=lambda count,off=0:self._rf(7,count,off)
        with self.loop(name,0,rows,m_tile) as row:
            nr=minimum(m_tile,sub(rows,row));cells=mul(nr,width);acc=self._rf(6,cells,1024)
            self.vec('add',[imm(0),imm(0)],acc)
            groups=[]
            for item in chunks:
                if groups and item[0]==groups[-1][-1][0] and item[1]==groups[-1][-1][1]+groups[-1][-1][3]*width and item[2]==groups[-1][-1][2]+groups[-1][-1][3] and item[3]==groups[-1][-1][3]:
                    groups[-1].append(item)
                else:groups.append([item])
            for gi,group in enumerate(groups):
                lane,off,klo,dk=group[0]
                if len(group)>1:
                    with self.loop(name+'k'+str(gi),0,len(group)) as ki:
                        kk=add(klo,mul(ki,dk));boff=add(off,mul(ki,dk*width))
                        self.ld(hbm(add(a.offset,mul(row,stride),kk),mul(nr,dk),[nr,dk],[stride,1]),r(mul(nr,dk)))
                        self.emit('MMA.ACC',a=r(mul(nr,dk)),b=self._rf(lane,dk*width,boff),acc=acc,m=nr,n=width,k=dk,event=None)
                else:
                    self.ld(hbm(add(a.offset,mul(row,stride),klo),mul(nr,dk),[nr,dk],[stride,1]),r(mul(nr,dk)))
                    self.emit('MMA.ACC',a=r(mul(nr,dk)),b=self._rf(lane,dk*width,off),acc=acc,m=nr,n=width,k=dk,event=None)
            if epilogue in ('gelu','resbias'):
                with self.loop(name+'b',0,nr) as rr:
                    self.vec('add',[self._rf(6,width,bias_off),imm(0)],r(width,add(1024,mul(rr,width))))
            if epilogue=='gelu':
                self.vec('add',[acc,r(cells,1024)],acc)
                self.vec('mul',[acc,acc],r(cells));self.vec('mul',[r(cells),acc],r(cells))
                self.vec('fma',[r(cells),imm(.044715),acc],r(cells));self.vec('mul',[r(cells),imm(sqrt(2/pi))],r(cells))
                self.sfu('tanh',r(cells),r(cells));self.vec('add',[r(cells),imm(1)],r(cells))
                self.vec('mul',[acc,imm(.5)],r(cells,512));self.vec('mul',[r(cells,512),r(cells)],acc)
            elif epilogue in ('res','resbias'):
                self.ld(hbm(add(residual.offset,mul(row,n),col),cells,[nr,width],[n,1]),r(cells))
                self.vec('add',[acc,r(cells)],acc)
                if epilogue=='resbias':self.vec('add',[acc,r(cells,1024)],acc)
            self.st(acc,hbm(add(out.offset,mul(row,n),col),cells,[nr,width],[n,1]))
    def export(self,qkv,k,v,head,row_lo,row_hi,name):
        with self.loop(name,row_lo,row_hi,8) as row:
            for comp,t in [(1,k),(2,v)]:
                self.ld(hbm(add(qkv.offset,mul(row,768),comp*256+head*32),256,[8,32],[768,1]),self._rf(7,256))
                self.st(self._rf(7,256),hbm(add(t.offset,head*65*32,mul(row,32)),256))
    def attn(self,qkv,ctx,k,v,head,blo,bhi,name):
        r=lambda n,o=0:self._rf(7,n,o)
        def score_view(key):
            x=r(64,add(512,mul(key,8)));x.update(shape=[8,8],strides=[64,1]);return x
        with self.loop(name,blo,bhi) as block:
            row=mul(block,8)
            self.ld(hbm(add(qkv.offset,mul(row,768),head*32),256,[8,32],[768,1]),r(256))
            self.vec('add',[imm(0),imm(0)],r(512,512))
            with self.loop(name+'k',0,add(block,1)) as key:
                self.ld(hbm(add(k.offset,head*65*32,mul(key,256)),256,[32,8],[1,32]),r(256,256))
                self.vec('add',[imm(0),imm(0)],r(64,1024))
                self.emit('MMA.ACC',a=r(256),b=r(256,256),acc=r(64,1024),m=8,n=8,k=32,event=None)
                self.vec('mul',[r(64,1024),imm(1/sqrt(32))],r(64,1024));self.vec('add',[r(64,1024),imm(0)],score_view(key))
            with self.loop(name+'s',0,8) as local:
                limit=add(row,local,1);slot=add(512,mul(local,64))
                self.reduce('max',r(limit,slot),r(1,1344));self.vec('sub',[r(limit,slot),r(1,1344)],r(limit,slot))
                self.sfu('exp',r(limit,slot),r(limit,slot));self.reduce('sum',r(limit,slot),r(1,1345));self.vec('div',[r(limit,slot),r(1,1345)],r(limit,slot))
            with self.loop(name+'z',0,7) as local:
                self.vec('add',[imm(0),imm(0)],r(sub(7,local),add(512,mul(local,64),row,local,1)))
            self.vec('add',[imm(0),imm(0)],r(256,1088))
            with self.loop(name+'v',0,add(block,1)) as key:
                self.ld(hbm(add(v.offset,head*65*32,mul(key,256)),256),r(256,256))
                self.emit('MMA.ACC',a=score_view(key),b=r(256,256),acc=r(256,1088),m=8,n=32,k=8,event=None)
            self.st(r(256,1088),hbm(add(ctx.offset,mul(row,256),head*32),256,[8,32],[256,1]))
    def reduce_partial(self,part,y,res,rows,col,row_lo,row_hi,name):
        with self.loop(name,row_lo,row_hi,32) as row:
            count=512;acc=self._rf(6,count,1024)
            self.ld(hbm(add(part.offset,(col//16)*rows*256,mul(row,256),col),count,[32,16],[256,1]),acc)
            for kg in range(1,4):
                self.ld(hbm(add(part.offset,(kg*16+col//16)*rows*256,mul(row,256),col),count,[32,16],[256,1]),self._rf(7,count))
                self.vec('add',[acc,self._rf(7,count)],acc)
            self.ld(hbm(add(res.offset,mul(row,256),col),count,[32,16],[256,1]),self._rf(7,count))
            self.vec('add',[acc,self._rf(7,count)],acc)
            with self.loop(name+'b',0,32) as rr:self.vec('add',[self._rf(6,16,1552),imm(0)],self._rf(7,16,add(1024,mul(rr,16))))
            self.vec('add',[acc,self._rf(7,count,1024)],acc)
            self.st(acc,hbm(add(y.offset,mul(row,256),col),count,[32,16],[256,1]))

def generate_m1_p1():
    layout=build_layout(MODELS['M1'],'P1');lines=[];origin=layout.symbols['scratch'].address//4
    workers,shared=_open_workers(layout,lines,'r',origin,[i%16 for i in range(64)],shared_bytes=0)
    # _open_workers uses Builder; replace it before invoking this function.
    p=workers[0];x=p.symbol('input/prompt');out=p.symbol('output/hidden');rows=128
    for layer in range(3):
        w=_weights(p,f'layer{layer}/');weights=[]
        for b in workers:b.reset_cache()
        for i,b in enumerate(workers):
            it={};it['qkv']=b.cache(w['wqkv'],i*12,(i+1)*12,128);weights.append(it)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[:32]):weights[i]['wo']=b.cache(w['wo'],i*8,(i+1)*8,64)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers):weights[i]['w1']=b.cache(w['w1'],i*16,(i+1)*16,32)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers):
            kg,ng=divmod(i,16);t=Tensor(w['w2'].offset+kg*256*256,(256,256));weights[i]['w2']=b.cache(t,ng*16,(ng+1)*16,32)
            b.cache_vector(w['b1'],i*16,16,1536);b.cache_vector(w['b2'],ng*16,16,1552)
            if i>=32:
                for j,key in enumerate(['ln1_g','ln1_b','ln2_g','ln2_b']):b.cache_vector(w[key],0,256,j*256)
        emit_barrier(lines,workers)
        qkv=shared.alloc(rows,768);ctx=shared.alloc(rows,256);res=shared.alloc(rows,256);ln1=shared.alloc(rows,256);ln2=shared.alloc(rows,256);act=shared.alloc(rows,1024);y=shared.alloc(rows,256)
        k=p.symbol(f'layer{layer}/new_k');v=p.symbol(f'layer{layer}/new_v')
        for i,b in enumerate(workers[32:]):b.norm(x,ln1,256,0,256,f'l{layer}n1',i*4,(i+1)*4)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers):b.gemm_cached(ln1,weights[i]['qkv'],qkv,rows,256,768,i*12,f'l{layer}q',m_tile=16)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[:32]):
            member,local=divmod(i,16);head,half=divmod(local,2)
            b.export(Tensor(qkv.offset+member*64*768,(64,768)),Tensor(k.offset+member*8*65*32,k.shape),Tensor(v.offset+member*8*65*32,v.shape),head,half*32,(half+1)*32,f'l{layer}e')
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[:32]):
            member,local=divmod(i,16);head,half=divmod(local,2)
            b.attn(Tensor(qkv.offset+member*64*768,(64,768)),Tensor(ctx.offset+member*64*256,(64,256)),Tensor(k.offset+member*8*65*32,k.shape),Tensor(v.offset+member*8*65*32,v.shape),head,half*4,(half+1)*4,f'l{layer}a')
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[:32]):b.gemm_cached(ctx,weights[i]['wo'],res,rows,256,256,i*8,f'l{layer}o',m_tile=32,epilogue='res',residual=x)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[32:]):b.norm(res,ln2,256,512,768,f'l{layer}n2',i*4,(i+1)*4)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers):b.gemm_cached(ln2,weights[i]['w1'],act,rows,256,1024,i*16,f'l{layer}f',epilogue='gelu')
        emit_barrier(lines,workers)
        part=shared.alloc(64,rows,256)
        for i,b in enumerate(workers):
            kg,ng=divmod(i,16)
            b.gemm_cached(Tensor(act.offset+kg*256,(rows,256),row_stride=1024),weights[i]['w2'],Tensor(part.offset+i*rows*256,(rows,256)),rows,256,256,ng*16,f'l{layer}w')
        emit_barrier(lines,workers)
        for i,b in enumerate(workers):
            rg,ng=divmod(i,16);b.reduce_partial(part,y,res,rows,ng*16,rg*32,(rg+1)*32,f'l{layer}r')
        emit_barrier(lines,workers);x=y
    for member in range(2):
        for i,b in enumerate(workers[:16]):b.store_slice(Tensor(x.offset+member*64*256,(64,256)),out.offset+member*65*256,64,256,i*16,(i+1)*16,'out')
    _commit(lines,0)
    # Use the validated conservative first-new-position generator.
    from .schedule import emit_p1_fused_step_layer
    x=p.symbol('input/step0');stepworkers=workers[:32]
    # The existing parallel-step accepts32; conservative original accepts8.
    from .parallel_step import emit_p1_fused_step_layer as parallel_step
    for layer in range(3):
        k=p.symbol(f'layer{layer}/new_k');v=p.symbol(f'layer{layer}/new_v')
        x=parallel_step(lines,stepworkers,shared,x,_weights(p,f'layer{layer}/'),256,1024,8,32,f's{layer}',64,
                        [Tensor(k.offset+m*8*65*32,k.shape) for m in range(2)],[Tensor(v.offset+m*8*65*32,v.shape) for m in range(2)])
    for member in range(2):
        for i,b in enumerate(workers[:16]):b.store_slice(Tensor(x.offset+member*256,(1,256)),out.offset+(member*65+64)*256,1,256,i*16,(i+1)*16,'newout')
    for b in workers:b.finish()
    return '\n'.join(lines)+'\n',layout,shared.cursor
