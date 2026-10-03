"""Current-layer RF weights and a six-bank row-prefetch pipeline."""
from math import sqrt,pi
from .sh_wide_dense import dense
from .compiler import Builder,Tensor,hbm,imm,add,mul,minimum,sub,columns,emit_barrier
from .schedule import _open_workers,_weights,_commit
from codesign.challenge.workload import MODELS
from codesign.challenge.abi import build_layout

class RFBuilder(Builder):
    def cache_wide(self,t,lo,hi):
        assert hi-lo==64 and t.shape[0]==256
        self.ld(hbm(t.offset+lo,16384,[256,64],[t.shape[1],1]),self._sh(1024,16384))
        return (64,1024)
    def reduce_wide(self,part,y,res,rows,col,row_lo,row_hi,name):
        self.vec('add',[imm(1),imm(0)],self._rf(11,12))
        with self.loop(name,row_lo,row_hi,4) as row:
            ng=col//64;acc=self._rf(14,256)
            self.ld(hbm(add(part.offset,ng*rows*64,mul(row,64)),1024,[4,256],[4*rows*64,1]),self._rf(15,1024))
            self.vec('add',[imm(0),imm(0)],acc)
            self.emit('MMA.ACC',a=self._rf(11,4,8),b=self._rf(15,1024),acc=acc,m=1,n=256,k=4,event=None)
            self.ld(hbm(add(res.offset,mul(row,256),col),256,[4,64],[256,1]),self._rf(15,256))
            self.vec('add',[acc,self._rf(15,256)],acc)
            self.ld(self._sh(0,64),self._rf(15,64,512))
            self.emit('MMA.ACC',a=self._rf(11,4),b=self._rf(15,64,512),acc=acc,m=4,n=64,k=1,event=None)
            self.st(acc,hbm(add(y.offset,mul(row,256),col),256,[4,64],[256,1]))
    def reset_cache(self):self.used=[0]*13
    def cache(self,t,lo,hi,depth):
        chunks=[];width=hi-lo
        if width in (32,48):
            for part in range(width//16):
                for k in range(0,256,64):
                    lane=part*4+k//64
                    self.ld(hbm(t.offset+k*t.shape[1]+lo+part*16,1024,[64,16],[t.shape[1],1]),self._rf(lane,1024))
                    chunks.append((part,lane,0,k,64))
            return (width,chunks)
        base=4 if width==8 else 0
        for k in range(0,t.shape[0],depth):
            dk=min(depth,t.shape[0]-k)
            if t.shape[1]==768:lane=base+k//64;off=0
            else:lane=base+(k*width)//1024;off=(k*width)%1024
            chunks.append((lane,off,k,dk))
        groups=[]
        for item in chunks:
            if groups and item[0]==groups[-1][-1][0] and item[1]==groups[-1][-1][1]+groups[-1][-1][3]*width and item[2]==groups[-1][-1][2]+groups[-1][-1][3] and item[3]==groups[-1][-1][3]:groups[-1].append(item)
            else:groups.append([item])
        self.cache_id=getattr(self,'cache_id',0)+1
        for gi,g in enumerate(groups):
            lane,off,k,dk=g[0]
            if len(g)>1:
                with self.loop('cache'+str(self.cache_id)+'g'+str(gi),0,len(g)) as idx:
                    self.ld(hbm(add(t.offset,lo,mul(add(k,mul(idx,dk)),t.shape[1])),dk*width,[dk,width],[t.shape[1],1]),self._rf(lane,dk*width,add(off,mul(idx,dk*width))))
            else:self.ld(hbm(t.offset+k*t.shape[1]+lo,dk*width,[dk,width],[t.shape[1],1]),self._rf(lane,dk*width,off))
        return (width,chunks)
    def cache_vector(self,t,lo,n,off):
        if off<1024:self.ld(hbm(t.offset+lo,n),self._sh(off,n))
        else:self.ld(hbm(t.offset+lo,n),self._rf(14,n,off-1024))
    def norm(self,x,y,d,gamma,beta,name,lo,hi):
        r=lambda n,o=0:self._rf(15,n,o)
        self.ld(self._sh(gamma,d),self._rf(14,d,0));self.ld(self._sh(beta,d),self._rf(14,d,256))
        with self.loop(name,lo,hi) as row:
            self.ld(hbm(add(x.offset,mul(row,d)),d),r(d))
            self.reduce('sum',r(d),r(1,768));self.vec('mul',[r(1,768),imm(1/d)],r(1,768))
            self.vec('sub',[r(d),r(1,768)],r(d,256));self.vec('mul',[r(d,256),r(d,256)],r(d,512))
            self.reduce('sum',r(d,512),r(1,768));self.vec('mul',[r(1,768),imm(1/d)],r(1,768))
            self.vec('add',[r(1,768),imm(1e-5)],r(1,768));self.sfu('rsqrt',r(1,768),r(1,768))
            self.vec('mul',[r(d,256),r(1,768)],r(d,256))
            self.vec('mul',[r(d,256),self._rf(14,d,0)],r(d,256));self.vec('add',[r(d,256),self._rf(14,d,256)],r(d,256))
            self.st(r(d,256),hbm(add(y.offset,mul(row,d)),d))
    def gemm_cached(self,a,b,out,rows,k,n,col,name,m_tile=32,epilogue=None,residual=None,bias_off=1536):
        width,chunks=b;stride=a.row_stride or k
        r=lambda count,off=0:self._rf(15,count,off)
        with self.loop(name,0,rows,m_tile) as row:
            nr=minimum(m_tile,sub(rows,row));cells=mul(nr,width);acc=self._rf(14,cells,0)
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
                    self.vec('add',[self._rf(14,width,bias_off-1024),imm(0)],r(width,add(512,mul(rr,width))))
            if epilogue=='gelu':
                self.vec('add',[acc,r(cells,512)],acc)
                self.vec('mul',[acc,acc],r(cells));self.vec('mul',[r(cells),acc],r(cells))
                self.vec('fma',[r(cells),imm(.044715),acc],r(cells));self.vec('mul',[r(cells),imm(sqrt(2/pi))],r(cells))
                self.sfu('tanh',r(cells),r(cells));self.vec('add',[r(cells),imm(1)],r(cells))
                self.vec('mul',[acc,imm(.5)],r(cells,512));self.vec('mul',[r(cells,512),r(cells)],acc)
            elif epilogue in ('res','resbias'):
                self.ld(hbm(add(residual.offset,mul(row,n),col),cells,[nr,width],[n,1]),r(cells))
                self.vec('add',[acc,r(cells)],acc)
                if epilogue=='resbias':self.vec('add',[acc,r(cells,512)],acc)
            self.st(acc,hbm(add(out.offset,mul(row,n),col),cells,[nr,width],[n,1]))
    def export(self,qkv,k,v,head,row_lo,row_hi,name):
        with self.loop(name,row_lo,row_hi,8) as row:
            for comp,t in [(1,k),(2,v)]:
                self.ld(hbm(add(qkv.offset,mul(row,768),comp*256+head*32),256,[8,32],[768,1]),self._rf(15,256))
                self.st(self._rf(15,256),hbm(add(t.offset,head*65*32,mul(row,32)),256))
    def preload_kv(self,k,v,head):
        for part in range(2):
            self.ld(hbm(k.offset+head*65*32+part*1024,1024),self._rf(6+part,1024))
            self.ld(hbm(v.offset+head*65*32+part*1024,1024),self._rf(8+part,1024))
        self.kv_preloaded=True
    def attn(self,qkv,ctx,k,v,head,blo,bhi,name):
        r=lambda n,o=0:self._rf(14,n,o-1024) if isinstance(o,int) and o>=1024 else self._rf(15,n,o)
        if not getattr(self,'kv_preloaded',False):self.preload_kv(k,v,head)
        self.kv_preloaded=False
        def score_view(key):
            x=r(64,add(512,mul(key,8)));x.update(shape=[8,8],strides=[64,1]);return x
        with self.loop(name,blo,bhi) as block:
            row=mul(block,8)
            self.ld(hbm(add(qkv.offset,mul(row,768),head*32),256,[8,32],[768,1]),r(256))
            self.vec('add',[imm(0),imm(0)],r(512,512))
            for part in range(1 if bhi<=4 else 2):
                lo=part*4;hi=minimum(add(block,1),4) if part==0 else add(block,1)
                with self.loop(name+'k'+str(part),lo,hi) as key:
                    kv=self._rf(6+part,256,mul(sub(key,lo),256));kv.update(shape=[32,8],strides=[1,32])
                    self.vec('add',[imm(0),imm(0)],r(64,1024))
                    self.emit('MMA.ACC',a=r(256),b=kv,acc=r(64,1024),m=8,n=8,k=32,event=None)
                    self.vec('mul',[r(64,1024),imm(1/sqrt(32))],r(64,1024));self.vec('add',[r(64,1024),imm(0)],score_view(key))
            with self.loop(name+'s',0,8) as local:
                limit=add(row,local,1);slot=add(512,mul(local,64))
                self.reduce('max',r(limit,slot),r(1,1344));self.vec('sub',[r(limit,slot),r(1,1344)],r(limit,slot))
                self.sfu('exp',r(limit,slot),r(limit,slot));self.reduce('sum',r(limit,slot),r(1,1345));self.vec('div',[r(limit,slot),r(1,1345)],r(limit,slot))
            with self.loop(name+'z',0,7) as local:
                self.vec('add',[imm(0),imm(0)],r(sub(7,local),add(512,mul(local,64),row,local,1)))
            self.vec('add',[imm(0),imm(0)],r(256,1088))
            for part in range(1 if bhi<=4 else 2):
                lo=part*4;hi=minimum(add(block,1),4) if part==0 else add(block,1)
                with self.loop(name+'v'+str(part),lo,hi) as key:
                    self.emit('MMA.ACC',a=score_view(key),b=self._rf(8+part,256,mul(sub(key,lo),256)),acc=r(256,1088),m=8,n=32,k=8,event=None)
            self.st(r(256,1088),hbm(add(ctx.offset,mul(row,256),head*32),256,[8,32],[256,1]))
    def reduce32(self,part,y,res,rows,col,row_lo,row_hi,name):
        self.vec('add',[imm(1),imm(0)],self._rf(11,20))
        with self.loop(name,row_lo,row_hi,8) as row:
            count=256;acc=self._rf(14,count,0);ng=col//32
            self.ld(hbm(add(part.offset,ng*rows*32,mul(row,32)),1024,[4,256],[8*rows*32,1]),self._rf(15,1024))
            self.vec('add',[imm(0),imm(0)],acc)
            self.emit('MMA.ACC',a=self._rf(11,4,16),b=self._rf(15,1024),acc=acc,m=1,n=256,k=4,event=None)
            self.ld(hbm(add(res.offset,mul(row,256),col),count,[8,32],[256,1]),self._rf(15,count))
            self.vec('add',[acc,self._rf(15,count)],acc)
            self.emit('MMA.ACC',a=self._rf(11,8),b=self._rf(14,32,528),acc=acc,m=8,n=32,k=1,event=None)
            self.st(acc,hbm(add(y.offset,mul(row,256),col),count,[8,32],[256,1]))

def generate_m1_p1():
    layout=build_layout(MODELS['M1'],'P1');lines=[];origin=layout.symbols['scratch'].address//4
    workers,shared=_open_workers(layout,lines,'r',origin,[i%16 for i in range(64)],shared_bytes=4096)
    # _open_workers uses Builder; replace it before invoking this function.
    # One 68-KiB weight workspace per SM; other groups retain 4 KiB.
    import json
    for i,b in enumerate(workers):
        b.shared_bytes=69632 if i<16 else 4096
    for i,line in enumerate(lines):
        if line.startswith('WG.BEGIN '):
            arg=json.loads(line[9:]);arg['shared_bytes']=69632 if int(arg['wg'][1:])<16 else 4096
            lines[i]='WG.BEGIN '+json.dumps(arg,separators=(',',':'))
    p=workers[0];x=p.symbol('input/prompt');out=p.symbol('output/hidden');rows=128
    for layer in range(3):
        w=_weights(p,f'layer{layer}/');weights=[{} for _ in workers]
        for i,b in enumerate(workers[32:]):
            for j,key in enumerate(['ln1_g','ln1_b','ln2_g','ln2_b']):b.cache_vector(w[key],0,256,j*256)
        qkv=shared.alloc(rows,768);ctx=shared.alloc(rows,256);res=shared.alloc(rows,256);ln1=shared.alloc(rows,256);ln2=shared.alloc(rows,256);act=shared.alloc(rows,1024);y=shared.alloc(rows,256)
        k=p.symbol(f'layer{layer}/new_k');v=p.symbol(f'layer{layer}/new_v')
        for i,b in enumerate(workers[32:]):b.norm(x,ln1,256,0,256,f'l{layer}n1',i*4,(i+1)*4)
        emit_barrier(lines,workers)
        for j,i in enumerate(range(16)):
            b=workers[i]
            weights[i]['qkv']=b.cache(w['wqkv'],i*48,(i+1)*48,64)
            if j%2==1:emit_barrier(lines,workers)
        emit_barrier(lines,workers)
        dense(lines,workers,[(b,ln1,weights[i]['qkv'],qkv,i*48) for i,b in enumerate(workers[:16])],rows,256,768,f'l{layer}q',n_tile=48,k_tile=64)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[:32]):
            member,local=divmod(i,16);head,half=divmod(local,2)
            b.export(Tensor(qkv.offset+member*64*768,(64,768)),Tensor(k.offset+member*8*65*32,k.shape),Tensor(v.offset+member*8*65*32,v.shape),head,half*32,(half+1)*32,f'l{layer}e')
            if i%8==7:emit_barrier(lines,workers)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[:32]):
            member,local=divmod(i,16);head,half=divmod(local,2)
            b.preload_kv(Tensor(k.offset+member*8*65*32,k.shape),Tensor(v.offset+member*8*65*32,v.shape),head)
            if i%8==7:emit_barrier(lines,workers)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[:32]):
            member,local=divmod(i,16);head,half=divmod(local,2)
            b.attn(Tensor(qkv.offset+member*64*768,(64,768)),Tensor(ctx.offset+member*64*256,(64,256)),Tensor(k.offset+member*8*65*32,k.shape),Tensor(v.offset+member*8*65*32,v.shape),head,half*4,(half+1)*4,f'l{layer}a')
        emit_barrier(lines,workers)
        for j,i in enumerate([off+base for off in range(4) for base in range(0,16,4)]):
            b=workers[i]
            weights[i]['wo']=b.cache(w['wo'],i*16,(i+1)*16,64)
            if j%4==3:emit_barrier(lines,workers)
        emit_barrier(lines,workers)
        dense(lines,workers,[(b,ctx,weights[i]['wo'],res,i*16) for i,b in enumerate(workers[:16])],rows,256,256,f'l{layer}o',n_tile=16,k_tile=64,residual=x)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[32:]):b.norm(res,ln2,256,512,768,f'l{layer}n2',i*4,(i+1)*4)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[:16]):
            weights[i]['w1']=b.cache_wide(w['w1'],i*64,(i+1)*64)
            b.ld(hbm(w['b1'].offset+i*64,64),b._sh(0,64))
            if i%2==1:emit_barrier(lines,workers)
        emit_barrier(lines,workers)
        dense(lines,workers,[(b,ln2,weights[i]['w1'],act,i*64) for i,b in enumerate(workers[:16])],rows,256,1024,f'l{layer}f',n_tile=64,k_tile=64,gelu=True)
        emit_barrier(lines,workers)
        part=shared.alloc(16,rows,64)
        for i,b in enumerate(workers[:16]):
            kg,ng=divmod(i,4);t=Tensor(w['w2'].offset+kg*256*256,(256,256))
            weights[i]['w2']=b.cache_wide(t,ng*64,(ng+1)*64)
            b.ld(hbm(w['b2'].offset+ng*64,64),b._sh(0,64))
            if i%2==1:emit_barrier(lines,workers)
        emit_barrier(lines,workers)
        dense(lines,workers,[(b,Tensor(act.offset+(i//4)*256,(rows,256),row_stride=1024),weights[i]['w2'],Tensor(part.offset+i*rows*64,(rows,64)),0) for i,b in enumerate(workers[:16])],rows,256,64,f'l{layer}w',n_tile=64,k_tile=64)
        emit_barrier(lines,workers)
        for i,b in enumerate(workers[:16]):
            rg,ng=divmod(i,4);b.reduce_wide(part,y,res,rows,ng*64,rg*32,(rg+1)*32,f'l{layer}r')
        emit_barrier(lines,workers);x=y
    for member in range(2):
        for i,b in enumerate(workers[:16]):b.store_slice(Tensor(x.offset+member*64*256,(64,256)),out.offset+member*65*256,64,256,i*16,(i+1)*16,'out')
    _commit(lines,0)
    # Use the validated conservative first-new-position generator.
    from .schedule import emit_p1_fused_step_layer
    x=p.symbol('input/step0');stepworkers=workers[:32]
    # The existing parallel-step accepts32; conservative original accepts8.
    from .step_cached import emit_p1_fused_step_layer as parallel_step
    for layer in range(3):
        k=p.symbol(f'layer{layer}/new_k');v=p.symbol(f'layer{layer}/new_v')
        x=parallel_step(lines,stepworkers,shared,x,_weights(p,f'layer{layer}/'),256,1024,8,32,f's{layer}',64,
                        [Tensor(k.offset+m*8*65*32,k.shape) for m in range(2)],[Tensor(v.offset+m*8*65*32,v.shape) for m in range(2)])
    for member in range(2):
        for i,b in enumerate(workers[:16]):b.store_slice(Tensor(x.offset+member*256,(1,256)),out.offset+(member*65+64)*256,1,256,i*16,(i+1)*16,'newout')
    for b in workers:b.finish()
    return '\n'.join(lines)+'\n',layout,shared.cursor
