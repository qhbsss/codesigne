"""Layer-private N32 FFN groups, reuse normalization across two N16 panels."""
import json
from .compiler import Tensor,hbm,imm,emit_barrier

def emit(lines,core,workers,shared,res,weights,y,d,f,name):
    assert len(workers)==16 and d==128 and f==512
    # The core fence immediately before this call completes all WO stores.
    # Explicit dependencies also release the separate FFN workgroups.
    stores=[]
    for line in reversed(lines):
        if line.startswith('ST '):
            args=json.loads(line[3:]);dst=args['dst']
            if dst['space']=='HBM' and isinstance(dst['offset'],int) and res.offset<=dst['offset']<res.offset+d:stores.append(args['event'])
        if len(stores)==8:break
    assert len(stores)==8
    lines.append('BARRIER '+json.dumps(dict(wgs=[w.wg for w in workers],events=stores),separators=(',',':')))
    partial=shared.alloc(16,d);first_done=[]
    for i,w in enumerate(workers):
        t=weights[i];r=lambda count,off=0:w._rf(15,count,off)
        w.ld(hbm(res.offset+i*8,8),r(8,928))
        w.vec('add',[r(8,928),imm(0)],w._rf(14,8,520))
        if not getattr(w,'b2_loaded',False):
            w.ld(hbm(t['b2'].offset+i*8,8),w._rf(14,8,512));w.b2_loaded=True
        flat=[b for pair in t['w2parts'] for b in pair]
        for part in range(2):
            w.reuse_norm=(part!=0)
            w.ffn_prefetch=flat[:2] if part==0 else []
            w.ffn_wait_events=list(first_done) if part==0 and i>=8 and any(b.rf_chunks in w.pending_weights for b in flat) else []
            w.affine_gemm(res,t['gamma'],t['betas'][part],t['w1parts'][part],Tensor(96,(1,f),space='RF',lane=15),1,d,f,name+'f'+str(part),n_lo=i*32+part*16,n_hi=i*32+(part+1)*16,epilogue='bias_gelu',bias=t['b1'])
            if part==0:w.vec('add',[r(16,96),imm(0)],r(16,960))
        w.reuse_norm=False
        w.vec('add',[r(16,96),imm(0)],r(16,112));w.vec('add',[r(16,960),imm(0)],r(16,96))
        for part in range(2):w.vec('add',[imm(0),imm(0)],r(d,768 if part==0 else 384))
        for part in range(2):
            for half,b in enumerate(t['w2parts'][part]):
                w.lines.extend(w.pending_weights.pop(b.rf_chunks,[]))
                (bank,off,_,depth),=b.rf_chunks;assert depth==16
                w.emit('MMA.ACC',a=r(16,96+part*16),b=w._rf(bank,1024,off),acc=r(64,(768 if part==0 else 384)+half*64),m=1,n=64,k=16,event=None)
        w.vec('add',[r(d,768),r(d,384)],r(d,768))
        w.st(r(d,768),hbm(partial.offset+i*d,d))
        if i<8:first_done.append(json.loads(lines[-1][3:])['event'])
    emit_barrier(lines,workers)
    for i,w in enumerate(workers):
        r=lambda count,off=0:w._rf(15,count,off);col=i*8
        w.ld(hbm(partial.offset+col,128,[16,8],[d,1]),r(128,64))
        w.vec('add',[imm(1),imm(0)],r(16))
        w.vec('add',[imm(0),imm(0)],r(8,512))
        w.emit('MMA.ACC',a=r(16),b=r(128,64),acc=r(8,512),m=1,n=8,k=16,event=None)
        w.vec('add',[r(8,512),w._rf(14,8,512)],r(8,48))
        w.vec('add',[r(8,48),w._rf(14,8,520)],r(8,48))
        w.st(r(8,48),hbm(y.offset+col,8))
    emit_barrier(lines,core+workers)
