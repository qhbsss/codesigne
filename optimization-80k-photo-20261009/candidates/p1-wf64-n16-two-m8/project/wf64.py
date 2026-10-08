import json
from math import sqrt,pi
from .compiler import hbm,imm,add,mul,sub,emit_barrier
from .wide_qkv import dense as original

def dense(lines,allworkers,tasks,rows,k,n,name,**kwargs):
    if not kwargs.get('gelu',False):return original(lines,allworkers,tasks,rows,k,n,name,**kwargs)
    assert rows==128 and k==256 and len(tasks)==64
    for w,a,b,out,col in tasks:w.ld(hbm(a.offset,1024,[16,64],[256,1]),w._rf(12,1024))
    emit_barrier(lines,allworkers)
    ident=name+'tile';tile={'var':ident};row=mul(tile,16)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=8,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,256))
    for ki in range(4):
        refs=[]
        for w,a,b,out,col in tasks:
            lane,off,kk,depth=next(ch for ch in b[1] if ch[2]==ki*64)
            for ri in range(2):
                w.emit('MMA.ACC',a=w._rf(12+ki%2,512,ri*512),b=w._rf(lane,1024,off),acc=w._rf(14,128,ri*128),m=8,n=16,k=64,event=None)
                refs.append(json.loads(lines[-1][8:])['event'])
        following=tile if ki<3 else {'min':[add(tile,1),7]}
        for w,a,b,out,col in tasks:
            w.ld(hbm(add(a.offset,mul(following,16*256),((ki+1)%4)*64),1024,[16,64],[256,1]),w._rf(12+(ki+1)%2,1024))
        if ki in (0,2):
            for w,a,b,out,col in tasks:
                if hasattr(w,'w2_prefetch'):
                    t,i=w.w2_prefetch
                    if i//16!=ki//2:continue
                    w.emit('WAIT',wg=w.wg,events=refs)
                    kg,ng=divmod(i,8);panel=sub({'ceildiv':[add(tile,1),4]},1);kk=mul({'mod':[tile,4]},64)
                    w.ld(hbm(add(t.offset,kg*256*256,mul(kk,256),ng*32,mul(panel,16)),1024,[64,16],[256,1]),w._rf(add(4,tile),1024))
    for w,a,b,out,col in tasks:
        acc=w._rf(14,256);temp=w._rf(15,256);half=w._rf(15,256,256)
        w.vec('add',[imm(1),imm(0)],w._rf(15,16))
        w.emit('MMA.ACC',a=w._rf(15,16),b=w._rf(14,16,512),acc=acc,m=16,n=16,k=1,event=None)
        w.vec('mul',[acc,acc],temp);w.vec('fma',[temp,imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],temp)
        w.vec('mul',[temp,acc],temp);w.sfu('tanh',temp,temp);w.vec('mul',[acc,imm(.5)],half);w.vec('fma',[half,temp,half],acc)
        w.st(acc,hbm(add(out.offset,mul(row,1024),col),256,[16,16],[1024,1]))
    emit_barrier(lines,allworkers)
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')

def fill_w2(lines,owners,t,compute_lines):
    return {i:(32,[(p,4+p*4+kk,0,kk*64,64) for p in range(2) for kk in range(4)]) for i in range(32)}
