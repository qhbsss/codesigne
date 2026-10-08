"""Two W2 row accumulators; align on MAC events without fencing prior stores."""
import json
from .compiler import hbm,imm,add,mul,emit_barrier

def dense(lines,allworkers,tasks,rows,k,n,name,y,res):
    assert len(tasks)==32 and rows==128 and k==256 and n==32
    for w,a,b,out,col in tasks:
        for ki in range(4):w.ld(hbm(a.offset+ki*64,1024,[16,64],[a.row_stride or k,1]),w._rf(8+ki,1024))
    emit_barrier(lines,allworkers)
    ident=name+'tile';tile={'var':ident};base=mul({'mod':[tile,2]},512);row=mul(tile,16)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=8,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,512,base))
    def load(ki):
        following=add(8,{'mod':[add(mul(add(tile,1),4),ki//64),6]});nextrow=mul({'min':[add(tile,1),7]},16)
        for w,a,b,out,col in tasks:w.ld(hbm(add(a.offset,mul(nextrow,a.row_stride or k),ki),1024,[16,64],[a.row_stride or k,1]),w._rf(following,1024))
    refs=[]
    for ki in range(0,256,64):
        if ki:load(ki)
        current=add(8,{'mod':[add(mul(tile,4),ki//64),6]})
        for w,a,b,out,col in tasks:
            for panel in range(2):
                _,bank,off,_,depth=next(c for c in b[1] if c[0]==panel and c[3]==ki)
                acc=w._rf(14,256,add(base,panel*16));acc.update(shape=[16,16],strides=[32,1])
                w.emit('MMA.ACC',a=w._rf(current,1024),b=w._rf(bank,1024,off),acc=acc,m=16,n=16,k=64,event=None)
            if ki==192:refs.append(json.loads(lines[-1][8:])['event'])
        if not ki:load(ki)
    stores={}
    for i,(w,a,b,out,col) in enumerate(tasks):
        if i>=8:
            w.st(w._rf(14,512,base),hbm(add(out.offset,mul(row,out.row_stride or n),col),512,[16,32],[out.row_stride or n,1]))
            stores[i]=json.loads(lines[-1][3:])['event']
    for i,(w,a,b,out,col) in enumerate(tasks[:8]):
        w.emit('WAIT',wg=w.wg,events=[stores[i+kg*8] for kg in range(1,4)])
        acc=w._rf(14,512,base);temp=w._rf(15,512)
        for kg in range(1,4):
            other=tasks[i+kg*8][3]
            w.ld(hbm(add(other.offset,mul(row,32)),512),temp)
            w.vec('add',[acc,temp],acc)
        w.ld(hbm(add(res.offset,mul(row,256),i*32),512,[16,32],[256,1]),temp)
        w.vec('add',[acc,temp],acc)
        w.vec('add',[imm(1),imm(0)],w._rf(15,16,800))
        w.emit('MMA.ACC',a=w._rf(15,16,800),b=w._rf(15,32,768),acc=acc,m=16,n=32,k=1,event=None)
        w.st(acc,hbm(add(y.offset,mul(row,256),i*32),512,[16,32],[256,1]))
    for w,a,b,out,col in tasks:w.emit('WAIT',wg=w.wg,events=refs)
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
