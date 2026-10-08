"""Two W2 row accumulators; align on MAC events without fencing prior stores."""
import json
from .compiler import hbm,imm,add,mul,emit_barrier

def dense(lines,allworkers,tasks,rows,k,n,name):
    assert len(tasks)==32 and rows==128 and k==256 and n==32
    for w,a,b,out,col in tasks:
        for ki in range(4):w.ld(hbm(a.offset+ki*64,1024,[16,64],[a.row_stride or k,1]),w._rf(ki,1024))
    emit_barrier(lines,allworkers)
    ident=name+'tile';tile={'var':ident};base=mul({'mod':[tile,2]},512);row=mul(tile,16)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=8,step=1),separators=(',',':')))
    for w,a,b,out,col in tasks:w.loops.append(ident)
    for w,a,b,out,col in tasks:w.vec('add',[imm(0),imm(0)],w._rf(14,512,base))
    def load(ki):
        following=ki//64;nextrow=mul({'min':[add(tile,1),7]},16)
        for w,a,b,out,col in tasks:w.ld(hbm(add(a.offset,mul(nextrow,a.row_stride or k),ki),1024,[16,64],[a.row_stride or k,1]),w._rf(following,1024))
    refs=[]
    for ki in range(0,256,64):
        current=ki//64
        for w,a,b,out,col in tasks:
            for panel in range(2):
                _,bank,off,_,depth=next(c for c in b[1] if c[0]==panel and c[3]==ki)
                acc=w._rf(14,256,add(base,panel*16));acc.update(shape=[16,16],strides=[32,1])
                w.emit('MMA.ACC',a=w._rf(current,1024),b=w._rf(bank,1024,off),acc=acc,m=16,n=16,k=64,event=None)
            if ki==192:refs.append(json.loads(lines[-1][8:])['event'])
        load(ki)
    for w,a,b,out,col in tasks:w.st(w._rf(14,512,base),hbm(add(out.offset,mul(row,out.row_stride or n),col),512,[16,32],[out.row_stride or n,1]))
    for w,a,b,out,col in tasks:w.emit('WAIT',wg=w.wg,events=refs)
    for w,a,b,out,col in tasks:w.loops.pop()
    lines.append('END.FOR {}')
