"""W2 rows depend on actual W1/GELU store events, never future inputs."""
import json,re
from .compiler import hbm,imm,add,mul,emit_barrier

def consume_w2(lines,owners,act,part,rows,weights,producer):
    refs={}
    for line in producer:
        if line.startswith('ST '):
            d=json.loads(line[3:]);refs[d['src']['wg']]=d['event']
    assert len(refs)==32
    variables={tuple(re.findall(r'\{([^}]+)\}',v)) for v in refs.values()}
    assert len(variables)==1
    (ident,)=variables.pop();tile={'var':ident};row=mul(tile,16)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=rows//16,step=1),separators=(',',':')))
    for w in owners:w.loops.append(ident)
    lines.append('BARRIER '+json.dumps(dict(wgs=[w.wg for w in owners],events=list(refs.values())),separators=(',',':')))
    for w in owners:w.vec('add',[imm(0),imm(0)],w._rf(14,512))
    for ki in range(4):
        for i,w in enumerate(owners):
            w.ld(hbm(add(act.offset,(i//8)*256,mul(row,1024),ki*64),1024,[16,64],[1024,1]),w._rf(8+ki,1024))
        for i,w in enumerate(owners):
            for panel in range(2):
                _,lane,off,_,dk=next(c for c in weights[i]['w2'][1] if c[0]==panel and c[3]==ki*64)
                acc=w._rf(14,256,panel*16);acc.update(shape=[16,16],strides=[32,1])
                w.emit('MMA.ACC',a=w._rf(8+ki,1024),b=w._rf(lane,1024,off),acc=acc,m=16,n=16,k=64,event=None)
    for i,w in enumerate(owners):
        w.st(w._rf(14,512),hbm(add(part.offset,i*(rows*32+64),mul(row,32)),512,[16,32],[32,1]))
    emit_barrier(lines,owners)
    for w in owners:w.loops.pop()
    lines.append('END.FOR {}')
