"""Start one W2 bank per W1 tile after its real MAC events, during GELU."""
import json,re
from .compiler import hbm,add,mul,sub

def floor(x,n):return sub({'ceildiv':[add(x,1),n]},1)

def fill_w2(lines,owners,t,compute_lines):
    # All event definitions occur earlier in source. Reuse the same FOR var
    # so event templates stay identical through deterministic compaction.
    refs={}
    for line in compute_lines:
        if line.startswith('MMA.ACC '):
            d=json.loads(line[8:])
            if d['k']==64:refs[d['acc']['wg']]=d['event']
    assert len(refs)==32
    variables={tuple(re.findall(r'\{([^}]+)\}',event)) for event in refs.values()}
    assert len(variables)==1
    (ident,)=variables.pop();bank={'var':ident}
    part=floor(bank,4);kk=mul({'mod':[bank,4]},64)
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=8,step=1),separators=(',',':')))
    lines.append('BARRIER '+json.dumps(dict(wgs=[w.wg for w in owners],events=list(refs.values())),separators=(',',':')))
    for w in owners:w.loops.append(ident)
    for i,w in enumerate(owners):
        kg,ng=divmod(i,8)
        w.ld(hbm(add(t.offset,kg*256*256,mul(kk,256),ng*32,mul(part,16)),1024,[64,16],[256,1]),w._rf(bank,1024))
    for w in owners:w.loops.pop()
    lines.append('END.FOR {}')
    return {i:(32,[(p,p*4+k,0,k*64,64) for p in range(2) for k in range(4)]) for i in range(32)}
