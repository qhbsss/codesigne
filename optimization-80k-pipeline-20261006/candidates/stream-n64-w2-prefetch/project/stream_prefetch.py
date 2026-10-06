"""Two W2 banks after each completed streamed W1 K phase."""
import json
from .compiler import hbm,add,mul,sub

def fill_stream_w2(lines,owners,t,ident,events):
    phase={'var':ident}
    lines.append('FOR '+json.dumps(dict(var=ident,start=0,stop=4,step=1),separators=(',',':')))
    lines.append('BARRIER '+json.dumps(dict(wgs=[w.wg for w in owners],events=events),separators=(',',':')))
    for w in owners:w.loops.append(ident)
    for extra in range(2):
        bank=add(mul(phase,2),extra)
        panel=sub({'ceildiv':[add(bank,1),4]},1)
        kk=mul({'mod':[bank,4]},64)
        for i,w in enumerate(owners):
            kg,ng=divmod(i,8)
            w.ld(hbm(add(t.offset,kg*256*256,mul(kk,256),ng*32,mul(panel,16)),1024,[64,16],[256,1]),w._rf(bank,1024))
    for w in owners:w.loops.pop()
    lines.append('END.FOR {}')
    return {i:(32,[(p,p*4+k,0,k*64,64) for p in range(2) for k in range(4)]) for i in range(32)}
