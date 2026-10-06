"""N64 stream: eight RF accumulator banks, four weight banks, one A bank."""
import json
from math import sqrt,pi
from .compiler import hbm,imm,add,mul,emit_barrier

def produce_w1(lines,workers,a,weight,bias,out,rows,name):
    assert rows==128 and len(workers)==16
    for i,w in enumerate(workers):
        for bank in range(8):w.vec('add',[imm(0),imm(0)],w._rf(bank,1024))
        w.ld(hbm(bias.offset+i*64,64),w._rf(14,64,512))
    phase_id=name+'k';tile_id=name+'row';phase={'var':phase_id};tile={'var':tile_id}
    lines.append('FOR '+json.dumps(dict(var=phase_id,start=0,stop=4,step=1),separators=(',',':')))
    for w in workers:w.loops.append(phase_id)
    for i,w in enumerate(workers):
        for panel in range(4):
            w.ld(hbm(add(weight.offset,i*64,panel*16,mul(phase,64*1024)),1024,[64,16],[1024,1]),w._rf(8+panel,1024))
    emit_barrier(lines,workers)
    lines.append('FOR '+json.dumps(dict(var=tile_id,start=0,stop=8,step=1),separators=(',',':')))
    for w in workers:w.loops.append(tile_id)
    for w in workers:w.ld(hbm(add(a.offset,mul(tile,16*256),mul(phase,64)),1024,[16,64],[256,1]),w._rf(12,1024))
    for w in workers:
        for panel in range(4):
            # floor(tile/4) expressed with the ISA's ceildiv primitive.
            from .compiler import sub
            bank=add(panel*2,sub({'ceildiv':[add(tile,1),4]},1))
            off=mul({'mod':[tile,4]},256)
            w.emit('MMA.ACC',a=w._rf(12,1024),b=w._rf(8+panel,1024),acc=w._rf(bank,256,off),m=16,n=16,k=64,event=None)
    emit_barrier(lines,workers)
    for w in workers:w.loops.pop()
    lines.append('END.FOR {}')
    for w in workers:w.loops.pop()
    lines.append('END.FOR {}')
    for i,w in enumerate(workers):
        for panel in range(4):
            for half in range(2):
                acc=w._rf(panel*2+half,1024);t=lambda n,o=0:w._rf(15,n,o)
                w.vec('add',[imm(1),imm(0)],w._rf(12,64))
                w.emit('MMA.ACC',a=w._rf(12,64),b=w._rf(14,16,512+panel*16),acc=acc,m=64,n=16,k=1,event=None)
                w.vec('mul',[acc,acc],t(1024));w.vec('fma',[t(1024),imm(.044715*sqrt(2/pi)),imm(sqrt(2/pi))],t(1024))
                w.vec('mul',[t(1024),acc],t(1024));w.sfu('tanh',t(1024),t(1024))
                w.vec('mul',[acc,imm(.5)],w._rf(12,1024));w.vec('fma',[w._rf(12,1024),t(1024),w._rf(12,1024)],acc)
                w.st(acc,hbm(out.offset+i*64+panel*16+half*64*1024,1024,[64,16],[1024,1]))
