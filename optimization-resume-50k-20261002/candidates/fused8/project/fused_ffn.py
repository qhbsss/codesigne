"""RF-resident W1/GELU/W2 fusion; no activated HBM tensor."""
from math import sqrt, pi
from .compiler import hbm, imm

def view(w, lane, rows, cols, offset=0, stride=None):
    v=w._rf(lane, rows*cols, offset)
    if stride is not None:
        v.update(shape=[rows,cols],strides=[stride,1])
    return v

def fused(w, x, w1, b1, w2, partial, rows, d, f, k_lo, width, name):
    assert rows==64 and width==128
    # W1: retain four 32x64 accumulator panels in lanes 2..5.
    for lane in (2,3,4,5):
        w.vec('add',[imm(0),imm(0)],w._rf(lane,2048))
    for k in range(0,d,32):
        for part in range(2):
            w.ld(hbm(w1.offset+k*f+k_lo+part*64,2048,[32,64],[f,1]),w._rf((1,6)[part],2048))
        for panel in range(2):
            w.ld(hbm(x.offset+panel*32*d+k,1024,[32,32],[d,1]),w._rf(0,1024))
            for part in range(2):
                w.emit('MMA.ACC',a=w._rf(0,1024),b=w._rf((1,6)[part],2048),acc=w._rf(2+part*2+panel,2048),m=32,n=64,k=32,event=None)
    # Broadcast bias, then apply GELU without spilling activation panels.
    for part in range(2):
        w.ld(hbm(b1.offset+k_lo+part*64,64),w._rf(6,64))
        for panel in range(2):
            lane=2+part*2+panel
            for row in range(32):
                w.vec("add",[w._rf(6,64),imm(0)],w._rf(7,64,row*64))
            xrf=w._rf(lane,2048)
            w.vec("add",[xrf,w._rf(7,2048)],xrf)
            w.vec('mul',[xrf,xrf],w._rf(1,2048))
            w.vec('mul',[w._rf(1,2048),xrf],w._rf(1,2048))
            w.vec('fma',[w._rf(1,2048),imm(0.044715),xrf],w._rf(1,2048))
            w.vec('mul',[w._rf(1,2048),imm(sqrt(2/pi))],w._rf(1,2048))
            w.sfu('tanh',w._rf(1,2048),w._rf(1,2048))
            w.vec('add',[w._rf(1,2048),imm(1)],w._rf(1,2048))
            w.vec('mul',[xrf,imm(.5)],w._rf(0,2048))
            w.vec('mul',[w._rf(0,2048),w._rf(1,2048)],xrf)
    # W2: source is a strided RF view, each task owns 128 K rows.
    for col in range(0,d,64):
        for lane in (6,7):
            w.vec('add',[imm(0),imm(0)],w._rf(lane,2048))
        for part in range(2):
            for inner in (0,32):
                w.ld(hbm(w2.offset+(k_lo+part*64+inner)*d+col,2048,[32,64],[d,1]),w._rf(0,2048))
                for panel in range(2):
                    w.emit('MMA.ACC',a=view(w,2+part*2+panel,32,32,inner,64),b=w._rf(0,2048),acc=w._rf(6+panel,2048),m=32,n=64,k=32,event=None)
        for panel in range(2):
            w.st(w._rf(6+panel,2048),hbm(partial.offset+panel*32*d+col,2048,[32,64],[d,1]))

def reduce(w, partial, out, residual, bias, rows, d, tasks, row_lo,row_hi,name):
    # Block reduction keeps row-major partial sums and follows increasing K order.
    for row in range(row_lo,row_hi,8):
        nr=min(8,row_hi-row)
        for col in range(0,d,64):
            n=nr*64
            w.ld(hbm(partial.offset+row*d+col,n,[nr,64],[d,1]),w._rf(2,n))
            for task in range(1,tasks):
                lane=task%2
                w.ld(hbm(partial.offset+task*rows*d+row*d+col,n,[nr,64],[d,1]),w._rf(lane,n))
                w.vec('add',[w._rf(2,n),w._rf(lane,n)],w._rf(2,n))
            w.ld(hbm(residual.offset+row*d+col,n,[nr,64],[d,1]),w._rf(5,n))
            w.vec('add',[w._rf(2,n),w._rf(5,n)],w._rf(2,n))
            w.ld(hbm(bias.offset+col,64),w._rf(6,64))
            for rr in range(nr):
                w.vec("add",[w._rf(2,64,rr*64),w._rf(6,64)],w._rf(7,64,rr*64))
            w.st(w._rf(7,n),hbm(out.offset+row*d+col,n,[nr,64],[d,1]))
