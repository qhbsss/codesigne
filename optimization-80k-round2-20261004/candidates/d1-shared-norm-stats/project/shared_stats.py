"""A dedicated workgroup produces mean/inverse stdev for a published input."""
import json
from .compiler import hbm,imm

def produce(w,x,k,shared):
    out=shared.alloc(1,2);r=lambda n,o=0:w._rf(15,n,o)
    w.ld(hbm(x.offset,k),r(k))
    w.reduce('sum',r(k),r(1,700));w.vec('mul',[r(k),r(k)],r(k,256));w.reduce('sum',r(k,256),r(1,701))
    w.vec('mul',[r(1,700),imm(1/k)],r(1,700));w.vec('mul',[r(1,700),r(1,700)],r(1,702))
    w.vec('sub',[imm(1e-5),r(1,702)],r(1,702));w.vec('fma',[r(1,701),imm(1/k),r(1,702)],r(1,701))
    w.vec('max',[r(1,701),imm(1e-5)],r(1,701));w.sfu('rsqrt',r(1,701),r(1,701))
    w.st(r(2,700),hbm(out.offset,2));return out.offset,json.loads(w.lines[-1][3:])['event']
