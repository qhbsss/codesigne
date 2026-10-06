"""One tile softmax merge for split decode attention."""
from .compiler import hbm,imm

def combine(self,max_base,sum_base,context_base,dest,parts,hd):
    lane=getattr(self,'transient_lane',7)
    r=lambda count,off=0:self._rf(lane,count,off)
    assert sum_base==max_base+1
    self.ld(hbm(max_base,parts*2),r(parts*2,0))
    mx=r(parts,0);mx.update(shape=[parts],strides=[2])
    sm=r(parts,1);sm.update(shape=[parts],strides=[2])
    self.ld(hbm(context_base,parts*hd),r(parts*hd,64))
    self.reduce('max',mx,r(1,650))
    self.vec('sub',[mx,r(1,650)],r(parts,32))
    self.sfu('exp',r(parts,32),r(parts,32))
    self.vec('mul',[sm,r(parts,32)],r(parts,48))
    self.reduce('sum',r(parts,48),r(1,651))
    self.vec('add',[imm(0),imm(0)],r(hd,700))
    self.emit('MMA.ACC',a=r(parts,32),b=r(parts*hd,64),acc=r(hd,700),m=1,n=hd,k=parts,event=None)
    self.vec('div',[r(hd,700),r(1,651)],r(hd,700))
    self.st(r(hd,700),hbm(dest,hd))
