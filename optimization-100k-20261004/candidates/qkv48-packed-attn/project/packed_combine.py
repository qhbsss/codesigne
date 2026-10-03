"""One contiguous transfer for packed context/max/sum attention partials."""
from .compiler import hbm,imm

def combine(self,max_base,sum_base,context_base,dest,parts,hd):
    r=lambda count,off=0:self._rf(15,count,off)
    self.ld(hbm(context_base,parts*(hd+2)),r(parts*(hd+2),64))
    maxima=r(parts,64+hd);maxima.update(shape=[parts],strides=[hd+2])
    sums=r(parts,65+hd);sums.update(shape=[parts],strides=[hd+2])
    context=r(parts*hd,64);context.update(shape=[parts,hd],strides=[hd+2,1])
    self.reduce('max',maxima,r(1,650))
    self.vec('sub',[maxima,r(1,650)],r(parts,32));self.sfu('exp',r(parts,32),r(parts,32))
    self.vec('mul',[sums,r(parts,32)],r(parts,48));self.reduce('sum',r(parts,48),r(1,651))
    self.vec('add',[imm(0),imm(0)],r(hd,700))
    self.emit('MMA.ACC',a=r(parts,32),b=context,acc=r(hd,700),m=1,n=hd,k=parts,event=None)
    self.vec('div',[r(hd,700),r(1,651)],r(hd,700));self.st(r(hd,700),hbm(dest,hd))
