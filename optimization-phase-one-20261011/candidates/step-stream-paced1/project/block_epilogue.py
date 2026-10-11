"""Vectorize complete GEMM epilogue panels and W2 partial reductions."""
from .compiler import hbm,imm,add,mul

def epilogue(self,out,n_extent,col,row_index,acc_off,n,rows,kind,bias,residual,acc_lane,bias_rf_off=None):
    count=rows*n
    acc=self._rf(acc_lane,count,acc_off)
    if kind in ('bias_gelu','residual_bias'):
        if bias_rf_off is None:
            self.ld(hbm(add(bias.offset,col),n),self._rf(6,n))
            bias_rf_off=0
        with self.loop('epibias',0,rows) as row:
            self.vec('add',[self._rf(6,n,bias_rf_off),imm(0)],self._rf(14,n,mul(row,n)))
    if kind=='bias_gelu':
        self.vec('add',[acc,self._rf(14,count)],self._rf(7,count))
        self._gelu_rf(7,count)
        result=self._rf(0,count)
    else:
        self.ld(hbm(add(residual.offset,mul(row_index,n_extent),col),count,[rows,n],[n_extent,1]),self._rf(0,count))
        self.vec('add',[acc,self._rf(0,count)],self._rf(7,count))
        result=self._rf(7,count)
        if kind=='residual_bias':
            self.vec('add',[result,self._rf(14,count)],self._rf(0,count))
            result=self._rf(0,count)
    self.st(result,hbm(add(out.offset,mul(row_index,n_extent),col),count,[rows,n],[n_extent,1]))

def reduce_four(self,partial,out,residual,bias,rows,d,n_group,row_lo,row_hi,name):
    n=d//4;col=n_group*n
    self.ld(hbm(add(bias.offset,col),n),self._rf(6,n))
    for row in range(row_lo,row_hi,8):
        nr=min(8,row_hi-row);count=nr*n
        self.ld(hbm(partial.offset+n_group*rows*d+row*d+col,count,[nr,n],[d,1]),self._rf(2,count))
        for k in range(1,4):
            lane=k%2
            self.ld(hbm(partial.offset+(k*4+n_group)*rows*d+row*d+col,count,[nr,n],[d,1]),self._rf(lane,count))
            self.vec('add',[self._rf(2,count),self._rf(lane,count)],self._rf(2,count))
        self.ld(hbm(residual.offset+row*d+col,count,[nr,n],[d,1]),self._rf(5,count))
        self.vec('add',[self._rf(2,count),self._rf(5,count)],self._rf(2,count))
        for r in range(nr):
            self.vec('add',[self._rf(2,n,r*n),self._rf(6,n)],self._rf(7,n,r*n))
        self.st(self._rf(7,count),hbm(out.offset+row*d+col,count,[nr,n],[d,1]))
