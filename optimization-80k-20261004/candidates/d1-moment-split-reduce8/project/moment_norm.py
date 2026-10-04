"""Algebraic FP32 LayerNorm using moments and a fused affine transform.

This changes the schedule and rounding, not learned parameters or the model.
Numerical acceptance must be checked against every public stress fixture.
"""
from .compiler import hbm, imm

def norm(self, source, gamma, beta, out, d):
    r = lambda count, off=0: self._rf(15, count, off)
    self.ld(hbm(source.offset, d), r(d))
    self.vec('mul', [r(d), r(d)], r(d, 256))
    self.reduce('sum', r(d), r(1, 700))
    self.reduce('sum', r(d, 256), r(1, 701))
    self.vec('mul', [r(1, 700), imm(1/d)], r(1, 700))
    self.vec('mul', [r(1, 701), imm(1/d)], r(1, 701))
    self.vec('mul', [r(1, 700), r(1, 700)], r(1, 702))
    self.vec('sub', [r(1, 701), r(1, 702)], r(1, 701))
    self.vec('max', [r(1, 701), imm(0)], r(1, 701))
    self.vec('add', [r(1, 701), imm(1e-5)], r(1, 701))
    self.sfu('rsqrt', r(1, 701), r(1, 701))
    assert gamma.space == beta.space == 'RF'
    self.vec('mul', [self._rf(gamma.lane, d, gamma.offset), r(1, 701)], r(d, 256))
    self.vec('sub', [imm(0), r(1, 700)], r(1, 700))
    self.vec('fma', [r(d, 256), r(1, 700), self._rf(beta.lane, d, beta.offset)], r(d, 384))
    self.vec('fma', [r(d), r(d, 256), r(d, 384)], r(d, 128))
    if out.space != 'RF':
        self.st(r(d, 128), hbm(out.offset, d))
