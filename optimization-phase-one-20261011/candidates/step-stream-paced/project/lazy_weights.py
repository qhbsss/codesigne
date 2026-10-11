"""Issue each persistent weight slice at its first actual use."""
from .compiler import Builder

def install():
    cache=Builder.cache_weight_rf
    def delayed(w,*args,**kwargs):
        start=len(w.lines);t=cache(w,*args,**kwargs)
        saved=w.lines[start:];del w.lines[start:]
        if not hasattr(w,'pending_weights'):w.pending_weights={}
        w.pending_weights[t.rf_chunks]=saved
        return t
    Builder.cache_weight_rf=delayed
    gemm=Builder.gemm
    def execute(w,a,b,*args,**kwargs):
        if b.rf_chunks and hasattr(w,'pending_weights'):
            w.lines.extend(w.pending_weights.pop(b.rf_chunks,[]))
        return gemm(w,a,b,*args,**kwargs)
    Builder.gemm=execute
