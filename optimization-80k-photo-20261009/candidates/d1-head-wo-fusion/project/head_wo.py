"""Stable attention-head merge followed by split-K WO projection."""
import json
from .compiler import hbm,imm

def emit_head_wo(lines,workers,shared,x,weights,meta,partial,res,events,tasks,parts,hd,d):
    outputs=shared.alloc(4,d);producer=[];width=d//2
    for index,task in tasks.items():
        w=workers[index];head,half=divmod(task,2)
        lines.append('WAIT '+json.dumps(dict(wg=w.wg,events=events[head]),separators=(',',':')))
        w.combine_online_attention_persistent(meta.offset+2*head*parts,meta.offset+2*head*parts+1,partial.offset+head*parts*hd,None,parts,hd)
        b=weights[index]['wo'];lines.extend(w.pending_weights.pop(b.rf_chunks,[]));acc=w._rf(15,width,768)
        w.vec('add',[imm(0),imm(0)],acc)
        for lane,offset,kstart,depth in b.rf_chunks:
            w.emit('MMA.ACC',a=w._rf(15,depth,700+kstart),b=w._rf(lane,depth*width,offset),acc=acc,m=1,n=width,k=depth,event=None)
        start=len(lines);w.st(acc,hbm(outputs.offset+head*d+half*width,width))
        producer.extend(json.loads(l[3:])['event'] for l in lines[start:] if l.startswith('ST '))
    for half,index in enumerate(list(tasks)[:2]):
        w=workers[index];lines.append('WAIT '+json.dumps(dict(wg=w.wg,events=producer),separators=(',',':')))
        acc=w._rf(15,width,768);tmp=w._rf(15,width,832)
        w.ld(hbm(outputs.offset+half*width,width),acc)
        for head in range(1,4):
            w.ld(hbm(outputs.offset+head*d+half*width,width),tmp);w.vec('add',[acc,tmp],acc)
        w.ld(hbm(x.offset+half*width,width),tmp);w.vec('add',[acc,tmp],acc);w.st(acc,hbm(res.offset+half*width,width))
