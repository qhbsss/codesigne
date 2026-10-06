"""Prefetch into distinct layer-private RF after actual projection MAC events."""
import json

def load(lines,w,t,events,w1_only=False):
    tensors=t['w1parts']+([] if w1_only else [b for pair in t['w2parts'] for b in pair])
    commands=[]
    for b in tensors:commands.extend(w.pending_weights.pop(b.rf_chunks,[]))
    if commands:
        lines.append('BARRIER '+json.dumps(dict(wgs=[w.wg],events=events),separators=(',',':')))
        lines.extend(commands)
