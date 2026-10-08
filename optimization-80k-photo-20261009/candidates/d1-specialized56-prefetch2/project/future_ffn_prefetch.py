"""Prefetch next-layer FFN from real QKV writes, two consumers per wave."""
import json

def emit(lines,tasks,refs):
    previous=refs
    for start in range(0,len(tasks),2):
        pair=tasks[start:start+2]
        lines.append('BARRIER '+json.dumps(dict(wgs=[w.wg for w,_ in pair],events=previous),separators=(',',':')))
        perworker=[]
        for w,weights in pair:
            loads=[]
            for b in weights:
                loads.extend(command for command in w.pending_weights.pop(b.rf_chunks,[]) if command.startswith('LD '))
            perworker.append((w,loads))
        assert len(perworker[0][1])==len(perworker[1][1])==8
        for bank in range(8):
            previous=[]
            for w,loads in perworker:
                command=loads[bank];lines.append(command);previous.append(json.loads(command[3:])['event'])
            lines.append('BARRIER '+json.dumps(dict(wgs=[w.wg for w,_ in pair],events=previous),separators=(',',':')))
