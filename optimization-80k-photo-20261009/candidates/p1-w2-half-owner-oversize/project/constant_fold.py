import json,math
OPS={'add','mul','min','max','ceildiv','mod'}
def fold(v):
    if isinstance(v,list):return [fold(x) for x in v]
    if not isinstance(v,dict):return v
    out={k:fold(x) for k,x in v.items()}
    if len(out)==1 and (op:=next(iter(out))) in OPS:
        xs=out[op]
        if all(type(x)==int for x in xs):
            if op=='add':return sum(xs)
            if op=='mul':return math.prod(xs)
            if op=='min':return min(xs)
            if op=='max':return max(xs)
            if op=='ceildiv':return -(-xs[0]//xs[1])
            if op=='mod':return xs[0]%xs[1]
    return out

def compact(text):
    result=[]
    for line in text.splitlines():
        op,payload=line.split(' ',1);result.append(op+' '+json.dumps(fold(json.loads(payload)),separators=(',',':')))
    return '\n'.join(result)+'\n'
