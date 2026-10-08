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
        if op in ('add','mul'):
            flat=[]
            for x in xs:
                if isinstance(x,dict) and list(x)==[op]:flat.extend(x[op])
                else:flat.append(x)
            constants=[x for x in flat if type(x)==int];others=[x for x in flat if type(x)!=int]
            value=sum(constants) if op=='add' else math.prod(constants)
            if op=='mul' and value==0:return 0
            if not others:return value
            if value!=(0 if op=='add' else 1):others.append(value)
            return others[0] if len(others)==1 else {op:others}
    return out

def compact(text):
    result=[]
    for line in text.splitlines():
        op,payload=line.split(' ',1);result.append(op+' '+json.dumps(fold(json.loads(payload)),separators=(',',':')))
    return '\n'.join(result)+'\n'
