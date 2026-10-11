"""Bijective identifier shortening, preserving JSON keys and event references."""
import json,re,string

def compact(text):
    wgs=sorted(set(re.findall(r'"wg":"([^\"]+)"',text)))
    variables=sorted(set(re.findall(r'"var":"([^\"]+)"',text)))
    events=sorted(set(re.findall(r'"event":"([^\"]+)"',text)))
    letters=string.ascii_letters
    def short(i):
        result=''
        while True:
            result=letters[i%52]+result;i=i//52-1
            if i<0:return result
    mapping={w:(letters[i] if i<len(letters) else 'g'+str(i-len(letters))) for i,w in enumerate(wgs)}
    mapping.update({v:'v'+short(i) for i,v in enumerate(variables)})
    for i,event in enumerate(events):
        refs=re.findall(r'\{([^}]+)\}',event)
        mapping[event]='e'+short(i)+''.join('_{'+mapping.get(v,v)+'}' for v in refs)
    def rewrite(value):
        if isinstance(value,dict):return {k:rewrite(v) for k,v in value.items()}
        if isinstance(value,list):return [rewrite(v) for v in value]
        if isinstance(value,str):
            if value in mapping:return mapping[value]
            return re.sub(r'\{([^}]+)\}',lambda r:'{'+mapping.get(r[1],r[1])+'}',value)
        return value
    lines=[]
    for line in text.splitlines():
        if not line.strip():continue
        op,payload=line.split(' ',1)
        lines.append(op+' '+json.dumps(rewrite(json.loads(payload)),separators=(',',':')))
    return '\n'.join(lines)+'\n'
