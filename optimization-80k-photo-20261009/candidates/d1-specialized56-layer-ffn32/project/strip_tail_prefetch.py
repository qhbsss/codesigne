"""Peel each dense loop's last tile and omit its unused next-tile DMA."""
import json

def mentions(value,ident):
    if isinstance(value,dict):return value.get('var')==ident or any(mentions(v,ident) for v in value.values())
    if isinstance(value,list):return any(mentions(v,ident) for v in value)
    return False

def next_tile(value,ident):
    if isinstance(value,dict):return ('min' in value and mentions(value['min'],ident)) or any(next_tile(v,ident) for v in value.values())
    if isinstance(value,list):return any(next_tile(v,ident) for v in value)
    return False

def strip(text):
    lines=text.splitlines();result=[];index=0;changed=0
    while index<len(lines):
        line=lines[index]
        if not line.startswith('FOR '):result.append(line);index+=1;continue
        header=json.loads(line[4:]);depth=1;end=index+1
        while depth:
            if lines[end].startswith('FOR '):depth+=1
            elif lines[end].startswith('END.FOR'):depth-=1
            if depth:end+=1
        body=lines[index+1:end];ident=header['var'];omit=[]
        if header['start']==0 and isinstance(header['stop'],int) and header['stop']>1 and header['step']==1:
            for i,b in enumerate(body):
                if b.startswith('LD '):
                    a=json.loads(b[3:])
                    if a['src']['space']=='HBM' and a['dst']['space']=='RF' and next_tile(a['src'].get('offset'),ident):omit.append(i)
        if omit:
            assert any(b.startswith('MMA.ACC ') for b in body)
            result.append(line)
            for i,command in enumerate(body):
                if i in set(omit):
                    # One iteration before the final tile; zero on the final tile.
                    stop={'min':[{'add':[{'var':ident},1]},header['stop']-1]}
                    guard=dict(var='unused_prefetch_guard',start={'var':ident},stop=stop,step=1)
                    result.append('FOR '+json.dumps(guard,separators=(',',':')))
                    result.append(command);result.append('END.FOR {}')
                else:result.append(command)
            result.append(lines[end]);changed+=1
        else:result.extend(lines[index:end+1])
        index=end+1
    assert changed==12,changed
    return '\n'.join(result)+'\n'
