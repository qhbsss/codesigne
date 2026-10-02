"""Bijective shortening of WG, loop and event identifiers; preserve interpolation."""
import argparse,re,json,string
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('path');a=p.parse_args();path=Path(a.path);s=path.read_text()
wg=sorted(set(re.findall(r'"wg":"([^"]+)"',s)))
vars=sorted(set(re.findall(r'"var":"([^"]+)"',s)))
events=sorted(set(re.findall(r'"event":"([^"]+)"',s)))
letters=string.ascii_letters
mapping={w:(letters[i] if i<len(letters) else 'g'+str(i-len(letters))) for i,w in enumerate(wg)}
mapping.update({v:'v'+str(i) for i,v in enumerate(vars)})
for i,e in enumerate(events):
 refs=re.findall(r'\{([^}]+)\}',e)
 mapping[e]='e'+str(i)+''.join('_{'+mapping.get(v,v)+'}' for v in refs)
def repl(m):
 value=m[1]
 if value in mapping:return '"'+mapping[value]+'"'
 if '{' in value:
  value=re.sub(r'\{([^}]+)\}',lambda r:'{'+mapping.get(r[1],r[1])+'}',value)
 return '"'+value+'"'
s=re.sub(r'"([^"\\]*)"',repl,s);path.write_text(s);print('bytes',len(s.encode()))
