import importlib.util,json
from pathlib import Path
from project.schedule import generate_m1_p1
from codesign.challenge.isa import _prepare,_expand
from project.strip_tail_prefetch import strip as guarded
p=Path('../../project/strip_tail_prefetch.py');s=importlib.util.spec_from_file_location('oldstrip',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
text,_,_=generate_m1_p1();a=guarded(text);b=m.strip(text)
def expanded(text):
 lines,ends=_prepare(text.splitlines());return ((i.op,i.args) for i in _expand(lines,ends,0,len(lines),{},[0]))
from itertools import zip_longest
n=0
for left,right in zip_longest(expanded(a),expanded(b)):
 assert left==right,(n,left,right)
 n+=1
out=dict(kind='guarded-tail-expanded-instruction-equivalence',passed=True,dynamic_instructions=n,guarded_source_bytes=len(a.encode()),peeled_source_bytes=len(b.encode()))
Path('../../reports/p1-w2-four-m8-guarded-tail-equivalence.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
