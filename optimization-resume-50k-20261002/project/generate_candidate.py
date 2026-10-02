"""Load isolated candidate generators without changing the current project."""
import argparse,importlib.util,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('candidate');a=p.parse_args();root=Path('candidates')/a.candidate/'project'
spec=importlib.util.spec_from_file_location(a.candidate,root/'__init__.py',submodule_search_locations=[str(root)]);m=importlib.util.module_from_spec(spec);sys.modules[a.candidate]=m;spec.loader.exec_module(m)
s=__import__(a.candidate+'.schedule',fromlist=['generate_m1_p1'])
for name,g in [('M1_P1',s.generate_m1_p1),('M2_D1',s.generate_m1_d1)]:
 text,_,_=g();(root.parent/(name+'.asm')).write_text(text);print(name,len(text))
