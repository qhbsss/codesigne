"""Verify recorded artifact bindings and regenerate every saved candidate."""
import argparse,hashlib,json,tempfile,subprocess,sys,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--regenerate',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];ms=json.loads((root/'candidate-manifest.json').read_text());bn={m['candidate']:m for m in ms}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for m in ms:
 c=root/'candidates'/m['candidate']
 for name,expected in m['artifacts'].items():
  f=c/name
  if a.regenerate and name.endswith('.asm') and not f.exists():continue
  assert f.stat().st_size==expected['bytes'] and digest(f)==expected['sha256'],f
 for name,expected in m['generator_sha256'].items():assert digest(c/name)==expected,c/name
 if a.regenerate:
  with tempfile.TemporaryDirectory(prefix='codesign-trial-') as tmp:
   env=os.environ.copy();env.update(PYTHONPATH=str(root),OPENBLAS_NUM_THREADS='1')
   subprocess.run([sys.executable,'-m','project.build_programs','--output',tmp],cwd=c,env=env,check=True,stdout=subprocess.DEVNULL)
   for name in ['M1_P1.asm','M2_D1.asm']:assert digest(Path(tmp)/name)==m['artifacts'][name]['sha256'],(c,name)
bs=json.loads((root/'report-artifact-bindings.json').read_text())
for b in bs:
 assert digest(root/b['report'])==b['report_sha256'],b['report']
 assert b['artifacts']==bn[b['candidate']]['artifacts'],b['candidate']
 d=json.loads((root/b['report']).read_text())
 if 'provenance' in d:
  art=b['artifacts']
  from codesign.challenge.runner import digest_json
  from codesign.challenge.hardware import Hardware
  hardware=Hardware.from_dict(json.loads((root/'candidates'/b['candidate']/'hardware.json').read_text()))
  assert d['provenance']['hardware_sha256']==digest_json(hardware.to_dict())
  assert all(d['provenance']['program_sha256'][case]==art[case+'.asm']['sha256'] for case in ['M1_P1','M2_D1'])
print('PASS:',len(ms),'candidate bindings;',len(bs),'raw reports; regeneration:',a.regenerate)
