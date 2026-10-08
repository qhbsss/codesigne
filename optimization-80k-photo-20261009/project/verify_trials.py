"""Verify source/report bindings and reproduce ignored generated ASM artifacts."""
from pathlib import Path
import hashlib,json,subprocess,sys,tempfile,os
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((ROOT/'trials-manifest.json').read_text());count=0
for path,sha in m['public_sources_sha256'].items():assert digest(ROOT/path)==sha,path
assert digest(ROOT/'baseline_manifest.json')==m['baseline_manifest_sha256']
for t in m['trials']:
 c=ROOT/'candidates'/t['candidate']
 for path,sha in t['source_sha256'].items():assert digest(c/path)==sha,path
 for path,sha in t['reports_sha256'].items():assert digest(ROOT/path)==sha,path
 assert digest(c/'hardware.json')==t['artifacts']['hardware.json']['sha256']
 with tempfile.TemporaryDirectory(prefix='photo-rebuild-') as d:
  env=dict(os.environ,PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1')
  p=subprocess.run([sys.executable,'-m','project.build_programs','--output',d],cwd=c,env=env,capture_output=True,text=True)
  assert p.returncode==0,(t['candidate'],p.stderr)
  for name in ['M1_P1.asm','M2_D1.asm']:
   rebuilt=Path(d)/name
   assert digest(rebuilt)==t['artifacts'][name]['sha256'],(t['candidate'],name,'SHA mismatch')
   assert rebuilt.stat().st_size==t['artifacts'][name]['bytes']
  count+=1;print(t['candidate'],'source, reports and byte regeneration passed',flush=True)
print(count,'candidates verified; no complete score inferred from partial timings')
