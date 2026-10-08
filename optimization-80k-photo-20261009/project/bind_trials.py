"""Bind experiment reports to exact program and source bytes; no estimated scores."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 trials=[];highest=63872.93194880659
 for c in sorted((ROOT/'candidates').iterdir()):
  if not c.is_dir():continue
  files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in [c/'hardware.json',c/'M1_P1.asm',c/'M2_D1.asm'] if p.exists()}
  source={str(p.relative_to(c)):digest(p) for p in sorted((c/'project').glob('*.py'))}
  reports={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'reports').glob(c.name+'-*.json'))}
  full_score=None
  grade=ROOT/'reports'/(c.name+'-grade.json')
  if grade.exists():
   g=json.loads(grade.read_text())
   assert g['provenance']['program_sha256']=={case:files[case+'.asm']['sha256'] for case in ('M1_P1','M2_D1')}
   if g['eligible']:
    full_score=g['experimental_score'];highest=max(highest,full_score)
  trials.append(dict(candidate=c.name,artifacts=files,source_sha256=source,reports_sha256=reports,program_size_valid=all(v['bytes']<=8388608 for k,v in files.items() if k.endswith('.asm')),complete_dual_case_score=full_score))
 unchanged={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'codesign').rglob('*')) if p.is_file() and p.suffix in ('.py','.json')}
 out=dict(kind='photo-method-trials-byte-bindings-v1',baseline_commit='7f791b2',target_score=80000,achieved_target=highest>80000,highest_complete_local_score=highest,public_sources_sha256=unchanged,baseline_manifest_sha256=digest(ROOT/'baseline_manifest.json'),trials=trials)
 (ROOT/'trials-manifest.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
 print(len(trials),'candidates bound')
if __name__=='__main__':main()
