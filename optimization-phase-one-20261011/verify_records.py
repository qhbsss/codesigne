"""Verify experimental artifacts and their unchanged public evaluator sources."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--bind', action='store_true', help='Create bindings after runs finish')
args = parser.parse_args()
base = root.parent / 'final-optimized-64214-20261009'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if args.bind:
    result = {'base': base.name, 'base_score': 64214.04366988358, 'candidates': {}}
    for candidate in sorted((root / 'candidates').iterdir()):
        paths = [*candidate.joinpath('project').glob('*.py'),
                 *candidate.joinpath('programs').glob('*.asm'),
                 *candidate.glob('*.json')]
        result['candidates'][candidate.name] = {'files': {
            str(p.relative_to(candidate)): sha(p) for p in sorted(set(paths))
            if p.name != 'baseline-grade.json'}}
    (root / 'artifact-bindings.json').write_text(json.dumps(result, indent=2) + '\n')

bindings = json.loads((root / 'artifact-bindings.json').read_text())
missing_programs = []
for name, item in bindings['candidates'].items():
    candidate = root / 'candidates' / name
    for relative, digest in item['files'].items():
        path = candidate / relative
        if not path.exists() and relative.startswith('programs/'):
            missing_programs.append(str(path))
            continue
        assert sha(path) == digest, f'Artifact changed: {path}'
    for p in candidate.joinpath('codesign').rglob('*.py'):
        assert p.read_bytes() == (base / p.relative_to(candidate)).read_bytes(), p
    grade = candidate / 'candidate-grade.json'
    if grade.exists():
        report = json.loads(grade.read_text())
        if all((candidate / 'programs' / f'{case}.asm').exists() for case in ('M1_P1', 'M2_D1')):
            import subprocess, sys
            command = '''import json
from pathlib import Path
from codesign.challenge.hardware import Hardware
from codesign.challenge.runner import provenance
r=json.loads(Path('candidate-grade.json').read_text())
h=Hardware.from_dict(json.loads(Path('hardware.json').read_text()))
p={c:Path('programs',c+'.asm').read_text() for c in ('M1_P1','M2_D1')}
assert provenance(h,p)==r['provenance']
'''
            subprocess.run([sys.executable, '-c', command], cwd=candidate, check=True)
print('PASS: bound source, reports, present programs, and unchanged public evaluator')
if missing_programs:
    print('Program byte verification incomplete; regenerate missing ASM:', len(missing_programs))
