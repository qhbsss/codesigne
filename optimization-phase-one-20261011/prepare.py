"""Restore unmodified public dependencies from the retained 64214 baseline."""
from pathlib import Path
import shutil
root = Path(__file__).resolve().parent
base = root.parent / 'final-optimized-64214-20261009'
for candidate in sorted((root / 'candidates').iterdir()):
    files = [base / 'challenge.py', base / 'baseline_manifest.json']
    files += [p for p in (base / 'codesign').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    for source in files:
        dest = candidate / source.relative_to(base)
        if dest.exists():
            assert dest.read_bytes() == source.read_bytes(), dest
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
print('Prepared unchanged public evaluator and frozen baseline for all candidates')
