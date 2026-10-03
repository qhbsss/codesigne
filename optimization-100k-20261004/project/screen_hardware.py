"""Run unchanged public D1 timing sequentially for saved hardware variants.

These are single-case timing reports, not complete grades. Keeping the cases
sequential limits CPU use while a complete grade runs independently.
"""
import argparse
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--program', required=True)
p.add_argument('--variants', nargs='+', required=True)
a = p.parse_args()
for variant in a.variants:
    report = Path('reports') / (variant + '-d1-exact.json')
    if report.exists():
        raise FileExistsError(report)
    with report.with_suffix('.log').open('w') as log:
        subprocess.run([
            sys.executable, '-m', 'project.exact_case', '--case', 'M2_D1',
            '--hardware', str(Path('reports') / ('hardware-' + variant + '.json')),
            '--program', a.program, '--report', str(report),
        ], stdout=log, stderr=subprocess.STDOUT, check=True)
    print(report, flush=True)
