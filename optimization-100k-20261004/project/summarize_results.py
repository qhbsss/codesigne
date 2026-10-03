"""Summarize complete raw grades without promoting partial timing to scores."""
import hashlib
import json
from pathlib import Path

paths = [Path('../final-optimized-50k-20261002/local-grade.json')]
paths += sorted(Path('reports').glob('*local-grade.json'))
rows = []
for path in paths:
    g = json.loads(path.read_text())
    rows.append({
        'raw_grade': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'eligible': g['eligible'],
        'experimental_score': g['experimental_score'],
        'official_score': g['score'],
        'cases': {
            case: {key: data['timing'][key] for key in
                   ('cycles', 'area_mm2', 'peak_window_power_w')}
            for case, data in g['cases'].items()
        },
    })
eligible = [r for r in rows if r['eligible'] and r['experimental_score'] is not None]
best = max(eligible, key=lambda r: r['experimental_score'])
out = {
    'kind': 'complete-public-local-grade-summary',
    'grades': rows,
    'best': best,
    'target_score': 100000,
    'target_achieved': best['experimental_score'] > 100000,
    'limitations': 'Local experimental scores only. Partial probes and stopped grades have no scores.',
}
Path('reports/complete-grade-summary.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({'best_score': best['experimental_score'], 'target_achieved': out['target_achieved']}))
