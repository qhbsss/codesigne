#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/Project/homework/submission-repair-20261002/official-linux
export PYTHONPATH=/mnt/d/Project/homework/submission-repair-20261002/numpy-site
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python3 ../preflight_linux.py
if test -e local-grade.json; then
  echo 'Refusing to overwrite existing grade report' >&2
  exit 1
fi
echo "Starting official seed 7 grade: $(date -Is)"
python3 -u challenge.py grade --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --baseline baseline_manifest.json --seed 7 --report local-grade.json > ../linux-grade.stdout.log 2>&1
echo "Official grade finished: $(date -Is)"
