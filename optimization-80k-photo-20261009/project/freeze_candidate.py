"""Save a complete eligible raw grade with the exact evaluated artifacts."""
import argparse
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--grade', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
g = json.loads(a.grade.read_text())
assert g['eligible'] and g['experimental_score'] > 63872.93194880659
assert all(g['cases'][case]['functional_passed'] for case in ('M1_P1', 'M2_D1'))
if a.output.exists():
    raise FileExistsError(a.output)
a.output.mkdir()
ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
for name in ('codesign',):
    shutil.copytree(name, a.output / name, ignore=ignore)
shutil.copytree(a.candidate / 'project', a.output / 'project', ignore=ignore)
for name in ('challenge.py', 'baseline_manifest.json'):
    shutil.copy(name, a.output / name)
shutil.copy(a.candidate / 'hardware.json', a.output / 'hardware.json')
(a.output / 'programs').mkdir()
for case in ('M1_P1', 'M2_D1'):
    shutil.copy(a.candidate / (case + '.asm'), a.output / 'programs' / (case + '.asm'))
shutil.copy(a.grade, a.output / 'local-grade.json')
score = g['experimental_score']
rows = []
for case in ('M1_P1', 'M2_D1'):
    t = g['cases'][case]['timing']
    rows.append(f"| {case} | {t['cycles']} | {t['peak_window_power_w']:.12f} W |")
text = f'''# Transformer codesign：{score:.8f} 分完整精评版本

公开本地评估器完整实验分数 **{score:.12f}**，`eligible=true`。这是原始报告的 `experimental_score`；没有服务器官方验证结果，`score`仍为null。八万分目标{'已经达到' if score > 80000 else '尚未达到'}。

| 案例 | 周期 | 1000周期滑动窗口峰值功耗 |
|---|---:|---:|
''' + '\n'.join(rows) + f'''

生成源码、硬件与程序均对应未经编辑的 `local-grade.json`。保留原始报告中的受评路径与哈希；执行以下命令验证 provenance、冻结 baseline 及逐字节重生成：

```bash
python -m project.verify_artifacts
OPENBLAS_NUM_THREADS=1 python challenge.py grade --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --baseline baseline_manifest.json --seed 7 --report rerun-grade.json
```

候选 `{a.candidate.name}`。本轮数学推导、功能与功耗失败的方案、局部计时和完整评估记录见 [实验目录](../optimization-80k-photo-20261009/README.md)。Transformer 模型、评估器与成本模型保持不变。
'''
(a.output / 'README.md').write_text(text)
print(a.output)
