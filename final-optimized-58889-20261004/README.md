# Transformer codesign：58889.39106291 分完整精评版本

公开本地评估器完整实验分数 **58889.391062909606**，`eligible=true`。这是原始报告的 `experimental_score`；没有服务器官方验证结果，`score`仍为null。八万分目标尚未达到。

| 案例 | 周期 | 1000周期滑动窗口峰值功耗 |
|---|---:|---:|
| M1_P1 | 331379 | 19.499781288178 W |
| M2_D1 | 35119 | 19.747539475556 W |

生成源码、硬件与程序均对应未经编辑的 `local-grade.json`。保留原始报告中的受评路径与哈希；执行以下命令验证 provenance、冻结 baseline 及逐字节重生成：

```bash
python -m project.verify_artifacts
OPENBLAS_NUM_THREADS=1 python challenge.py grade --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --baseline baseline_manifest.json --seed 7 --report rerun-grade.json
```

候选 `combined-wo-prefetch-parallel-step`。本轮数学推导、功能与功耗失败的方案、局部计时和完整评估记录见 [实验目录](../optimization-80k-20261004/README.md)。Transformer 模型、评估器与成本模型保持不变。
