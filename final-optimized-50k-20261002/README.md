# Transformer codesign：50944 分合格版本

完整本地公开评分器精评：**50944.151728641693 分**，`eligible=true`，超过50000目标。相较之前提交的34931.267616664205分提高45.84%。这是公开评分器的`experimental_score`；没有服务端官方验证结果，原始报告中的`score`仍为null。

| 案例 | 周期 | 1000-cycle滚动峰值功耗 |
|---|---:|---:|
| P1 | 374249 | 18.097779682953387 W |
| D1 | 41552 | 17.463376854983895 W |

面积23.36637760512 mm²。完整grade种子7及额外种子19均通过数值与HBM竞态检查。两个程序大小5792748/6280902字节，均低于8 MiB。公开评分器、成本模型与冻结baseline未修改。

## 复现

在本目录运行：

```bash
python -m project.build_programs
python -m project.verify_artifacts
OPENBLAS_NUM_THREADS=1 python challenge.py grade --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --baseline baseline_manifest.json --seed 7 --report local-grade-rerun.json
```

完整模拟本环境约27分钟。`project.build_programs`包含确定性的名称压缩，两个ASM逐字节重生成与受评输入相同。`local-grade.json`是未编辑的原始完整报告；其中artifact_paths保留当时实验目录路径，provenance哈希与本目录硬件、ASM、评分器一致，可用verify_artifacts验证。

关键改进是P1当前权重RF缓存、六bank行预取流水线、WO N16任务、packed W2归并，D1两层权重与历史KV常驻RF，以及针对持续HBM峰值的8-WG K/V分批搬运。优化推导、淘汰方向和失败完整报告见[分析记录](reports/optimization-analysis.md)。
