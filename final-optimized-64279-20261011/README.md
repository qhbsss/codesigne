# 第一阶段最高完整本地结果：64279.27023644

完整、未经编辑的公开评分见 local-grade.json，eligible=true，官方score=null，没有服务端验证回执。八万分尚未达到。

P1=300605周期、峰值19.85698089031607W；D1=32494周期、峰值19.769776618522137W；面积23.966565304320003mm²。

相对64214.04366988358增加65.22656656分（0.101577%）。改变P1的r16..r31工作组SM映射，以较重/较轻的注意力查询块配对；同组后续任务也使用新映射，完整P1减少611周期。D1程序逐字节不变。公开模型、成本和评估器未修改。

生成与验证：

```sh
OPENBLAS_NUM_THREADS=1 python -m project.verify_artifacts
OPENBLAS_NUM_THREADS=1 python challenge.py grade --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --baseline baseline_manifest.json --seed 7 --report rerun-grade.json
```

原始受评路径和provenance保留。实验记录见 ../optimization-phase-one-20261011/README.md，包含五个未采用候选与完整双案例评测。

该包为阶段提交材料；最终提交仍需按官方要求附完整可用原生agent记录。
