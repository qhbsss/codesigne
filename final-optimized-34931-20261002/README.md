# Transformer codesign：本轮最佳合格版本

本地公开评分器完整精评：**34931.267616664205 分**，相对历史最佳30394.215569735643提高14.9274%。**本轮未达到50000分。**

P1：528117 cycles、19.233686 W；D1：62630 cycles、18.073409 W；面积23.895888 mm²。功能、时序、面积与功耗门槛均通过。原始结果保存在 `local-grade.json`，未编辑评分数据；没有服务端验证结果。

采用双驻留工作组、矩阵块尾处理、W2分解及D1分段attention向量化合并。`project/`包含生成源码；程序重新生成后，两个ASM的SHA256均与原始评分报告一致。公开评分器源码保持历史版本不变。

## 复现

在本目录运行：

```bash
python -m project.compiler
OPENBLAS_NUM_THREADS=1 python challenge.py grade --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --baseline baseline_manifest.json --seed 7 --report local-grade-rerun.json
```

完整实验（包括未采用候选和原始日志）位于 `/workspace/codesigne/optimization-resume-50k-20261002`。方向分析、筛选结果和精评对比见 `reports/`。
