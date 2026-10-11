# 第一阶段优化结果（2026-10-11）

最高完整本地分数 **64279.27023644358**，eligible=true；官方score=null，八万分未达到。冻结版本：[final-optimized-64279-20261011](../final-optimized-64279-20261011/README.md)。

基线64214.04366988358，P1=301216、D1=32494。新版本P1=300605、D1=32494；分数提高0.10158%，P1减少611周期。面积23.966565304320003mm²，P1/D1峰值19.85698089031607/19.769776618522137W。

## 本轮六个候选

| 候选 | 范围 | 周期 | 峰值W | 检查与结论 |
|---|---|---:|---:|---|
| balanced | 完整P1+D1 |300605 / 32494|19.856981 / 19.769777|完整公开grade合格，64279.27023644358分 |
| d1-init-unbarriered | 完整D1计时 |32592|20.099363|双种子功能/竞争通过，但超功耗且慢于原版 |
| step-stream-paced | P1独立后缀 |53778|20.618196|双种子功能/竞争通过，但超功耗 |
| step-stream-paced2 | P1独立后缀 |58410|20.564850|仅计时，超功耗 |
| step-stream-paced1 | P1独立后缀 |64587|20.283480|仅计时，超功耗且变慢 |
| step-stream-projections-only | P1独立后缀 |56660|20.547824|仅计时，超功耗 |

balanced改变P1的r16..r31工作组SM映射，把第二batch较重/较轻的注意力查询块与另一batch配对；同组后续任务也使用该物理SM映射。D1逐字节不变。候选P1 seed7/19数值与HBM竞争检查通过，完整grade再执行seed7双案例；冻结版本另存seed19双案例检查。

d1-init-unbarriered取消初始化第32组后的全组屏障，实际消费者保留原依赖。流式系列采用N64/K64双权重缓冲，以真实LD完成事件控制下一批预取：每4、2、1组分别计时。仅投影版对QKV/WO采用流式加载，FFN恢复基线。所有失败均保留，不借用其他候选的功能检查。

门控从4组细化到1组，HBM读取从8876032增加至8968192字节（约1.04%），周期从53778增至64587。仅加入依赖不保证净收益；模型需要同时考虑窗口功耗、多播和等待。

## 评测边界与复现

公开评估器、固定模型、成本及评分均与64214基线逐字节一致。`candidates/balanced/candidate-grade.json`为未经编辑的完整双案例报告，运行耗时1372.19秒。复制的baseline-grade.json只是本地历史基线，不纳入Git；原始基线报告位于 ../final-optimized-64214-20261009/local-grade.json。

后缀工具假设保留的RF数据已准备，仅作局部筛选；不能将后缀或首层计时当作完整分数。balanced的重复首层筛选为让出CPU被主动终止，first-layer-interrupted.json保留原因；其完整grade独立完成。

公开评估器和baseline不重复纳入每个候选目录。首次从Git检出后，先在本实验目录执行 `python prepare.py`，从冻结64214基线恢复逐字节相同的公共依赖，再在候选目录执行：

```sh
OPENBLAS_NUM_THREADS=1 python -m project.build_programs
OPENBLAS_NUM_THREADS=1 python -m project.functional_check --seed 7 --seed 19 --report functional.json
OPENBLAS_NUM_THREADS=1 python challenge.py grade --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --baseline baseline_manifest.json --seed 7 --report candidate-grade.json
```

失败候选不需要重复完整grade。ASM未重复纳入Git，可由源码生成。`patches/`记录相对64214生成器的改动；`artifact-bindings.json`绑定各候选源码、程序与原始报告；运行 `python verify_records.py` 验证绑定和公开代码。旧候选继承的project.verify_artifacts属于历史基线，不用于证明新候选；冻结最高版本中的验证器才对应新完整成绩。

## 下一步

本轮仅获得小幅提升。三种预取门控与仅投影流式均未解决功耗；不要重复以“更多重叠”作为收益证明。下一次大幅优化应先逐阶段定位功耗峰值，再联合设计数据块归属、缓存驻留与加载节奏，保持源多播和完整写入依赖。没有证明八万分可达，也没有把参考图片成绩作为本仓库实测。
