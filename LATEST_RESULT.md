# 当前最高完整精评结果

公开本地完整实验分数 **63872.93194880659**，`eligible=true`；八万分尚未达到。官方 `score=null`。

[可复现版本](final-optimized-63872-20261006/README.md)与[未经编辑的完整双案例精评](final-optimized-63872-20261006/local-grade.json)。P1：303331 周期、19.626366347462234 W；D1：32613 周期、19.76977661852208 W；面积 23.966565304320003 mm²。seed19 两案例数值与竞争通过；公开 provenance、baseline 与逐字节重生成验证通过。

[阶段提交 ZIP](homework-submit-63872-20261006.zip)及[SHA256](homework-submit-63872-20261006.zip.sha256)，无外层目录，包含原始完整评分与未修改公开评估器。最终截止提交需附完整 agent trace，详见 ASSIGNMENT.md。

相比 62982.58 分，W1 四个 M8 行块合并为一次 M32 GELU，并用真实中间 MAC 事件分散 W2 权重预取，加上 W2 双累加器与真实 MAC WAIT，P1 减少 7455 周期；D1 按注意力头收窄事件依赖，减少 124 周期。分数提高约 1.41%。此前完整结果均保留。

[最新分块与流水实验](optimization-80k-headflow-20261006/README.md)；[内存与流式实验](optimization-80k-memory-20261006/README.md)；[此前流水实验](optimization-80k-pipeline-20261006/README.md)；[上一轮数学模型](optimization-80k-round2-20261004/MODEL_80K.md)。局部计时不能替代完整 grade。
