# 当前最高完整精评结果

公开本地完整实验分数 **62270.94140748526**，`eligible=true`；八万分尚未达到。官方`score=null`。

[可复现版本](final-optimized-62270-20261006/README.md)与[原始完整双案例精评](final-optimized-62270-20261006/local-grade.json)。P1：317930周期、18.701802265360474W；D1：32737周期、19.769776618522055W；面积23.966565304320003mm²。seed19两案例数值与竞争通过，哈希绑定及公开provenance、baseline、逐字节重生成验证通过。

[阶段提交ZIP](homework-submit-62270-20261006.zip)及[SHA256](homework-submit-62270-20261006.zip.sha256)，无外层目录，完整原始精评和公开评估器包含在内。最终截止提交需附完整agent trace，详见ASSIGNMENT.md。

收益来自用闲置RF保存归一化参数、释放SH面积增加DMA并发，以及在P1归一化之后加载QKV避免超功耗。此前61669等完整版本保留。

[最新方向与实验](optimization-80k-pipeline-20261006/README.md)；[上一轮数学模型](optimization-80k-round2-20261004/MODEL_80K.md)。未完成完整grade的候选不会替代最高完整成绩。
