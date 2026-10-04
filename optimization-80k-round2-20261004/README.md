# 第二轮八万分优化实验

最高完整精评仍以仓库根LATEST_RESULT.md及未经编辑的local-grade.json为准；局部计时不是完整分数。此目录保留候选生成源码、未编辑报告和[数学分析](MODEL_80K.md)。候选ASM可在各candidate目录执行 `OPENBLAS_NUM_THREADS=1 PYTHONPATH=../.. python -m project.build_programs --output .` 重生成。

| 候选 | 测试范围 | 周期 | 峰值W | 结论 |
|---|---|---:|---:|---|
| p1-hybrid-w1-n64-m8 | P1第一W1 |29372|5.032|较慢，功能seed7通过 |
| p1-hybrid-w1-n64-m16-pipelined | P1第一W1 |28764|7.1724|较慢，功能seed7通过 |
| d1-prefetch-generated-kv | D1全部 |34582|19.74754|功能seed7/19通过 |
| d1-w2-balanced-early | D1全部 |33324|20.69199|超功耗 |
| d1-w2-balanced-gelu | D1全部 |33544|20.5046|超功耗 |
| d1-w2-balanced-only | D1全部 |33722|20.3935|超功耗 |
| d1-w2-balanced-cohort16 | D1全部 |33632|19.74754|功能seed7/19通过 |
| d1-cold-affine-m3 | D1全部 |34043|19.34979|较慢 |
| d1-cold-affine-m3-late-cohort | D1全部 |33583|19.34979|当前新实验最快合规D1，功能seed7/19通过 |
| d1-shared-norm-stats | D1全部 |34957|19.73649|广播开销过大，功能seed7/19通过 |
| combined-remove-tail-prefetch | P1第一层 |86088|19.43135|较旧第一层86165仅快77周期，两案例seed7通过 |

新测试：combined-rf-norm-dma2-depth2及combined-rf-norm-dma2-reduce2。进度及结果以reports原始报告为准。未完成评估不列为可提交最高分。

公开评分器codesign/challenge.py、成本模型、baseline均未修改。power_hotspots仅观察公开调度器产生的事件，并校验独立窗口计算与公开峰值一致。任何超过20W的候选不采用。

新增两个硬件方案均通过两案例seed7/19功能/竞争验证。D1精评：DMA2×depth2为32737周期、19.769776619W；DMA2×depth1+reduce2为34374周期、19.462313693W。仅单项结论，仍需P1及完整grade。第三个候选combined-rf-norm-dma2-early-w2将每个W1行块后加载的W2 bank由一个增加到两个，P1 seed7功能通过，性能待评。

更正：depth2的P1第一层为84541周期、20.110189896W，违反功耗；reduce2为86233周期、19.661298369W。depth2完整grade已因功耗超限提前终止，没有完整分数。新增split-acc2将W1/W2的奇偶K64分块放入独立RF14/15累加器，末尾合并；split-acc是生成代码替换未命中的零变化对照，ASM与depth2相同，没有优化收益结论。

提前W2载入候选P1第一层85598周期，但峰值20.364843365W，超20W，淘汰。它仅节省635周期，却失去功耗合规，说明原分批等待大部分被计算覆盖。split-acc2的P1 seed7/19功能/竞争检查已通过，第一层计时进行中。

split-acc2第一层87061周期、20.110189896W，比depth2慢2520周期且仍超功耗，淘汰；其减少了部分HBM源读取但新增RF初始化/合并开销和调度变化抵消收益。parity-waves第一层测试进行中。

wave6第一层85624周期、20.110189896W，峰值与depth2相同，仍淘汰。这说明此前按W1批次分析的热点位置假设不足，需要定位真实功耗窗口，而不是继续盲调批次数量。备用原硬件组合combined-tail-prefetch-m3-late已启动完整grade。

parity-waves第一层85545周期、20.110189896W，峰值也相同，淘汰。已启动depth2第一层功耗窗口定位；W1批次是待验证假设，不能作为已确认的解释。

真正的depth2热点位于初始QKV权重加载，first-layer-hotspots原始报告独立能量窗口计算与公开峰值一致。新qkv3缩小这一阶段批次，第一层精评进行中。已准备cache1及cache1-dma2两个面积合规配置，cache1 D1单项评估开始；未测试配置不列性能结论。

cache1 D1为41327周期、13.188384913W，比同程序无cache的33583周期显著变慢。低功耗不能抵消cache查找/填充的延迟，不采用；cache1-dma2只准备硬件，尚未测试，没有性能结论。

## 新的完整成绩

combined-tail-prefetch-m3-late完整公开seed7评分61669.06154945007，eligible=true；P1 316000周期/19.499781288178355W，D1 33583周期/19.349794304127606W，面积23.84637760512mm²。完整原始报告为reports/combined-tail-prefetch-m3-late-grade.json，另有seed19两案例功能通过报告。已冻结到../final-optimized-61669-20261004，逐字节重生成及public provenance验证通过。仍未达到80000。

qkv3第一层87113周期、18.701802265W，合规但比原分块慢；norm-fence隔离归一化与首批QKV加载，第一层计时进行中。D1 reduce8加宽归约以减少重复NoC递送，两个初始布局版本功能失败（附原始失败报告），已修复RF输入与累加器重叠，当前版本功能测试进行中。
