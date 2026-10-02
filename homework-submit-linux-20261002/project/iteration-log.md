# 作业迭代记录

整理日期：2026-10-02。本记录是根据已保存源码、评分报告和原始会话整理的摘要，不是当时逐条记录的原始日志；详细过程以 `agent-trace/` 为准。

## 1. 任务与评分分析

阅读作业公告、README、ISA、ABI 和评分器源码。目标是在两个模型/场景都通过功能检查的前提下，优化硬件与汇编程序的组合。约束包括面积不超过 24 mm²、每场景峰值滚动 1000-cycle 平均功耗不超过 20 W、程序大小与展开指令/循环限制，以及相对基准的延迟门槛。冻结基准分为 1000。

## 2. 早期优化

探索 GEMM 分块、attention 分块、workgroup/SM 分配以及 RF 带宽和 TC K 并行度。曾以约 10465 分的完整评分版本作为回退。为节省精评时间，随后优先使用功能检查、粗估和单场景精确 timing 筛选候选。粗估并不能证明有效分数；面积、功耗、数值正确性和最终完整 grade 都需要验证。

## 3. 本次正式提交版本：28885.37935367514

最终选择已有完整 seed 7 `grade` 报告的版本：

- 16 SM，每 SM 一个 8×16 TC，K 并行度 2，RF 8R4W。
- 4 MiB cache、8 HBM channels、512 B/cycle 全局 NoC、64 B/cycle SM NoC。
- 2 DMA engines、DMA depth 1、无 SH、无专用 reduction units。
- P1 两个 batch 并行，每个 batch 8 个 worker，GEMM 列和 attention head 分配到 worker；prompt attention 按查询块重用 K/V。
- D1 使用 32 个 worker，每 head 八段历史键范围；权重常驻 RF，保持八次 `STEP.COMMIT` 的先后语义。

原始结果见根目录 `local-grade.json`：P1 680994 cycles，D1 71030 cycles，面积 23.895888312319997 mm²；两场景滚动峰值功耗分别 19.595123013482983 W 和 18.073408655806908 W。两个场景功能检查和全部评分门槛通过。

报告的 `experimental_score` 与 `gate_diagnostics.score` 为 28885.37935367514，顶层 `score=null`、`status=candidate` 是原评分器输出状态，未为展示分数而改写。此处仅声明本地分数。

## 4. 后续未采用的冲分实验

后续围绕 5–8 万分目标尝试或分析了 H50 硬件、D1 历史 KV 常驻 SH/4 段和 8 段、QKV 融合、N=128、LayerNorm 参数缓存、step batch 合并以及 P1 row pipeline。留存的相关报告在 `project/reports/`，详细工具命令、代码变更和判断见原始会话。

这些文件包括粗估、功能检查和单场景 timing，不等于新的完整有效评分。没有把这些实验的源码或 ASM 当成 28885 分提交件，也没有将其预测分数写入提交评分报告。本次打包请求优先于继续优化。

## 5. 本次补全提交格式

此前精简 ZIP 缺少公告要求的迭代日志、项目源码和 agent trace。本次根据公告与更新版 README 的合并要求补齐，并保留已验证版本的硬件、ASM 和原始报告。

从留存 checkpoint 恢复生成器，并在打包副本内恢复与提交 D1 对应的单行 GEMM 与八段 attention RF 偏移。该恢复仅用于复现冻结版本，不修改主工作目录中的后续实验。已逐个比较生成器输出与提交 ASM 的全文，P1 和 D1 均一致（按文本读取后的换行规则）。

打包验证另外检查官方 provenance、seed 7、原始报告 SHA-256、ISA 解析与限制、trace JSONL 格式、ZIP 内容和压缩包体积。所有文件逐项 SHA-256 见 `project/file-manifest.json`。此次没有重复耗时完整 grade；沿用与相同提交件匹配的已有精评原始结果。

## 6. 复验命令

在包根目录执行：

```text
python -m project.verify_package
python -m project.functional_check --seed 7 --race-only
python challenge.py grade --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --baseline baseline_manifest.json --seed 7 --report recheck-grade.json
```

最后一条为可选的完整精评，会耗时较长；应保留附带报告不动，写入新文件。
