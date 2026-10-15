# 五场景有限域：先预测选解，再完整评分

本轮仅比较参考硬件附近七个**明确生成的参数组合**，不是23个硬件字段及所有合法软件程序的全局最优证明。五场景共享硬件，种子7，使用相同的 `phase-two-static-v1` / `phase-two-compact-v2` 合约。预填充场景关闭与参考硬件 `sh_tc_bw=0` 不兼容的 `policy.sh_direct`。程序来自正式 `export`，独立估计器计算周期，选解时不读取这些新程序的实测成绩。

目标是对静态容量合格的候选最小化

\[
\hat J(d)=\sum_c w_c\log\hat T_c(d),\qquad
\hat S(d)=1000\exp\!\left(\sum_c w_c\log\frac{T_c^{ref}}{\hat T_c(d)}\right).
\]

只有全部场景通过评分器，\(\hat S\) 才是可以与实测分对比的**条件预测**。每wave还须检查每SM的 \(\sum r_g\le H_{rf}\)、\(\sum h_g\le H_{sh}\) 及并驻组数限制。

| RF KiB | 并驻组 | split_k | K分块 | 预测条件分 | 并驻容量检查 |
|---:|---:|---:|---:|---:|---|
| 64 | 2 | 1 | 原配置 | 1657.4162 | 拒绝；原始评分器五场景均失败 |
| 64 | 2 | 2 | 原配置 | 1558.5128 | 拒绝；五场景静态检查失败 |
| 128 | 2 | 1 | 原配置 | **1588.5336** | 通过，按预测选中 |
| 128 | 2 | 2 | 原配置 | 1498.2764 | 静态容量通过，未完整评分 |
| 64 | 1 | 1 | 原配置 | 1391.4972 | 静态容量通过，未完整评分 |
| 64 | 2 | 1 | 64 | — | 拒绝；预填充wave申请36+36 KiB RF |
| 64 | 2 | 1 | 32 | — | 拒绝；预填充wave申请36+36 KiB RF |

首轮估计遗漏了同SM多组RF/SH用量之和，故把1657.4162误选为最佳；`reference-four-designs-grade-20261010/grade.json` 的成绩是 `null`，**不能当作1657分**。模型已增加逐wave容量检查与回归测试；收缩为单组或仅缩小K分块均未超越可行的128 KiB参考配置。`rf_kib=128,split_k=1` 的五个程序SHA256与先前已评分的参考程序完全相同，这次仍重新运行评分器，未把旧分数冒充新结果。

完整评分器结果：`phase-two-starter/runs/reference-selected-grade-20261010/grade.json` 的 `status=complete`，`workers/summary.json` 中 `all_checked=true`、`all_feasible=true`，**1548.9935009026813分**（本地公开种子7，非服务器签名成绩）；硬件面积99.586293 AU。估计高于实测 **39.540103分，即相对实测高估2.552632%**。

| 场景 | 独立预测周期 | 完整评分周期 | (预测/实测−1) |
|---|---:|---:|---:|
| w-p | 87966137 | 91208716 | −3.5551% |
| a-p | 245330677 | 249709295 | −1.7535% |
| w-d1 | 18880252 | 19698605 | −4.1544% |
| w-d16 | 27797676 | 28034378 | −0.8443% |
| w-d4l | 56374620 | 57484249 | −1.9303% |

原始四点及全部哈希在 `phase-two-starter/runs/reference-four-designs-search-20261010/search.json`；RF64单组追加搜索在 `phase-two-starter/runs/reference-rf64-single-group-search-20261010/search.json`。K分块尝试的失败产物分别在 `phase-two-starter/runs/reference-rf64-small-k-search-20261010/` 和 `phase-two-starter/runs/reference-rf64-k32-search-20261010/`。原始四点搜索产生于添加并驻容量约束**之前**，故其 `best/` 是已被否决的候选；修正后不得继续使用它评分。已评分选解实际位于 `phase-two-starter/runs/reference-four-designs-search-20261010/candidate-0002/programs/`。

结论仅为**上述七组合中通过必要静态检查者的独立估计最优，以及选中程序公开种子上的真实成绩**：未对其余静态合格方案做完整评分，不能证明实测排名相同。除容量外，独立估计仍未完整刻画Cache、DMA/HBM信用与NoC逐事件争用、峰值功耗和FP32数值；其他硬件、软件分块和私有输入可能改变排名和有效性。
