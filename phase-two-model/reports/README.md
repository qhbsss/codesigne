# 本轮真实结果与证据范围

这些是实际运行保存的结果，不是预期输出。官方运行核心源码hash为：

`e636eb550dfb9f326f252010f9667778e7bb5c00ce9060ea2fbf3a28b9cf4d19`

## 五个正式参考程序

attention_stress seed 7，模型预测和完整evaluate使用同一程序SHA256。

| 案例 | 周期 | 周期差 | 全部时序报告字段 | 数值检查 |
|---|---:|---:|---|---|
| w-p | 91208716 | 0 | 一致 | 通过 |
| a-p | 249709295 | 0 | 一致 | 通过 |
| w-d1 | 19698605 | 0 | 一致 | 通过 |
| w-d16 | 28034378 | 0 | 一致 | 通过 |
| w-d4l | 57484249 | 0 | 一致 | 通过 |

`five-reference-summary.json`汇总；`reference/<case>/`保留原始prediction、完整evaluation、comparison和命令/耗时。没有跑新的完整五案例优化，也没有生成聚合grade.json或服务器回执；1548.9935009仅为从这些原始周期推导的本地参考分数。

时序后端使用官方共享核心，因此零gap不构成独立模型等价证明。五案例比较完成后，适配器增加了官方非functional静态检查，覆盖HBM race、alias和额外shape规则；时间核心未改。最终适配器已重测小型完整程序和负例，未重复五个昂贵案例。原始报告保留各自后端源hash，不能把它们当成最终适配器同一个源码hash的运行。

## 独立模块验证

`tests.json`保存最终6个测试、运行日志与模型文件SHA256。正例覆盖12组确定性随机硬件：独立面积、所有付费compute profile、单组周期/功耗、多组固定轮转/共享端口/事件跳时。周期、bytes、FMA和scheduler iterations严格一致；energy/power按声明浮点容差一致。

负例包括容量、async destination未wait使用、跨组HBM写race，以及“时序可计算但数值错误”的完整程序。最后一个反例证明prediction不能当作eligible证书。

## 两种MIP

- `analytical-kernel-solution.json`：144个硬件/软件组合的独立数学模型，最优177周期，HiGHS gap=0；成本没有调用官方运行后端。求解后对照，各建模指标一致。这是compute-only GEMM，不是正式Transformer。
- `small-mip-solution.json`：小型完整Transformer的8个硬件/软件组合，最优23616周期，声明有限域log gap=0；完整程序时间使用共享精确运行子问题。本次下界较松，访问了全部8点，不宣称减少搜索评估次数。
- `small-best/`：生成的硬件、config与静态程序；不具备正式五案例shape。
- `small-best-seed19.json`：同一最优小程序在seed19也通过数值与时序对照，仍不是隐藏输入普遍证明。

## 实际模型修订

1. 初版硬件/微步骤lower bound与小型完整运行差距很大，说明仅core工作量不能作为真实延迟预测；它被明确降为定界用途。
2. 加入独立多组固定轮转、read/core/write phase和共享端口，计算子集不仅复现周期，也复现scheduler iterations。
3. 源码审核发现Scheduler不负责全部跨组HBM race和alias检查，增加官方非functional静态pass，并用故意交叉写的负例确认拒绝。
4. 增加数值错误程序反例，强制报告将conditional timing与functional correctness分开。

这些是实际实现/测试产生的修订链；尚未实现完整多智能体作业框架或通用浮点证明。
