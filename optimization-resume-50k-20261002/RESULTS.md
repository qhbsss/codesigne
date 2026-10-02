# 完整精评结果
分数仅引用原始公开 `challenge.py grade` 报告。功能-only、单项时序和阶段时序不能替代完整grade。
|版本|P1 cycles|D1 cycles|面积 mm²|P1 W|D1 W|合格本地分|
|---|---:|---:|---:|---:|---:|---:|
|历史起点|611770|71412|23.935701|16.935987|18.074404|30394.21557|
|resident2|540914|71412|23.935701|19.234681|18.074404|32323.68877|

原始报告：[resident2-local-grade.json](reports/resident2-local-grade.json)。

保守组合完整精评合格：**34931.26762分**。P1 528117 cycles / 19.233686 W；D1 62630 cycles / 18.073409 W；面积23.895888 mm²。原始报告为 reports/conservative-final-local-grade.json。

merged128完整精评不合格：P1 544669 / 20.489169 W，D1 62630 / 18.073409 W。功耗超限，不能计分。

RF-stationary P1功能通过，W1孤立段33343 cycles（低于保守版41519），功耗8.770227 W，HBM读取274624 bytes。但尚无全程序有效分。50k目标尚未证明达成。
