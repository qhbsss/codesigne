# 十万分目标：数学分析与实验

数学推导、目标周期乘积、资源下界、限制条件见 [MATHEMATICAL_ANALYSIS.md](MATHEMATICAL_ANALYSIS.md)。本轮从已有完整合格的 50944.15172864169 分版本继续研究，不修改模型、公开评估器或成本模型。

## 完整实测结果

QKV48 分散加载方案完整本地精评 **51409.458970859465**，`eligible=true`。P1 367505 周期，D1 41552 周期，面积23.36637760512 mm²，峰值分别18.09777968295148 / 17.463376854983895 W。相较50944分约提高0.913%。未达到十万分。原始报告为 `reports/scatter4-local-grade.json`，冻结版本为 `../final-optimized-51409-20261004`。

完全移除行块屏障的方案在第一层测得21.40148747165039W，已停止，不存在完整分数；状态见 `reports/async-dense-stopped.json`。保留FFN屏障、只移除QKV屏障的候选仍在精评。

## 证据与复现

`reports/` 保存原始功能检查、局部精确计时、资源下界及完整 grade（存在时）。局部计时和被停止的 grade 不构成新分数。`candidate-manifest.json` 记录候选源文件与生成 ASM 的哈希。未受评的源码候选也保留，不能视为通过检查。

候选生成器保存在 `candidates/<name>/project/`，硬件为同目录 `hardware.json`。实验 ASM 不全部提交到 Git，按以下命令可重生成：

```bash
cd candidates/qkv48-scatter4
OPENBLAS_NUM_THREADS=1 PYTHONPATH=../.. python -m project.build_programs --output .
cd ../..
OPENBLAS_NUM_THREADS=1 python challenge.py grade \
  --hardware candidates/qkv48-scatter4/hardware.json \
  --program-p1 candidates/qkv48-scatter4/M1_P1.asm \
  --program-d1 candidates/qkv48-scatter4/M2_D1.asm \
  --baseline baseline_manifest.json --seed 7 --report reports/rerun-grade.json
```

这里 `PYTHONPATH=../..` 提供本轮原样复制的公开评估器包；工作目录中的候选 `project` 包优先。根目录 `project` 则保存基线编译器与诊断工具。`power_observed_grade.py` 仅记录进度并调用原始 grade，不改变评分路径。

## 已获得的筛选结果

| 方向 | 比较范围 | 基线周期 | 新周期 | 判断 |
|---|---|---:|---:|---|
| QKV 48列，顺序加载 | 隔离 QKV / 第一层完整前缀 | 15647 / 102279 | 13919 / 105507 | 计算快，但全层慢，停止完整评估，无新分数 |
| QKV 48列，HBM 分散加载 | 初始化/Norm/权重加载/QKV 前缀 | 22279 | 20031 | 继续完整精评 |
| 去掉行块全局屏障 | 同一完整 QKV 前缀 | 22279 | 18461 | 数值与竞争检查通过，继续评估 |
| FFN 64列，SH 权重 | 隔离 W1 | 23191 | 40207 | 淘汰 |
| SH 增加银行、端口 | 隔离 W1 | 23191 | 36639 | 淘汰 |
| FFN 64列，RF/SH 混合 | 隔离 W1 | 23191 | 36943 | 淘汰 |
| W1 32行分块 | 隔离 W1 | 23191 | 23787 | 淘汰 |
| W2 K128/N64 | 隔离 W2 | 17599 | 19775 | 数值通过，速度不佳，淘汰 |
| D1 打包注意力部分结果 | 完整单案例 D1 时序 | 41552 | 41658 | 数值通过，淘汰 |
| 增加1归约器 | 完整单案例 D1 时序 | 41552 | 42042 | 淘汰 |
| 增加2归约器，SFU8 | 完整单案例 D1 时序 | 41552 | 41603 | 淘汰 |
| DMA2/SFU13 | 完整单案例 D1 时序 | 41552 | 41350 | 峰值20.319W，淘汰 |
| TC8x8/K4 | 完整单案例 D1 时序 | 41552 | 41972 | 淘汰 |
| TC8x8/K4/DMA2 | 完整单案例 D1 时序 | 41552 | 41678 | 峰值20.328W，淘汰 |
| TC8x16/K1/DMA2 | 完整单案例 D1 时序 | 41552 | 47580 | 峰值20.131W，淘汰 |
| 两层交替16组 FFN | 完整单案例 D1 时序 | 41552 | 42142 | 数值通过，但峰值20.353W，淘汰 |

隔离 GEMM 从固定程序片段构造诊断程序，不包含权重缓存加载，不用于数值正确性或评分。第一层、QKV 前缀与完整 P1 是不同的比较范围。

`ffn16-draft-functional-seed19.json` 记录草稿传输长度不匹配的失败；修正后的程序与功能检查为 `ffn16-functional-seed19.json`，随后精确时序仍显示速度与功耗不合格。失败的草稿不代表保存的最终源码版本。

## 后续方向

十万分要求周期乘积缩到原来的 25.95%，本轮的局部改善不足以推断达标。优先研究能显著减少 NoC/RF 工作量的阶段融合、权重复用、数据布局与同步结构；同时检查新增通信、寄存器银行和 20W 滑动窗口限制。仅增加 TC、扩大分块或添加缓存，没有数学或实测依据能保证十万分。
