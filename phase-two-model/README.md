# 第二阶段：可运行数学模型与精确运行子问题

当前实现绑定原始 starter 的 `phase-two-compact-v2` / `phase-two-static-v1`，官方源码未修改。

这不是已经证明所有合法程序全局最优的模型。实现有两个明确不同的求解模式：

1. **纯数学内核 MIP**：独立 Python 面积、微步骤和单组时序方程计算候选成本，HiGHS 求解，不调用运行后端获取目标值。输出后才与官方核心对照。范围是有限的 compute-only GEMM 内核。
2. **完整程序 MIP + 精确运行子问题**：硬件/软件变量、面积和独立微步骤下界形成 MIP；程序生成后，共享的官方状态转移核心精确计算周期、流量和功耗。不是独立模拟器，也不是免运行的闭式预测。数值正确性另由完整官方 evaluator 检查。

第二种方式有意复用官方核心：Cache/LRU、HBM credits、DMA FIFO、NoC、多播、轮转和事件跳时不采用理想排程近似。因此能最大程度消除时序语义漂移，但“时序一致”不是独立等价证明，更不是FP32正确性证明。

## 文件

| 文件 | 职责 |
|---|---|
| `MODEL.md` | 已实现数学约束、目标、运行语义、最优性范围、剩余gap |
| `model/hardware.py` | 23硬件字段、组合合法性、独立完整面积和功耗方程 |
| `model/profiles.py` | 独立付费微步骤与保守资源下界 |
| `model/serial.py` | 单组无HBM计算的独立周期、能量和滑动窗口方程 |
| `model/compute.py` | 多组/多SM计算的独立固定仲裁、共享端口和事件跳时模型 |
| `model/kernel.py` | 不用运行成本的纯数学内核MIP、代码生成和事后对照 |
| `model/tool.py` | 完整程序预测、对照、有限空间MIP分解和JSONL生成 |
| `runtime/src/main.rs` | 官方不可变状态转移后端；不执行数值reference检查 |
| `tests/test_model.py` | 独立模型与官方微步骤/运行报告的差分测试 |
| `specs/small.json` | 8方案小型完整Transformer搜索空间 |
| `specs/five-bounded.json` | 五场景共享硬件的4方案接口示例，尚未作为优化实验跑完 |
| `reports/` | 本轮保留的真实验证结果与求解摘要 |

## 构建

需要 Python 3.11+、NumPy/SciPy（HiGHS MILP）、Rust 1.90+。从 `phase-two-model/` 执行：

```sh
python3 -m pip install -r requirements.txt
cargo build --release --locked --manifest-path runtime/Cargo.toml
cargo build --release --locked --manifest-path ../phase-two-starter/source/Cargo.toml --features compact --bin vnext-concurrent
python3 -m unittest discover -s tests -v
```

锁文件保留；官方源hash、程序SHA256、后端源hash和命令绑定在报告中。执行目录必须是新的，工具不会覆盖历史结果。超时、失败的命令日志保留；主机失败不会静默作为“数学不可行”剪枝。

## 使用

### 纯数学内核求解

```sh
python3 -m model.kernel runs/new-kernel
```

解空间同时包含RF容量/端口、TC形状/K并行度及软件K分块，共144组合；按数学周期和功耗求解，输出 `hardware.json`、`kernel-wave.json`、`solution.json`。这是内核，不是五案例提交程序，也没有官方成绩。

### 完整程序精确时序预测

```sh
python3 -m model.tool predict ../phase-two-starter/examples/small-prefill.jsonl runs/new-prediction
```

输出 `prediction.json`。`numerical_correctness=not_checked`，不能以此声明eligible。它不计算FP32数据、初始化有效性或独立reference输出；功耗检查结果仅属于时序层。

### 预测与完整验证逐字段对照

```sh
python3 -m model.tool compare ../phase-two-starter/examples/small-prefill.jsonl runs/new-comparison --seed 7 --timeout 600
```

对照完整 `report`、header、step cycles、work units、HBM峰值和程序/源码hash，保留两份原始报告及 `comparison.json`。完整数值报告独立运行，但时序核心共享，不能把它包装成两个独立模拟器相互验证。

### 小型完整硬件—软件MIP

```sh
python3 -m model.tool optimize specs/small.json runs/new-small-search --max-evaluations 8
```

先从离散变量生成合法程序和数学下界，不读取已完成实验成绩；MIP选提案，精确运行子问题按需更新该点成本；可能改善incumbent时执行数值验证。停止后报告声明空间内gap。示例下界较松，实际访问了全部8点，所以**本轮没有证明MIP节约了模拟次数**。

`best/` 含硬件、配置和JSONL；它是小型教学工作负载，不能用来领取五案例成绩。用正式五案例配置调用同一接口才能生成五个正式程序；`specs/five-bounded.json` 是这样的接口示例，但完整优化可能很慢，不能把它当成本轮已完成的实验。

## 官方评分衔接

完整五案例优化产生 `best/programs/{w-p,a-p,w-d1,w-d16,w-d4l}.jsonl` 后：

```sh
python3 ../phase-two-starter/tools/grade_v09.py runs/YOUR_FIVE_RUN/best/programs runs/final-grade --binary ../phase-two-starter/source/target/release/vnext-concurrent --seed 7 --jobs 2 --cpu-budget 4 --total-memory-gib 8 --timeout 600
```

保留原始grade、contract、worker报告；改程序后重评。多个种子仍只是数值测试，不是任意输入证明。

## 能力与限制

本轮五个正式参考程序均通过完整数值验证，预测与官方时序报告逐字段一致；详见 `reports/README.md`。独立模型与完整共享后端的验证范围分别记录，不能合并宣称完整独立证明。

- 硬件菜单和资源方程已落地；参考生成器支持的policy、并发、预取参数可作为离散变量。
- 任意JSONL程序可进入精确时序后端；模型自动综合程序的范围仍是官方生成器实现族。
- 计算部分的固定仲裁已有独立Python事件约束实现；包含通信的完整仲裁仍由共享运行子问题计算，不允许MILP自行创造不可执行的理想排程。
- 完整程序没有脱离运行核心的纯MILP等价编码；没有通用FP32/隐藏输入证明。
- source/host/RSS/timeout约束不由数学模型保证，完整官方验收仍必要。
- 当前不是完整作业要求的多智能体框架；这里实现的是供该框架使用的建模、求解、生成和验证模块。

参见 `MODEL.md` 中的精确覆盖表与最优性条件。不要把“有共享核心的零时序gap”“有限空间求解gap为零”和“真实全局最优”混为一谈。
