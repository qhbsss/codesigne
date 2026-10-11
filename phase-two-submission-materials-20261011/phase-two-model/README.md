# 第二阶段：可运行数学模型与精确运行子问题

聊天界面不显示 LaTeX 时，可用现代浏览器离线打开 `FORMULAS.html`，直接查看排版后的公式、变量表和求解流程图；无需外部脚本、字体或联网。

截至 2026-10-11 的最新五场景有限域 MIP **条件预测 3883.1222 分**；应用户明确要求，仍对所选**同一 SHA**五程序进行完整本地评分，seed 7 数值与资源均通过，真实本地得分 **3441.4747**，比预测低 **441.6475 分（11.37%）**，仍未达到 5000。原始评分见 `../phase-two-starter/runs/k4-rf-causal-rf-reuse-joint-grade-20261011/grade.json`，逐场景误差、方法局限见 `reports/k4-rf-causal-rf-reuse-joint-grade-20261011.md`；`FORMULAS.html` 的 1976 分数据表仅是较早的历史演算样例。没有服务端签名回执。

当前实现绑定原始 starter 的 `phase-two-compact-v2` / `phase-two-static-v1`，官方源码未修改。

这不是已经证明所有合法程序全局最优的模型。实现有三个明确不同的求解模式：

1. **纯数学内核 MIP**：独立 Python 面积、微步骤和单组时序方程计算候选成本，HiGHS 求解，不调用运行后端获取目标值。输出后才与官方核心对照。范围是有限的 compute-only GEMM 内核。
2. **完整程序 MIP + 精确运行子问题**：硬件/软件变量、面积和独立微步骤下界形成 MIP；程序生成后，共享的官方状态转移核心精确计算周期、流量和功耗。不是独立模拟器，也不是免运行的闭式预测。数值正确性另由完整官方 evaluator 检查。
3. **完整五场景独立估计搜索**：生成有限个参数组合，对静态容量合格的程序用独立完整时间近似排序，再把预测最好的候选交完整评分器验证。预测不是最优性下界，搜索只覆盖声明的候选域。

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
| `specs/five-reference-bounded.json` | 已评分参考硬件附近的4组合独立估计搜索；按场景关闭不兼容的SH直连 |
| `specs/five-rf64-single-group.json` | 对RF容量不足的方案使用单组并驻后重新生成并估计 |
| `specs/five-rf64-small-k.json`、`specs/five-rf64-k32.json` | 尝试缩小K分块保留双组并驻，供失败约束追踪 |
| `specs/k4-rf-*.json` | 5000分门槛搜索：已有五程序的硬件重配及按场景重新生成的软件策略 |
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

### 完整程序独立周期与成绩估计

```sh
python3 -m model.tool estimate ../phase-two-starter/examples/small-prefill.jsonl runs/new-estimate
python3 -m model.tool estimate-suite ../phase-two-starter/programs/reference runs/new-five-estimate
python3 -m model.tool estimate-suite ../phase-two-starter/programs/reference runs/new-five-check --grade ../phase-two-starter/runs/reference-model-grade-20261010-1200/grade.json
```

`estimate.json` 不调用官方时序核心：展开compact控制结构，按命令依赖、SM引擎、DMA前端、HBM通道和写入行冲突估算每个wave，逐wave累计；成绩是“**所有案例通过验证时**”的预测值。`--grade` 只在全部程序SHA256吻合时展示已测误差，不参与估计。该值有经验修正、**不是下界/最优性证书**，尤其Cache命中/合并、多播、请求仲裁等仍未独立建模。验收仍须用官方评分器。

按波次付费原因诊断，可用 `python3 -m model.stage_profile PROGRAM.jsonl runs/profile.json`。输出记录程序 SHA、同一独立估计器的总周期及按输出张量和指令类别拆分的周期；合计与 `model.tool estimate` 一致，仍**不是官方实测或可删除的独立算子时间**。本轮 `a-p/w-d1/w-d4l` 的独立诊断与数值验证范围见 `reports/causal-triangular-20261011.md`、`runs/stage-profile-20261011/`。

### 有限参数域的独立模型搜索

```sh
python3 -m model.tool search-estimate specs/five-reference-bounded.json ../phase-two-starter/runs/new-grid-search --timeout 600
python3 ../phase-two-starter/tools/grade_v09.py ../phase-two-starter/runs/new-grid-search/best/programs ../phase-two-starter/runs/new-grid-grade --binary ../phase-two-starter/source/target/release/vnext-concurrent --seed 7 --jobs 2 --cpu-budget 4 --total-memory-gib 8 --timeout 600
```

`search-estimate` 对声明的有限域逐点生成五场景程序并以独立公式计算预测成本，选择该域中**预测最优**的程序，输出 `search.json`、完整候选记录和 `best/programs/`，不调用官方时序评分。`specs/five-reference-bounded.json` 使用已评分参考程序的共享硬件作为起点，只有 `hardware.rf_kib∈{64,128}` 与 `policy.split_k∈{1,2}`，最优性结论**只覆盖这四个组合**；对 `w-p` / `a-p` 用逐场景 `overrides` 将 `policy.sh_direct` 设为 `false`，因为参考硬件关闭了SH-TC通道，否则导出的 `mma_shared` 指令不可运行。参数域中的字段覆盖逐场景设置。预测排名不等于实测排名，必须随后用评分器确认正确性与成绩。旧的 `specs/five-bounded.json` 从starter默认硬件出发，硬件与已评分参考程序不同，不应误当作参考附近的搜索域。

实际首轮 `rf_kib=64,parallel_groups=2` 程序超过每SM的并驻RF容量，评分器拒绝给分；初版估计只检查组数，漏掉每wave的RF/SH预留总和。现已在估计器中加上这项必要可行性检查，不合格候选会记录在 `rejections.json` 而不是参与最优排名。用 `specs/five-rf64-single-group.json` 验证 `parallel_groups=1` 的修正候选；`policy.k=64/32` 仍在与K分块无关的36 KiB/组vector wave触发双组容量超限。首轮留下的预测值是**无效候选的条件成本**，不得称为有效最优分数。

本轮七个候选中，通过必要容量检查的三个候选以 `rf_kib=128,parallel_groups=2,split_k=1` 预测最高（1588.5336条件分）；评分器重新运行同一五程序得到1548.9935分，预测偏高2.55%，程序与已评分参考程序哈希完全一致，并非新增成绩突破。完整候选对照及排除理由见 `reports/five-search-20261010.md`；无法据此推出完整程序全局最优或真实评分排名最优。

### 5000分预测门槛

```sh
python3 -m model.tool search-estimate specs/k4-rf-dma-search.json ../phase-two-starter/runs/YOUR_NEW_SEARCH --timeout 600
python3 -m model.tool grade-if-target ../phase-two-starter/runs/YOUR_NEW_SEARCH ../phase-two-starter/runs/YOUR_NEW_GRADE --target-score 5000 --timeout 600
```

五场景 `estimate-suite` 对现成 `k4-rf` 程序在本机耗时约247秒；对单个仅改变硬件的候选 `search-estimate` 通常约250–270秒，重新从config导出与压缩程序的耗时可能更长。搜索命令**只作独立预测**；`grade-if-target` 必须看到 `search.json` 的五场景条件预测不低于指定门槛，并核对选中程序SHA256，才运行完整评分。门槛未满足时输出错误、不创建评分目录。预测门槛不保证程序通过数值/功耗检验；只有评分器产生的结果才是实测分数。

`program_base` 模式只允许变更硬件字段：复制既有compact程序的命令体，更新五个程序的共享硬件头，并检查参考程序哈希、形状和面积。**它是具体静态程序的硬件重配，不承诺等于用新硬件config重新导出的程序。** DMA×4的 `w-p` 重导出实验确实生成了不同命令体；因此预计达到5000时应评分 `best/programs/` 中的**同一哈希**，而非换成重新导出的文件。当前搜索记录和未达门槛的原因见 `reports/target-5000-search-20261010.md`。

### 共享硬件、逐场景软件搜索

```sh
python3 -m model.tool search-estimate specs/k4-rf-casewise.json ../phase-two-starter/runs/YOUR_NEW_CASEWISE --timeout 600
```

设置 `domains` 为共享硬件字段（可为空），以 `case_domains` 为每个场景声明独立软件字段（`policy.*`、`parallel_groups` 或 `async_prefetch`）。每个硬件点各场景的候选只导出、压缩、估计一次；独立选取最小预测周期后组合五场景得分。输出 `domain_designs` 为笛卡尔积规模、`evaluated_designs` 为逐场景静态可行组合数、`estimated_programs` 为实际通过容量检查的单场景程序次数，`best/programs/` 和 SHA 可直接用于 `grade-if-target`。`case_domains` 不支持 `program_base` 的静态命令体重配；选择共享硬件时必须从正式 `export` 重新生成软件程序。只能证明声明的有限域内**条件预测最优**，不证明准确性、正确性、官方分数或所有可编程布局的最优性。实测记录见 `reports/casewise-search-20261010.md`。

正式实验枚举了32种五程序组合，仅生成和估计10个单场景选项，实测1127.68秒；选中程序全部与旧 `k4-rf` 同SHA，条件预测1976.2252分。未达到5000门槛，`grade-if-target` 已阻止启动完整评分器。本轮加速不等于得到更高的预测分或新的真实成绩。

### 预测与完整验证逐字段对照

```sh
python3 -m model.tool compare ../phase-two-starter/examples/small-prefill.jsonl runs/new-comparison --seed 7 --timeout 600
```

对照完整 `report`、header、step cycles、work units、HBM峰值和程序/源码hash，保留两份原始报告及 `comparison.json`。完整数值报告独立运行，但时序核心共享，不能把它包装成两个独立模拟器相互验证。

### 小型完整硬件—软件MIP

新增 `model-first` 将共享硬件与各场景软件选项**分别建为整数变量**，先对生成程序计算可证资源/依赖下界，求解HiGHS主问题，再对求解器提出的程序逐个运行官方时序与数值验证，并用真实周期更新主问题；只有搜索域全部定界才报告零gap。它不会为了找解而手动运行参数网格，若下界不足以剪枝会明确记录全部已验选项，不能宣称求解器加速。详见 `model/structured.py`、`MODEL.md` 第6节以及 `reports/model-first-20261010.md`。

```sh
python3 -m model.tool model-first specs/small-model-first.json runs/new-model-first --timeout 60 --max-verifications 10
```

`model-first` 在每个已准备的候选目录写入`prepared.json`（配置SHA、程序SHA、独立周期下界与耗时估计）。执行意外中断后，使用相同的spec、相同的输出路径并添加`--resume`，只复用配置与程序SHA均匹配的完整候选；未完成的中间输出会重新生成。完整`solution.json`已存在或spec变更时拒绝续跑，避免误将不同实验的结果拼接。

完整五场景可以用同一格式声明共享硬件菜单和逐场景 `case_domains`，并配置正数 `target_score`。执行顺序是导出与建模→共享硬件MIP求**可证分数上界**→仅当上界仍允许达标时独立预测筛查→所选程序完整数值验证→仅达到预测门槛时运行聚合评分。上界低于门槛时生成 `bound-screen.json` 并停止为 `below_proven_score_upper_bound_unverified`；其“不可达”仅对声明的生成器菜单成立，既不是成绩也不是全部合法ISA程序的上界。预测低于门槛时仍是**未验证、未证明最优**；仅预测过线才允许运行完整评分。

已经按正式五案例 `specs/k4-rf-casewise.json` 从零运行该流程：32种组合、10个单场景模型选项，最高条件预测1976.2252分、墙钟1552.27秒；5000门槛拒绝评分，无新实测分数或全局证书。模型可证下界对应的乐观最高分约8731.98，**不能从此推出真实5000不可达**。见 `reports/model-first-20261010.md`。

后续针对5000门槛的三组正式场景建模域均未触发真实评分：扩展prefill/decoder策略的32组合预测1928.2968（安全分数上界6146.2434）；四种共享硬件重配预测最高2143.5316（安全上界8793.3150）；两项双组并驻矩形prefill策略预测最高1976.2252（安全上界8573.9982）。截至上述实验的**最高只是有限域条件预测2143.5316，不是正式实测**；新硬件缩减HBM队列而独立模型尚未刻画准确队列信用，实际排名未知。完整原始目录、耗时、程序SHA和模型不足见 `reports/target-5000-model-first-20261010.md`。

`model-first` 另支持 `program_base` 配合 `hardware_choices`：列出几个完整的共享硬件变更字典，按源程序SHA固定五套已有compact命令体，仅重写硬件头，重新计算各候选的资源/依赖保守下界、并驻约束与独立耗时，交同一MIP选择。该模式禁止改变 `case_domains` 和场景覆盖；其程序**不是**按新硬件重新导出的命令体。`specs/k4-rf-replay-hardware-target-5000.json` 声明四种面积可行的硬件，其中包含先前预测最好的DMA2/SFU15组合；下界仅是声明的静态程序域的证书，不能由此声称软件或硬件全局最优。命令：

```sh
python3 -m model.tool model-first specs/k4-rf-replay-hardware-target-5000.json runs/k4-rf-replay-hardware-target-5000 --timeout 600 --max-verifications 32
```

软件决策也可用 `case_choices` 为各场景指定非笛卡尔的完整组合（不能与该场景的 `case_domains` 或 `program_base` 混用）。`specs/k4-rf-prefill-rectangular-target-5000.json` 保留原有 `m=32,n=64,parallel_groups=2`，并新增 `m=64,n=32,parallel_groups=2`，研究在保持双组并驻时增加权重B的跨行复用、但增加激活A访问的取舍；解题器自动选择两项prefill程序，decode沿用原配置。两项软件方案均由官方 `export` 生成新程序，仍需经过独立估计≥5000分的门槛，不能以新配置的理论潜力冒充成绩。

还可在导出模式下同时声明 `fusion_factors` 与 `case_choices`/`case_domains`：先按每个软件配置导出指令体，再为每个融合因子生成可检查的wave重排程序，最后以共享硬件及各场景的软件配置、融合因子的**联合笛卡尔积**求解。`max_designs` 对完整乘积计数；`prepared.json` 保存生成程序SHA与融合统计，续跑校验SHA。与仅重写硬件头的 `program_base` 模式不同，此模式的软件配置确实改变指令体；预测仍非FP32、功耗或真实评分证明。

已有不同静态调度指令体时，可在 `program_base` 模式下增设逐场景 `program_choices` 文件列表；未列出的场景仍从 `program_base/{case}.jsonl` 读取。MIP对每场景选择唯一 `program_choice`，与共享 `hardware_choices`、可选 `fusion_factors` 一起形成完整设计域；每个源程序必须与场景配置的形状、批次、decode和原硬件头完全一致。每项的源SHA和重配后的程序SHA分别保存在 `prepared.json`，`--resume` 拒绝源程序变化。这允许严谨地比较 `concat`、`pipeline` 与更大融合因子的**真实不同指令体**；正式示例见 `specs/k4-rf-tc-pipeline-hybrid-target-5000.json`，仍坚持聚合预测达到5000才启动完整评分。

该轮八项静态程序组合的MIP求解，`runs/k4-rf-tc-pipeline-hybrid-20261011/screen.json`的最高条件预测为**3144.8955分**，选中五份实际程序的SHA和预测周期均保留；`official_score=null`，仍不足5000，不能称为真实成绩。配套硬件排布与被淘汰的共享B尝试、双TC尝试见`reports/wave-fusion-model-20261010.md`；后续改进记录见`reports/causal-triangular-20261011.md`。旧程序曾有历史实测1948.8761分，不能用于推断这些新程序的真实得分。

新增的 RF 偏置/GELU/残差融合、复合波次拆分及同组 GEMM 面板复用生成**可执行的不同静态程序**，并送入同一共享硬件、逐场景软件 MIP：

```sh
python3 -m model.tool model-first specs/k4-rf-causal-rf-reuse-joint-target-5000.json runs/new-rf-reuse-joint --timeout 600 --max-verifications 0
```

`runs/k4-rf-causal-rf-reuse-joint-20261011/screen.json` 从六种五场景组合中选择最高条件预测 **3883.1222 分**；按预测五案等比再提速约 **1.2876 倍**才能到 5000。原搜索按当时的 5000 预测门槛停止，`solution.json` 原样保留 `below_prediction_target_unverified`、`official_score=null`，不回填事后成绩。用户随后明确要求独立完整验证选中的五份同 SHA 程序，`../phase-two-starter/runs/k4-rf-causal-rf-reuse-joint-grade-20261011/grade.json` 记录 seed 7 本地评分 **3441.4747 分**，全部数值/功耗/面积通过，实际五案等比提速需约 **1.4529 倍**才能到 5000。此前教学程序的同输出 SHA 与功耗检查不能取代这次五大程序的实测。安全下界给出的 `12513.8404` 是**乐观上界**而非预测，更不说明全局能达到该分数；详情见 `reports/causal-triangular-20261011.md` 与 `reports/k4-rf-causal-rf-reuse-joint-grade-20261011.md`。

```sh
python3 -m model.tool optimize specs/small.json runs/new-small-search --max-evaluations 8
```

先从离散变量生成合法程序和数学下界，不读取已完成实验成绩；MIP选提案，精确运行子问题按需更新该点成本；可能改善incumbent时执行数值验证。停止后报告声明空间内gap。`spec` 可设置 `"estimate_candidates": true` 用独立周期估计优先验证候选，但**证书仍只使用保守下界和官方实测值**；先计算全部候选的估计会增加运行时间。示例下界较松，实际访问了全部8点，所以**本轮没有证明MIP节约了模拟次数**。

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
