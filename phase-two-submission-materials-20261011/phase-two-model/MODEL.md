# 已落地模型：方程、状态接口与最优性范围

若阅读器将下面的公式显示为原始 LaTeX，请在浏览器中打开同目录 `FORMULAS.html` 查看可离线渲染的公式与流程图。

依据starter提交 `a6266ce`。本文件描述真实实现，而不是未展开的理想规格。

## 1. 两层模型及其边界

完整程序的设计变量是共享硬件 \(H\) 与各场景软件参数 \(\theta_c\)。\(G_c\) 是官方生成器，\(P_c=G_c(H,\theta_c)\) 是实际付费执行的静态 JSONL 程序。五场景集合为 \(\mathcal C=\{w\text{-}p,a\text{-}p,w\text{-}d1,w\text{-}d16,w\text{-}d4l\}\)，权重依次为 \((1/4,1/4,1/6,1/6,1/6)\)。完整程序的优化目标是：

\[
\min_{H,\{\theta_c\}}J(H,\theta)
=\sum_{c\in\mathcal C}w_c\ln T_c\bigl(H,G_c(H,\theta_c)\bigr),
\qquad H\in\mathcal H_{23},\quad H_c=H\quad(\forall c).
\]

固定参考周期 \(T_c^{\mathrm{ref}}=(187689274,374856358,27068433,42358913,70439292)\) 时，官方成绩为

\[
S=1000\exp\!\left(\sum_c w_c\ln\frac{T_c^{\mathrm{ref}}}{T_c}\right).
\]

所有门槛都通过的设计中，最大化 \(S\) 才等价于最小化 \(J\)。对每个正式场景和测试种子 \(\sigma\)，至少需要满足：

\[
\begin{aligned}
A(H)&\le100\;\mathrm{AU},&\quad M_c^{\mathrm{peak}}&\le2\;\mathrm{GiB},&
M_c^{\mathrm{cum}}&\le64\;\mathrm{GiB},\\
P_{c,100}&\le34\;\mathrm W,& P_{c,10000}&\le26\;\mathrm W,&
T_c&\le8\cdot10^9,\\
\left|y_{c,j}^{\sigma}-\hat y_{c,j}^{\sigma}\right|
&\le10^{-3}\!\left(1+\left|\hat y_{c,j}^{\sigma}\right|\right)
&&\text{对每个最终 hidden 和逐层新 K/V 元素 }j.
\end{aligned}
\]

这里 \(\hat y\) 是独立 FP64 参考结果舍入后的 FP32 值；还须满足静态语法、形状、容量、记录/指令/宿主工作量以及每 wave 周期门槛。\(T\)、功耗、流量由共享官方状态转移核心计算；FP32 数值和 reference 对照必须由完整 `evaluate` 检查。公开种子的通过结果不等于未知输入上的证明。

每个 wave 的组 \(g\) 固定到 SM \(s_g\)，并驻的静态容量还必须满足

\[
\forall w,s:\quad \sum_{g\in G_{w,s}}r_g\le H_{\mathrm{rf\_kib}},\quad
\sum_{g\in G_{w,s}}h_g\le H_{\mathrm{sh\_kib}},\quad
|G_{w,s}|\le H_{\mathrm{resident\_groups}}.
\]

其中 \(r_g,h_g\) 是导出程序中每组实际申请的 RF/SH KiB，不能只检查配置中的 `parallel_groups` 是否不超过 `resident_groups`。独立估计器现已逐wave检查这些必要条件，未通过者不能计入候选得分；其他合法性、功耗和数值门槛仍须真实评分器确认。

**精确的独立数学方程**覆盖单组compute-only wave；此外已有一个**完整程序的独立近似周期估计**，扫描compact/expanded程序的全部wave，不调用官方时序核心。后者包含DMA/HBM/NoC的主要时延与争用，但未模拟所有请求、Cache/多播、功耗、数值结果；它是搜索排序的预测值，既非精确时序，也非可证明下界。

## 2. 硬件有限选择与面积

每字段one-hot x_{j,k}，sum_k x=1。菜单和全部组合规则由 `hardware.py` 编码：TC开关/形状/Kp、SH开关/bank/interface、sfu≤vector、面积≤100。每个场景使用同一H。

单SM面积：

\[
\begin{aligned}
a_{SM}={}&.45+.018V+.06U+c[.072+.008(pq+p+q)k_p]+.018Rf_t\\
&+.004S+.015b+(u-1)(.0015S+.008b)+I_{B_t>0}(.28+.004B_t)\\
&+r(.10+.012V)+.514+d[.10+.006(4+2z)]\\
&+.256+.08(G-1)+.002zG+.15(B_s/64)^{1.3}+.04m.
\end{aligned}
\]

\[
A=Na_{SM}+I_C(3C+.6+.012\cdot256+.006\cdot64)
+ .65h+.0015hQ+1+.035N+.8(B_n/128)^{1.3}+.2m+2\le100.
\]

P0=.025A+.15h，面积mm²=.25A。浮点面积方程与源码代数等价，但运算顺序有最后若干bit差异；独立测试按1e-11 AU容差比较。极贴近100AU的边界仍以官方validate为准。

## 3. 独立微步骤

每指令产生 (RF读bytes、SH读service、SH→TC bytes、core duration/energy、RF写bytes、SH写service)。Python没有调用官方profile。

- Fill：按V切片，仅写RF。
- Vector：按V或U切片；immediate不读RF，stride0广播读4B，FMA额外读old dst；core分别4/12周期。
- Reduce：相邻二叉树，每层按V切片；专用core1周期，否则Vector4周期；尾部copy和实际merge数保留。
- MMA：逐p×q物理输出块，先读accumulator；逐Kp块读A/B、core=1+log2(Kp)，最后drain=p+q-2和写回；padding照付能量，不越界读。
- MMA.SH：B的实际word地址决定16元素批内bank conflict，付SH→TC接口。
- SH copy：每16word批、startup和地址生成、读广播/写bank冲突分别计算。

官方菜单允许参数不等于所有组合都数值通过；模板正确性须由完整evaluate检查。

## 4. 单组精确时间方程

给定一组固定顺序指令，没有DMA或其他组争用。每条指令issue付1周期；每个微步骤依次read→core→write。

RF service=ceil(bytes/BW)，依赖latency=2+floor(log2(max(R/32,1)))。SH latency=6+floor(log2(max(S/(4b),1)))。read阶段延迟为RF与SH路径取max；SH→TC路径另外串接interface service和2周期。write阶段取RF/SH路径max。

\[
\begin{aligned}
\ell_{RF}&=2+\lfloor\log_2\max(R/32,1)\rfloor,&
\ell_{SH}&=6+\lfloor\log_2\max(S/(4b),1)\rfloor,\\
D_{RF}^{r}(v)&=\mathbf1_{v>0}(\lceil v/B_R\rceil+\ell_{RF}),&
D_{SH}^{r}(v)&=\mathbf1_{v>0}(\nu_r(v)+\ell_{SH}),\\
D_{TC\leftarrow SH}(v)&=\mathbf1_{v>0}
\left(\nu_r(v)+\ell_{SH}+\lceil v/B_t\rceil+2\right),\\
D_{read}&=\max\{D_{RF}^{r},D_{SH}^{r},D_{TC\leftarrow SH}\},&
D_{write}&=\max\{D_{RF}^{w},D_{SH}^{w}\},\\
D_{step}&=D_{read}+D_{core}+D_{write},&
T_{\mathrm{serial\ wave}}&=1+\sum_i\left(1+\sum_{s\in i}D_{step,s}\right).
\end{aligned}
\]

\(B_R,B_W\) 分别为该RF tier的读/写端口带宽，\(\nu_r,\nu_w\) 为实际地址与bank冲突决定的服务周期；写路径采用相应写带宽、bank周期和依赖延迟。只有存在SH→TC读的步骤才使用 \(B_t\)。零bytes/零core不额外付phase周期；完成时可在同一时间issue下一指令。最后wave fence付1周期、每SM8pJ。无跨组竞争下这些是精确方程，不是roofline。

能源事件实际发生于service区间；RF整带宽部分和尾部单独计费，SH每个bank service周期逐项计费，不能把能量摊到依赖等待。

\[
P_W=P0+\max_\tau\sum_e \frac{E_e}{d_e}
|[s_e,s_e+d_e)\cap[\tau-W,\tau)|/(2000W).
\]

检查W=100/10000，限34/26W，包含idle tail。独立测试对周期/bytes/FMA用严格相等，对energy/power用浮点容差。

## 5. 完整程序精确状态接口

计算子集另有独立 `compute.py`：多组、多SM、多个TC/SFU/Vector/Reduction/Copy引擎，共享RF/SH/interface端口，组内phase阻塞和固定round扫描、next-event跳时全部显式编码。它不调用官方Scheduler。与源码对照时不仅cycles/bytes/FMA相同，scheduler iterations也严格相同，energy/power按浮点容差匹配。它明确拒绝DMA和wait，不作完整程序替代。

`runtime`是单独Rust crate，通过path dependency链接未修改的starter：

- 官方compact Reader负责静态repeat/template/address表达式与语法。
- 本层按实际allocation栈计算256B对齐基址、input长度和release/cache失效。
- 官方Scheduler负责组容量、token/wait、指令形状、固定仲裁、微phase、DMA和wave fence。
- 官方Memory负责64B请求、HBM250-cycle readiness、channel FIFO、credits至local ack、Cache四路LRU/MSHR、NoC/multicast和event ID。
- 官方Power负责100/10000周期窗口和float64累计。

完整程序也可以写成**隐式状态模型**。令 \(z_k\) 包括各组PC/ready/token、SM引擎与端口、DMA队列、HBM请求与credit、Cache标签/LRU/MSHR、NoC及功耗事件。官方调度和内存核心按下一个事件推进：

\[
(z_{k+1},\Delta t_k,\mathcal E_k)=F_H(z_k,P_c),\qquad
T_c=\sum_k\Delta t_k,\qquad
E_c=\sum_k\mathcal E_k+P_0(H)\cdot(2\cdot10^{-9}T_c)\;\mathrm s.
\]

其中 \(\mathcal E_k\) 以 J 计，\(P_0(H)=0.025A(H)+0.15h\) W。此式是共享模拟器实际执行的状态转移**接口**，不是已展开为MILP的独立方程；离散Cache命中、合并、FIFO等待及端口争用取决于之前事件次序，不能给每个候选简单代入一条固定带宽公式。

commit检查输出shape/view与当前input绑定，然后付空wave fence。**预测模式假设commit数值正确，不检查输出值**，所以错误程序也可能得到时间报告。这是刻意标明的conditional timing prediction。

本层还调用官方非functional静态检查，覆盖跨组HBM race、局部alias、shape等不能只由Scheduler检查的规则。为此分配NaN占位张量，但不生成fixture、执行FP32计算或计算reference。HBM峰值是模拟地址峰值而不是主机RSS。没有独立初始化证明、通用浮点约束、参考正确性或主机超时保证；这些由完整官方evaluate报告确认。

### 5.1 完整程序的独立时间估计（非精确后端）

`model/estimate.py` 独立逐条展开compact模板、重复、仿射整数表达式及命令；对每个wave以group PC和token排定计算/访存指令，在每个SM的issue端口与相应计算/DMA引擎上顺序预留。计算指令的单条时长使用第3、4节独立微步骤方程。对于DMA指令，16-word地址批内按实际64B行去重，记总行数为 \(N_d\)、HBM通道 \(h\) 上的行数为 \(N_{d,h}\)、元素数为 \(E_d\)，HBM就绪与通道预留近似为：

\[
A_d=t_d+1+6+250+\mathbf1_{\mathrm{store}}(\ell_{\mathrm{local}}+1),\qquad
F_h\leftarrow\max(F_h,A_d)+2N_{d,h}.
\]

\(t_d\) 为满足前序PC、SM issue端口和DMA引擎空闲的发射时间；\(F_h\) 是按发射顺序聚合预留的通道完成时间。若同一个**写**指令的相邻16-word批共享64B行，数目为 \(b_d\)，则同一地址未ack之前写请求会阻塞：

\[
\begin{aligned}
R_d&=\max\!\left\{\lceil E_d/16\rceil,\lceil N_d/2\rceil,
\left\lceil\frac{(72+8\mathbf1_{\rm store})N_d}{B_{\rm sm}}\right\rceil,
\left\lceil\frac{(72+8\mathbf1_{\rm store})N_d}{B_{\rm global}}\right\rceil,
\left\lceil\frac{4E_d}{B_{\rm local}}\right\rceil\right\},\\
\hat C_d&=\max\{\max_h F_h+7+\ell_{\rm local},\ A_d+R_d\}
+270\,b_d\,\mathbf1_{\mathrm{store}}.
\end{aligned}
\]

读写方向实际包长分别为72B、80B；上式只按总字节数估算端点/全局带宽，不拆分NoC上/下行。\(B_{\rm local}\) 是以RF带宽代理RF/SH局部服务的近似值，等候token的组必须等对应DMA完成才能继续。记每个wave聚合调度的最晚组完成时间为 \(C_w\)；对于无相邻批重复写行且DMA很多（\(D_w\ge128\)）的wave，为避免对长指令“整条占用通道”的过度串行，采用流体吞吐近似：

\[
\hat C_w=\min\{C_w,\max(2\max_h\sum_{d\in w}N_{d,h}
+310\lceil D_w/N_{\rm SM}\rceil,\ \max_s C^{\rm compute}_{w,s})\}.
\]

若无上述特殊情形，\(\hat C_w=C_w\)。对不同组写同一行，用冲突图（相同tensor和行所涉及的组）计算最大写者数 \(m_w\)、相邻写者去重边数 \(e_w\) 及每SM最多并驻组数 \(g_w\)，增加 \(K_w=\min\{d,g_w\}\max\{260(m_w-1),\lceil260e_w/4\rceil\}\)；这里 \(d\) 为DMA引擎数，多引擎并发时同一SM的行锁等待也可能扩大。从而独立完整周期/成绩估计为：

\[
\hat T_c=n_{\rm commit}+\sum_{w\in c}
\bigl[\hat C_w+K_w+1+20\mathbf1_{\rm DMA(w)}\bigr],\qquad
\hat S=1000\exp\!\left(\sum_c w_c\ln(T_c^{\rm ref}/\hat T_c)\right)
\quad\text{（仅在全部案例eligible时）}.
\]

HBM的250/2、16-word、每SM每周期2次提交是题目时序参数；270/260/310、并发冲突放大系数和每wave附加20是对有限样本争用的**经验校正**，不能写成物理定律或严格上/下界。长DMA的通道预留、缓存的命中/合并/驱逐、channel FIFO与credit、NoC逐包竞争、功耗与正确性均未完全展开。估计在不同硬件、容量或Cache组合上可能明显失真；绝不把 \(\hat S\) 当评分器结果或不经实测的最优解证明。

## 6. 有限域MIP与精确运行子问题

对声明的有限离散域 \(D=\prod_j D_j\)，每个 \(d\) 都对应共享硬件和各场景的一套软件参数；用tuple lifting引入二元变量 \(q_d\)，每字段取值引入 \(x_{j,k}\)：

\[
q_d\in\{0,1\},\quad \sum_d q_d=1,\quad
x_{j,k}=\sum_{d:d_j=k}q_d,\quad
\sum_d A(H_d)q_d\le100.
\]

参数tuple不是已有成绩表：先用生成器导出新程序，从付费指令计算保守下界，然后才调用运行核心测该提案。可选 `"estimate_candidates": true` 为全部候选计算 \(\hat T\)，优先验证预测最好的、且下界仍可能改善incumbent的候选；MIP的合法界/最终停止仍只根据下式的保守界和已验成本，不将经验估计用于数学剪枝。硬件面积和配置规则在生成前独立校验；正式程序的语义、功耗及数值正确性在选择候选后校验。

初始下界按**每个wave和固定SM归属**累计：每个SM的mandatory command issue、各计算引擎core占用（除以该SM该引擎数量）、以及所有非DMA计算指令的RF读写字节（除以该SM端口带宽）。这些资源允许互相重叠，所以逐wave取最大值；wave完成后下一wave才能开始，故逐wave求和并计入各自fence和commit。遗漏DMA的RF流量、HBM/NoC、SH服务、操作数延迟与依赖等待，因此仍只是保守lower bound。

\[
L_c(d)=\max\left\{1,\ n_{commit}+\sum_{w}\left[1+\max_{s}\left\{
I_{w,s},\ \left\lceil\frac{W_{w,s,e}}{C_e}\right\rceil\ (e\in E),\
\left\lceil\frac{R_{w,s}}{B_R}\right\rceil,\
\left\lceil\frac{D_{w,s}}{B_W}\right\rceil\right\}\right]\right\}.
\]

其中各组固定到SM \(s\)，\(I\) 是该SM命令数，\(W\) 是引擎core周期数，\(R,D\) 是付费计算指令的RF读、写字节，\(B_R,B_W\) 是对应端口带宽。专用reduction关闭时工作已计入Vector，不再除以r。主问题：

\[
\min_{q,x}\sum_d q_d\,b_d^{(k)},\qquad
b_d^{(0)}=\sum_cw_c\ln L_c(d),\qquad
b_d^{(k+1)}=\begin{cases}
\sum_cw_c\ln T_c(d),&d\text{ 已完成精确时序评估};\\
b_d^{(k)},&\text{其他候选}.
\end{cases}
\]

\(L_c(d)\) 和已测的 \(T_c(d)\) 是主问题的**常数系数**，没有在MILP里以连续变量拟合 \(\ln T\)。选择 \(d\) 后运行精确时序子问题；若功耗失败或确认数值不正确，禁用 \(q_d\)。仅在候选可能成为最优已验解时运行完整 `evaluate`。其他host失败/timeout不会作为数学不可行剪枝。

设 \(U_k\) 为已通过完整验收候选的最小 \(J\)，\(B_k\) 为主问题的合法下界，则在声明域和已检测正确性条件下 \(B_k\le J_D^\star\le U_k\)。若 \(B_k\ge U_k-10^{-9}\)，在该数值容差内停止；预算耗尽/solver未完成时保留未决 \(U_k-B_k\)。这是有限域精确点修正，不是已实现高效通用Benders cut生成器。可能访问全部设计；本轮8点实验确实如此，不宣称搜索加速。

五案例目标与官方加权几何平均等价；纯kernel直接最小化整数cycles。log浮点比较的1e-9容差及solver数值容差限定了证书强度，不是精确有理全局证明。

## 7. 生成、验证和身份绑定

MIP变量映射到真实config字段→官方export→compact JSONL。五案例shared hardware先检查，生成后再由官方grade contract验证。指令与真实分配/缓冲由生成器产生，不能只改变header而保留旧程序来假装软件适配。

预测前后检查程序SHA256；完整evaluate必须与预测的official source hash和程序hash相同。比较整个report（含所有stats/memory字段）、header、step cycles、work units和HBM峰值。数值comparison/output hash来自完整官方执行，不能由时序后端编造。

JSONL生成目前只覆盖官方generator实现族；可以对外部合法JSONL作预测，但不能自动综合任意融合/布局/在线attention程序。

## 8. 与官方模拟器的差距：可测量范围

定义解析下界的相对松弛 \(g_L(c)=1-L_c/T_c\)，独立近似估计的相对误差 \(e_c=\hat T_c/T_c-1\)，共享状态后端的差值 \(g_T(c)=T_c^{\rm shared}-T_c\)。三者针对不同模型，不能混为一个“预测误差”。

| 测试范围 | 现有证据 | 能支持的结论 |
|---|---|---|
| 小型完整程序，`examples/small-config.json` export，seed 7 | 逐wave下界597周期；独立估计23448；完整`evaluate`为23658周期；程序SHA256 `91280c717ef0101db5266c3f8f7ae2917e0ce41c2389f20ed1d1fad505f143bd` | \(g_L=97.48\%\)，实测约为下界的39.63倍；独立估计 \(e=-0.89\%\)。下界只用于剪枝，估计另用于排序。 |
| 五个正式参考程序，attention_stress seed 7 | 共享时序后端与完整官方评估逐字段相同、\(g_T=0\)，均通过数值检查；原始报告见 `reports/reference/` | 仅证明这些程序在相同源码状态转移核心下的对照；并非独立公式在五案例上预测零误差。 |
| 小型完整空间8个组合 | 访问8/8个候选，有限域log gap=0，最优23616周期；见 `reports/small-mip-solution.json` | 限定域内的收敛依赖运行子问题穷举，本轮尚无求解器节省评估次数的证据。 |

`reports/independent-reference-final-20261010/estimate.json` 为**独立时间模型**对相同五个正式参考程序的周期/成绩估计，与另外一次真实聚合评分 `phase-two-starter/runs/reference-model-grade-20261010-1200/grade.json` 按程序hash对照：

| 案例 | 独立估计周期 \(\hat T\) | 评分器周期 \(T\) | 相对误差 \(e\) |
|---|---:|---:|---:|
| w-p | 87966137 | 91208716 | -3.56% |
| a-p | 245330677 | 249709295 | -1.75% |
| w-d1 | 18880252 | 19698605 | -4.15% |
| w-d16 | 27797676 | 28034378 | -0.84% |
| w-d4l | 56374620 | 57484249 | -1.93% |

仅在这些程序均通过全部门槛的条件下，公式预测得分 **1588.5336**；该次真实本地评分为 **1548.9935**，预测偏高 **2.55%**。五案例中曾抽样小波形检查并调过冲突补偿，因此上表是校准样本的整程序对照，**不是五个独立留出测试**。另取小程序八个硬件/软件组合，预测与历史实测排序一致，单例相对误差-0.89%~-1.06%；更换小程序硬件并重新官方export/evaluate的独立留出样例，HBM通道4/SM4/通道2/DMA×2误差分别为-0.88%/-0.85%/-0.75%/-0.91%，开启Cache+多播后的误差为-5.34%。另外，正式 `w-d1` 的三种变硬件程序测量见 `reports/independent-holdout-20261010.json`：HBM4/NoC256估计26928180、实际28761282（-6.37%）；DMA×2/NoC256在**多引擎冲突修正后**估计27168468、实际27561795（-1.43%，这个配置参与了修正，非留出）；组合HBM4+DMA×2估计27913628、实际28962638（-3.62%，组合没有再次调参）。这些检验不足以保证任意候选的误差上限或最优候选不变。

一次正式五场景的**独立估计选解与随后重新评分**见 `reports/five-search-20261010.md`：原始7个待检组合中，初版漏建模并驻组RF和而误选的 `rf_kib=64,parallel_groups=2` 获得1657.4162的无效条件预测，真实评分器拒绝给分；加入第1节所述容量约束后，仅三个设计通过必要静态检查，选择的128 KiB方案预测1588.5336、实测1548.9935。它的程序哈希与上述参考方案完全一致，所以这个结果揭示了有限域的建模缺口和误差，没有找到比已知参考程序更高的有效分数。

继续扩大有限域的筛选见 `reports/target-5000-search-20261010.md`：沿已有 `k4-rf` 五程序的硬件重配及重新生成的软件策略，该轮独立估计的最高**条件预测**2142.4908；仍未达到5000门槛，因此没有新方案的正式评分结果。单套现成五程序估计约247秒，带完整重新导出时更久。该结果既不证明模型不能预测5000，也不证明真实最优低于5000；不同硬件可能改变生成器的程序体，所以硬件重配估计只绑定那份具体JSONL。当前独立模型还忽略HBM队列信用等状态，若后续候选预测跨过门槛，必须对同一SHA做完整官方检查。

搜索器现可对共享硬件与逐场景软件参数做**无损的预测目标分解**。记共享硬件为 \(h\in\mathcal H\)，各场景独立软件方案为 \(a_c\in\mathcal A_c(h)\)，实际导出的静态程序为 \(P_c(h,a_c)\)，独立估计周期为 \(\hat T_c(P_c,h)\)。因为固定硬件后的五个程序不共享指令调度或运行状态，且分数的对数为可加权和：

\[
\min_{h,\,a_1,\dots,a_5}\sum_c w_c\log\hat T_c(h,a_c)
=\min_{h\in\mathcal H}\sum_c w_c\log\!\left(\min_{a_c\in\mathcal A_c(h)}\hat T_c(h,a_c)\right).
\]

在每个候选均按官方生成器生成、通过现有必要静态容量筛选的前提下，右式与逐个枚举 \(|\mathcal H|\prod_c|\mathcal A_c(h)|\) 个完整组合具有**相同的有限域预测最优解**，只需 \(\sum_h\sum_c|\mathcal A_c(h)|\) 次单场景估计。并没有因加速搜索就修复模型遗漏的队列、功耗和浮点正确性，也未提供未声明的布局/融合/调度算法；独立估计对新程序的误差不能用已验证参考样本的2.55%代替。5000门槛仍只对预测同SHA程序启动正式评分。

`reports/casewise-search-20261010.md` 实测固定共享硬件下32个五场景软件组合只需10个单场景导出、压缩与估计，墙钟1127.68秒；有限域最优仍是原 `k4-rf` 五程序（同SHA），预测1976.2252分。门槛拒绝执行完整评分。这进一步定位了剩余高分障碍：需要超出现有 `m/split_k` 和既有程序的候选生成能力，不能把有限域的最优值解释成数学模型或真实运行的全局上限。

五个参考案例在 `reports/five-reference-summary.json` 中保留程序hash、源码hash及逐字段差分。它们是在后来补充非数值静态检查**之前**采集的；补充检查后没有重跑那套时序子报告。随后另跑过一次完整五案例本地聚合 `grade.json`（上述评分），但没有服务器成绩；不得将旧prediction记录误认为与新验收来自相同适配器源码hash。

| 层次 | 现在的覆盖 | 尚不能宣称或缺少的模型 |
|---|---|---|
| 硬件与计算 | 23字段面积、微步骤、无DMA的单组和多组计算有独立方程/事件模型；按抽样硬件与官方对照 | 浮点面积边界位级一致性、全部指令与所有参数的穷尽证明；零误差只对已测子集有效 |
| 完整通信 | 独立流式近似含DMA就绪/通道吞吐和写行争用；Cache/LRU/MSHR、HBM ready/credits、DMA队列、NoC、多播仍由共享官方核心精确运行 | 任意合法程序准确的独立通信状态约束、强于当前下界的可证安全资源界；Cache命中及请求合并取决于事件时序，无法从候选容量直接得到固定流量 |
| 功耗与容量 | 单组功耗有独立滑窗公式；完整程序功耗、2 GiB HBM峰值由官方后端报告 | 不经完整运行的功耗/峰值可行性证明；高功耗失败设计的界仍需验证 |
| 数值与程序 | 静态检查和公开种子完整 `evaluate`，结果绑定程序和源码hash | 不做FP32逐值传播与任意输入证明；自动生成仅覆盖官方generator实现族，未枚举全部合法静态算法 |
| 搜索与验收 | 独立估计已用于小型五案例有限域选解并随后运行完整评分；有限域MIP/精确点修正仍用严格下界与实测更新已验证解上下界 | 全部合法ISA和硬件的全局最优证明、足够紧的下界或强割、其余有效候选的真实排名及服务器回执；超时/RSS也没有数学保证 |

若“真实情况”指教学模拟器，完整时序接口对已测程序可逐字段复现；独立近似估计已从39.63倍的下界差距改善为上述小样本误差，但候选域的排序一致不构成最优性定理：若误差大小随硬件变化，预计最优和真正最优可能不一致。理论上可以展开有限状态机，当前实现仍无可求解的全程序纯MILP。若指真实芯片，本题给定的500 MHz、250周期HBM就绪时间、面积与能量系数均为教学假设，且不含物理工艺、温度、实际访存分布和系统负载；没有硅上测量与校准数据，无法推出真实设备延迟或功耗误差。

## 9. 模型先行的共享硬件—场景选择主问题

`model/structured.py` 把声明域按共享硬件 \(h\) 和各场景独立软件方案 \(a\) 因子化，令 \(x_h,y_{hca}\in\{0,1\}\)。对每个候选先用官方 `export` 得到**待优化程序**，再从其付费指令提取数学系数；此时尚未进行时序估计或评分。主问题为：

共享硬件还可由 `hardware_choices` 声明非笛卡尔的离散菜单；`program_base` 则固定已经生成的五套compact命令体，只对菜单中的硬件重写程序头部，再按同一约束重新计算下界、估计和MIP目标。后一模式不接受 `case_domains`，也不允许把重配的程序当成用新硬件重新 `export` 的程序；下述 \(a\) 在该模式每场景恰好只有一个静态选择。其可证范围为这些**具体程序体**与已声明硬件的组合，不涵盖其他软件排布。

`case_choices` 可为同一场景显式给出联动的软件变量字典（例如 `m=64,n=32,parallel_groups=2`），防止笛卡尔积混入无法满足RF并驻的 `m=64,n=64,parallel_groups=2`。这些字典由生成器各自产生独立程序；MIP中的 \(y_{hca}\) 仍在各场景选且仅选一个，而非根据已经完成的评分实验挑选方案。未列出的软件组合不属于证书范围。

\[
\min\sum_{h,c,a}w_c\ln L_{hca}\,y_{hca},\quad
\sum_h x_h=1,\quad \sum_a y_{hca}=x_h\ (\forall h,c),\quad
\sum_h A(h)x_h\le100.
\]

已证实功耗/数值不合格的选项置 \(y_{hca}=0\)；候选通过完整 `evaluate` 后把对应的 \(\ln L_{hca}\) 替换为 \(\ln T_{hca}\)，并重新求解。每步MIP的对偶界对声明的**静态可行且未发现违规**的选项是下界，完整验证且满足约束的方案成本是上界；只有二者收敛才报告有限域的log gap 0。未经验证的异常/超时不得当作违规剪枝。最优性不适用于未声明的软件布局、隐藏输入或新程序族。

这里 \(L_{hca}\) 是资源下界与新增**组内依赖关键路径下界**的最大值。对非DMA付费指令的每个步骤 \(s\)，定义最低串行时长 \(d_s=\mathrm{core}_s+\lceil R_s/B_R\rceil+\lceil W_s/B_W\rceil\)；无Cache且非空的HBM读取最低要经过250周期ready，其他DMA取零以保持保守。组内同步命令累加最低时长；异步命令只生成最早完成token，显式wait取token与组PC就绪时间的最大值。对每个wave取组内最大关键路径，再加fence和commit，与第6节资源下界取最大。此式忽略端口争用、Cache、写入耗时及其他正延迟，故**只能作下界，不能直接预测分数**；组内依赖方式对应静态程序而非虚构理想融合。参考小程序的资源下界597周期，加入依赖后约11437–11488周期，真实约23616–24152周期；这改进了证明/提案的强度，但仍不足以保证剪枝。

正式五案例配置 `target_score` 时，所有**模型系数建好后**先求安全成本下界 \(B=\min_{x,y}\sum_{h,c,a}w_c\ln L_{hca}y_{hca}\)。对应的乐观成绩上界为 \(S_{\rm upper}=1000\exp(\sum_c w_c\ln T_c^{\rm ref}-B)\)。若此值严格低于目标，则在声明的静态程序生成域中，即使全部选项都达到各自安全周期下界也无法达标；写入 `bound-screen.json`，终止为 `below_proven_score_upper_bound_unverified`，不消耗独立估计或评分。数值比较保留 \(10^{-9}\) log 单位余量防止浮点边界误拒。**这个有限域的数学上界不限制尚未生成的程序或其他硬件。**

只有该上界尚不能排除目标时，才用独立估计器给每个选项作条件预测，由同一共享硬件MIP求出声明域的最高预测成绩。未达到门槛则停止为 `below_prediction_target_unverified`，**不计算真实最优性gap或启动评分器**；达到门槛时才允许对求解器所选同SHA程序作完整验证/评分。估计不是合法剪枝界，故若MIP选择的当前提案预测不足门槛也必须停止并报告“未证明最优”，而不能删除该方案以伪造证书。

`reports/model-first-20261010.md`记录了正式五场景32组合从头建立严格数学下界、用MILP按各场景独立预测解出条件最高1976.2252分及低于5000而停止的完整运行（1552.27秒）；保守下界本身仍允许约8731.98分的乐观分数上界。此结果证明了**限定域内估计成本的最优组合**，未证明估计误差上限或真实5000不可达。当时计算剖析按完整指令内容缓存，后来按不影响微步骤的指令形状复用缓存，均未改变估计系数；若要进一步缩小真实最优性gap，仍需纳入安全的DMA/NoC/端口/功耗约束及真实可生成的新软件决策变量，而不是把估计值当合法定界。

正式融合软件与矩阵阵列排布的早期有限域见 `reports/wave-fusion-model-20261010.md`：软件选择现在可以是**不同的可执行JSONL指令体**，并可与导出策略及融合因子构成笛卡尔积。设每场景候选索引`a`确定一个输入程序SHA，`f,m`表示融合因子和模式，实际程序`P(h,c,a,f,m)`必须存在且通过容量检查；共享硬件变量`x_h`和选项变量`y_(h,c,a,f,m)`满足`Σ_h x_h=1`、`Σ_(a,f,m)y_(h,c,a,f,m)=x_h`、`Σ_h area(h)x_h≤100`。目标是`min Σ_c w_c Σ_(h,a,f,m)y_(h,c,a,f,m) log T_hat(P(h,c,a,f,m))`，目标值映射成**条件预测分**，不是实测。当时该特定小域最高预测**3144.8955分**、相应独立保守下界的乐观上界11450.8547分，均不代表目前新程序的最佳解。

后续静态程序的 bias/GELU/residual 融合、跨wave排程与组内 B 面板复用各自产生实际 JSONL；把完整组合后的每场景指令体当作独立软件选项 `a∈A_c(h)`，而不把小样的加速比例写成新目标系数。`specs/k4-rf-causal-rf-reuse-joint-target-5000.json` 在一套面积99.5543 AU的共享硬件和五场景8项实际程序上构造 `6` 种五案组合，按相同的 `min Σ_(h,c,a) w_c y_(h,c,a) ln T̂_(h,c,a)` 求得 **3883.1222 分条件预测最优**；安全下界给出的乐观上界是 **12513.8404 分**，两者不能混称。`runs/k4-rf-causal-rf-reuse-joint-20261011/` 保留所选五个程序 SHA 和每个估计周期，`solution.json` 为 `official_score=null`：预测未到5000，严格不运行完整五案评分器。模型并未证明所有合法硬件/程序全局不可达5000，也没有解决Cache/队列/数值功能与峰值功耗的建模缺口；上述条件预测也尚无同SHA五案的真实误差界。
