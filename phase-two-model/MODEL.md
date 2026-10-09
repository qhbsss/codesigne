# 已落地模型：方程、状态接口与最优性范围

依据starter提交 `a6266ce`。本文件描述真实实现，而不是未展开的理想规格。

## 1. 两层模型及其边界

完整程序模型：

\[
\min_{H,\theta_c}\sum_c w_c\log T_c(H,P_c(\theta_c))
\]

H为共享23字段硬件，theta为各案例的软件参数，P为官方数据无关生成器导出的静态程序。T、功耗和流量由共享官方状态转移关系计算；初始化、FP32和reference正确性由另一次完整evaluate检查。

纯数学模型只对**单组compute-only wave**展开时间与能量方程：不包含HBM、跨组竞争、Cache或NoC。对于这个子集，不需要运行后端来产生优化目标。

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
D_{step}=D_{read}+D_{core}+D_{write}.
\]

零bytes/零core不额外付phase周期；完成时可在同一时间issue下一指令。最后wave fence付1周期、每SM8pJ。无跨组竞争下这些是精确方程，不是roofline。

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

commit检查输出shape/view与当前input绑定，然后付空wave fence。**预测模式假设commit数值正确，不检查输出值**，所以错误程序也可能得到时间报告。这是刻意标明的conditional timing prediction。

本层还调用官方非functional静态检查，覆盖跨组HBM race、局部alias、shape等不能只由Scheduler检查的规则。为此分配NaN占位张量，但不生成fixture、执行FP32计算或计算reference。HBM峰值是模拟地址峰值而不是主机RSS。没有独立初始化证明、通用浮点约束、参考正确性或主机超时保证；这些由完整官方evaluate报告确认。

## 6. 有限域MIP与精确运行子问题

对声明域D=product_j D_j，用tuple lifting引入q_d：

\[
\sum_d q_d=1,\quad x_{j,k}=\sum_{d:d_j=k}q_d,\quad
\sum_d A(H_d)q_d\le100.
\]

参数tuple不是已有成绩表：先用生成器导出新程序，从付费指令计算保守下界，然后才调用运行核心测该提案。

初始下界：每wave各SM mandatory command issue最大值+fence；各计算引擎core工作量除以总引擎数，取max。计算read/core/write中的依赖等待被忽略，因此只能是lower bound。

\[
L_c(d)=\max\{L_{issue},\lceil W_{TC}/(Nc)\rceil,
\lceil W_{Vec}/N\rceil,\lceil W_{SFU}/N\rceil,
\lceil W_{Reduce}/(Nr)\rceil,1\}.
\]

专用reduction关闭时工作已计入Vector，不能再除以r。主问题：

\[
\min\sum_d q_d\sum_c w_c\log L_c(d).
\]

选择d后运行精确时序子问题，更新该tuple的成本为sum w log T；若功耗失败或确认数值不正确，禁用q_d。可能成为incumbent时运行完整evaluate。其他host失败/timeout不会作为数学不可行剪枝。

若master最优lower bound≥已验证incumbent（实现数值容差1e-9），在声明空间内停止。预算耗尽/solver未完成时保留未决gap。这里是有限域精确点修正，不是已实现高效通用Benders cut生成器。可能访问全部设计；本轮8点实验确实如此，不宣称搜索加速。

五案例目标与官方加权几何平均等价；纯kernel直接最小化整数cycles。log浮点比较的1e-9容差及solver数值容差限定了证书强度，不是精确有理全局证明。

## 7. 生成、验证和身份绑定

MIP变量映射到真实config字段→官方export→compact JSONL。五案例shared hardware先检查，生成后再由官方grade contract验证。指令与真实分配/缓冲由生成器产生，不能只改变header而保留旧程序来假装软件适配。

预测前后检查程序SHA256；完整evaluate必须与预测的official source hash和程序hash相同。比较整个report（含所有stats/memory字段）、header、step cycles、work units和HBM峰值。数值comparison/output hash来自完整官方执行，不能由时序后端编造。

JSONL生成目前只覆盖官方generator实现族；可以对外部合法JSONL作预测，但不能自动综合任意融合/布局/在线attention程序。

## 8. Gap清单

| 项目 | 已实现 | 仍缺 |
|---|---|---|
| 硬件菜单/面积 | 独立方程和官方双检查 | 浮点边界的位级独立证明 |
| 计算microsteps | 独立Python+差分测试 | 全参数穷尽证明 |
| 计算cycles/power | 单组方程及多组固定仲裁独立模型 | DMA/HBM通信的独立模型 |
| 完整时序/Cache/NoC | 共享官方精确状态转移 | 独立纯MIP状态编码 |
| 完整数值 | 独立reference的官方evaluate | 任意隐藏输入证明 |
| 软件合成 | 官方generator参数空间 | 全部合法ISA程序空间 |
| 最优性 | 声明有限域内的下界/终止条件 | 无限制全局最优、强cut和规模扩展 |
| 工程上限 | 官方parser/scheduler加本层计数检查 | host时间/RSS数学预测 |

零报告gap因为使用相同核心，不等于发现了可脱离模拟执行的全程序解析公式。本轮交付达到“可运行、可生成、可对照、能力边界明确”，尚不满足“独立数学模型完全替代模拟器”的强要求。
