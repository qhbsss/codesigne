# 第二阶段模型 V2：程序生成与请求级竞争约束

## 状态与范围

本文是基于官方源码的新版数学规格，不是已完成的完整求解器。现有 `estimate.py` 尚未实现本文件全部约束；3883.12 分仍是旧估计器预测，3441.47 分仍是已验证基线。不得用新增公式宣称已消除误差或得到新成绩。`MODEL.md` 保留现有实现与历史证据。

模型范围是声明的合法硬件菜单和有正确性条件的静态程序模板族。完整模型为离散事件约束优化；有限程序选项的外层可以实现为 MILP，内部精确时间不能当作任意自由调度。以下时间均为整数周期，区间采用半开区间。

## 1. 目标函数

五场景 c 的权重 w=(1/4,1/4,1/6,1/6,1/6)，参考周期 R=(187689274,374856358,27068433,42358913,70439292)。共享硬件 H、程序参数 θ_c、实际生成程序 P_c 及其请求级完成时间 T_c 满足：

\[
 \min J=\sum_c w_c\ln T_c,\quad P_c=G_c(H,\theta_c),\qquad
 S=1000\exp\left(\sum_c w_c\ln R_c-J\right).
\]

只有全部可行性门槛通过时 S 才是有效成绩。没有更改题目目标，也不把预测误差补偿作为新成绩。

若选项 a 是完整生成的程序、且固定 H 后各场景独立，可使用外层变量 x_h,y_hca：

\[
 x_h,y_{hca}\in\{0,1\},\quad\sum_hx_h=1,\quad
 \sum_{a\in A_c(h)}y_{hca}=x_h,\quad
 \min\sum_{h,c,a}w_c\ln T_{hca}\,y_{hca}.
\]

这里 T_hca 必须来自请求级子问题或已验证运行，才是相应成本；用近似时间代替只能证明预测目标最优。直接以变量 T 优化 ln(T) 并非线性 MIP；固定选项成本、有限枚举或分段近似是不同实现，近似必须报告误差。

## 2. 变量：决策和执行结果必须区分

| 类别 | 变量 | 含义 |
|---|---|---|
| 硬件决策 | x_jk∈{0,1}, H_j=Σ_k v_jk x_jk | 23 字段菜单，Σ_k x_jk=1，五案共享 H |
| 分块决策 | m_l,n_l,k_l；z_lκ∈{0,1} | 每算子选择一个合法分块模板 κ，而非任意不存在的指令 |
| 空间排布 | u_wgs∈{0,1} | wave w 的组 g 归属 SM s，Σ_s u_wgs=1 |
| 布局与复用 | b_lλ,f_lφ,r_lρ∈{0,1} | 实际地址布局、融合模板、A/B 复用模板；分别 one-hot |
| 预取与缓冲 | d_l∈有限菜单，v_bslot∈{0,1} | 预取距离和物理缓冲槽，必须导出真实 load_async/wait |
| 执行时间 | s_e,f_e∈Z≥0 | 微步骤、地址批、数据包、请求、局部 ack 的起止时间，由执行规则决定 |
| 请求状态 | a_q,i_q,r_q,h_q,o_q,k_q | 生成、提交、HBM准入、HBM开始、HBM完成、最后局部 ack |
| 执行状态 | Z_n,ρ_n,η_n | PC/token/队列/端口/缓存，轮转游标，同周期事件序号 |
| 输出 | T_c,E_c,P_c,W | 周期、能耗、窗口功率，不是可自由选择的优化旋钮 |

融合与复用的变量只激活已定义的模板。声明模板必须规定实际指令体、缓冲、读写集合和数值条件。模板外程序不属于最优性声明域。

## 3. 算法与存储合法性

矩阵乘模板覆盖每个所需输出及全部归约项：

\[
 C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj},\quad
 \bigcup_b I_b=[0,M),\ \bigcup_b J_b=[0,N),\ \bigcup_b K_b=[0,K).
\]

输出覆盖需唯一；split-K 的部分和必须显式付费归约。因果注意力只允许在 j≤i 范围计算有效值，j>i 的 softmax 权重必须为零，不能遗漏归一化分母或改用有限错误常量。跨模板 FP32 累加顺序可能变化，实数等价不构成数值验收证明。

对每个实际物理缓冲 b，大小为 n_b 字节，生命周期为 [α_b,β_b)：

\[
 \forall w,s:\ \sum_g u_{wgs}R_g\le1024H_{rf},\quad
 \sum_g u_{wgs}SH_g\le1024H_{sh},\quad
 \sum_g u_{wgs}\le H_{resident}.
\]

R_g/SH_g 必须容纳所有活跃物理槽和标量，不能只用理想算子张量大小替代。基础 GEMM RF 模板的必要条件为

\[
 4(d_A mk+d_Bkn+mn+n_{aux})\le R_g.
\]

广播、常量、SH 存放或非满尾块模板应按真实物理分配替换此式。复用同一槽的两个不同值须有 β_b≤α_b′ 或 β_b′≤α_b；异步消费者的最后读取必须早于覆盖写入。双缓冲必须真的分配两槽，不得只赋予重叠收益。

\[
 M^{peak}=\max_t\sum_{b:\alpha_b\le t<\beta_b}n_b^{allocated}\le2\,GiB,
 \qquad M^{cum}=\sum_{b\text{ allocation}}n_b^{allocated}\le64\,GiB.
\]

HBM 基址按官方栈分配和 256B 对齐；release 只能遵守实际分配规则，同时触发相应缓存失效。局部 RF/SH 不能跨 wave 保存。跨组读写必须满足官方 race/alias 规则；物理行锁不能使本来非法的元素数据竞争变合法。

## 4. 微步骤、端口与依赖

保留现有已验证计算微步骤 profile。对微步骤 e 的读、计算、写：

\[
 f_e^r=s_e^r+d_e^r,\quad s_e^{core}\ge f_e^r+\ell_e^r,\quad
 f_e^{core}=s_e^{core}+d_e^{core},\quad s_e^w\ge f_e^{core},\quad
 ready_e=f_e^w+\ell_e^w.
\]

端口服务与依赖延迟分开：等待延迟不持续占用端口、不均摊动态能量。每 SM、资源 r、周期 t：

\[
 \sum_{e\text{ using }(s,r)}\mathbf1\{s_e\le t<f_e\}\le cap_{s,r}(H).
\]

RF 读写、SH bank、SH→TC 接口和各计算引擎分别建资源，DMA 的读源/回写也进入同一端口约束。SH bank 冲突由实际 word 地址决定；不是总字节数除以带宽。

同步命令约束下一条 PC 就绪；异步命令只产生 token。对 DMA d：

\[
 F_d=\max_{q\in Q_d}k_q,\quad s_{wait(d)}\ge F_d,
 \quad s_{consumer}\ge f_{wait(d)}.
\]

DMA 完成同时要求地址已全部生成、frontend 为空、所有局部 ack 完成；空传输亦遵守官方命令流程。wave 完成后付 fence；commit 的空 fence 单独计入，T_c 为这些边界的真实完成时间。

## 5. 请求生成、DMA FIFO 与背压（新增）

DMA 按每批至多 16 个元素生成地址；批内同一 64B 行合成一个请求，不自动跨批去重。元素 x 的行及通道：

\[
 L(x)=64\left\lfloor\frac{base+4\,offset(x)}{64}\right\rfloor,
 \quad channel(q)=\left\lfloor\frac{L_q}{256}\right\rfloor\bmod H_{channels}.
\]

每 SM 每周期至多一次地址批生成、至多两次提交。每 DMA 引擎 FIFO 长度≤1+dma_depth，只有队首可前进；启动延迟及 load/store 源服务按源码支付。地址批生成还必须满足源码的 frontend 门槛：生成前该 SM front 数≤16，单批至多新增16，因此总上限32；store staging 数也须检查，不能只用32这个总容量。

\[
 \sum_{b\in batches(s)}\mathbf1\{a_b=t\}\le1,\quad
 \sum_{q\in Q(s)}\mathbf1\{i_q=t\}\le2,\quad
 Pending_s(t)=\sum_{q\in Q(s)}\mathbf1\{i_q\le t<k_q\}\le128.
\]

128 是当前官方 hardware 派生的 per-SM pending limit；实现必须从硬件方法读取，避免不同版本使用错误常量。store 必须先完成本地 RF/SH 源读取；load 只有完成返回后的本地 RF/SH 写入与依赖延迟才产生 ack。阻塞时不能跳过 FIFO 队首选择后面的请求。

## 6. HBM 准入、250 周期 ready、方向切换与信用（新增）

请求 q 经实际上行/lookup 路径到达准入队列，记 a_q^H。准入时间 r_q 由 FIFO 和信用决定，r_q≥a_q^H；ready 时间为 r_q+250。对按官方 ready 事件次序在同一通道相邻的 q′,q：

\[
 h_q=\max(r_q+250,o_{q'})+4\mathbf1\{dir(q)\ne dir(q')\},\qquad o_q=h_q+2.
\]

首次请求无方向切换代价。HBM 服务次序是执行结果，不能由优化器重排成理想 read/write 批次。

\[
 \forall h,t:\sum_{q\text{ physical HBM ticket on }h}
 \mathbf1\{r_q\le t<k_q^{last}\}\le H_{queue}.
\]

信用释放直到最后目的端的局部 ack；不是 o_q。合并读取只占一次物理 ticket，且 k_q^last=max_dest k_qdest。无 cache、多播关闭、无合并情形下，每个请求各计 ticket。行读写占用与 miss 合并的合法性另由状态确定，不能把所有相同行请求任意合并。

## 7. NoC 数据包资源与局部返回（新增）

每个事务拆成官方实际路径上的包 p（请求、读返回、写 payload、ack、fanout），分别列出经过的全局和 SM 端点资源。对资源 r 的逐周期发送字节 b_prt：

\[
 \sum_p b_{prt}\le B_r(H),\quad b_{prt}\ge0,\quad
 \sum_t b_{prt}=bytes_{pr},\quad
 s_{p,next}\ge f_{p,r}+\ell_{pr}.
\]

仅此容量式不保证真实调度：还要加入源码 FIFO/轮转规则、逐包剩余字节递推和事件次序。包长与方向必须从实际 packet 类型提取，不能用72/80B总量同时代替上行下行各阶段；传播延迟是路径参数，不能将依赖等待作为占用带宽。返回到 SM 后的 RF/SH write service 进入第4节共享端口，信用在其 ack 后释放。

## 8. 固定仲裁：排除虚假的最优调度（新增）

Z_n 含 PC、token、各 FIFO、引擎/端口空闲时刻、行锁、缓存、请求及功耗事件。事件键为 (t_n,η_n)，同周期按源码事件序号处理。轮转扫描优先级为 π_n(g)=(g-ρ_n) mod |G|。对每 SM 的命令 issue 候选，必须选扫描遇到的第一个合法可发射组，而非优化器自由选组：

\[
 issue_n(g)=1\Rightarrow eligible_n(g)=1\ \land\
 \neg\exists g':\ sameSM(g',g)\land eligible_n(g')\land\pi_n(g')<\pi_n(g).
\]

每次真实调度扫描后 ρ_(n+1)=(ρ_n+1) mod |G|；它不是每经过一周期就增加。没有立即可推进工作时，下一时间为官方规则下最早未来事件/ready 时刻；不能在跳过的周期虚构扫描。微phase推进、DMA推进、返回处理与命令发射的同周期顺序亦必须保持。

因此完整约束应表示为

\[
 (Z_{n+1},t_{n+1},\eta_{n+1},\mathcal E_n)
 =\Phi_H(Z_n,t_n,\eta_n,P_c),\quad T_c=t_{last}.
\]

第4—7节为 Φ 的可检查展开模块；容量不等式只是必要条件，只有固定决策规则全部展开才可能复现完整周期。其 Boolean implication、min/max、整数时间和状态条件可用 CP/SMT 或有界 big-M 展开；big-M 必须取有效时间上界，不能忽略同周期事件规则。完整程序规模很大，推荐主问题+事件子问题，而非逐周期铺满8×10^9列。

## 9. Cache / 多播的分层处理

第一实现域先约束 cache_mib=0、multicast=false，正好覆盖3441分基线；这样没有虚构命中率，仍有行锁和读请求状态规则。若扩域启用 cache/multicast，必须新增：4-way set/tag/valid/LRU stamp、实际 lookup/fill/失效顺序、MSHR容量和 FIFO、缓存每周期服务、未完成读取合并资格及≤32目的端的 fanout。hit_q 由此前同 set 状态计算，不能由求解器直接选成1；跨wave缓存状态不得默认清空，release按源码失效。

仅用“命中率×HBM流量”不属于新版精确子问题。第二层应在第一层请求级留出验证通过后再开放这些硬件变量。

## 10. 面积、能耗与数值门槛

保留 `hardware.py` 和 `MODEL.md` 的23字段面积和组合合法性；A(H)≤100，P0(H)=0.025A+0.15channels。事件 e 的能量 E_e 使用 pJ，实际付费服务区间 [s_e,s_e+d_e)：

\[
 P_W=P_0+\max_\tau\frac{1}{2000W}
 \sum_e\frac{E_e}{d_e}
 \left|[s_e,s_e+d_e)\cap[\tau-W,\tau)\right|,
 \quad P_{100}\le34,\quad P_{10000}\le26.
\]

固定事件后最大值可在事件起止点及其加 W 的断点检查，含 idle tail。所有 compute、地址生成、HBM、NoC、局部访问、cache 和 fence 能量均需进入，等待只计基础功率。不能用整程序平均功率代替窗口峰值。

输出与新 K/V 满足官方 FP32 计算参考：

\[
 |y_{cj}^{\sigma}-\hat y_{cj}^{\sigma}|\le10^{-3}(1+|\hat y_{cj}^{\sigma}|).
\]

当前采用经过验证的模板和同 SHA 官方数值验收；要将这个约束纯公式化，需要 FP32 舍入逐操作编码或针对输入域的误差证明，普通实数 MIP 无法代替。其他 record/work/step/wave/cycle/shape 限制从官方合同提取；host timeout/RSS不作为算法数学不可行的证据。

## 11. 求解界、误差与最优性

安全周期下界 L_c 采用已验证必要资源/依赖约束。若新增HBM/NoC下界，必须先证明请求数量不可因 cache/合并减少且路径服务确实不可重叠，不能把本文件的近似解作下界。外层合法对偶界 B 与完整通过验收的成本 U 满足

\[
 B\le J_D^*\le U,\quad
 S_{best}=1000e^{\sum w\ln R-U},\quad
 S_{upper}=1000e^{\sum w\ln R-B},\quad
 S_{upper}/S_{best}=e^{U-B}.
\]

仅预测最优没有 U-B 证书。预算结束报告未决 gap；即使域内 gap=0，也不代表模板族外或隐藏输入上的全局最优。

近似通信模型可用于候选排序。校准得到的误差区间不是可证时间区间，不可据此永久剪枝或报告数学最优。只有证明 L≤T≤U 或精确验证后，才可基于相应界淘汰。

## 12. 相比旧模型完善的内容与落地顺序

1. 用请求级准入/ready/service/ack代替整条DMA通道预留和270/260/310等经验惩罚；当前未完成实现，禁止宣称数值误差已改善。
2. 新增方向切换4周期、信用占用至最终局部ack，以及端点/全局NoC分别竞争；旧模型没有完整表达这些因果链。
3. 把DMA回写和compute接到同一RF/SH端口，将FIFO、frontend、提交上限及pending背压显式建模。
4. 固定仲裁与同周期事件规则限制时间变量，排除求解器构造模拟器不会执行的理想调度。
5. 扩展分块/融合/复用/空间/预取变量，并新增实际物理缓冲生命周期和真实程序生成约束；并非已枚举了所有模板。
6. 用合法下界与实测上界报告证书，预测误差和求解gap分别报告。

推荐实现顺序：无cache/多播请求级子问题→单请求/方向切换/credit/FIFO/共享端口微实验→未参与校准的波形留出验证→3441方案五案对照→扩大程序生成菜单→外层MIP与同SHA数值验收→最后启用cache/多播状态。每一步应检查cycles、流量、pending、能量和窗口功率；整个五案必须验证后才能报告模型精度和新分数。
