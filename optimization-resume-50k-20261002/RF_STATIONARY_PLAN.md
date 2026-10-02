# 新方向：64 WG RF-stationary prompt

发现：resident2 的 shared cache bank 最忙383027cycles/540914，NoC忙409417cycles，cache查询924496，NoC70.43MB。
merged128查询875646、NoC71.81MB，cycles544669，20.489W超限。单纯tile/row融合没有去掉远程权重供给与不同WG的时间漂移。

假设：64个WG，每SM4个，合并128行dense。每个WG缓存当前层自己的QKV/WO/W1/W2权重到RF，不缓存三个层。
- QKV64切片，每片12列，256*12=3072word（两个lane各1536，有512空洞）。
- WO32切片，每片8列，256*8=2048word，仅WG0..31做WO。
- W1分64个16列切片，每片256*16=4096word。
- W2分4K*16N=64任务，每片256K*16N=4096word。
- 无WO的WG32..63每组11264word，负责128行LN的4行；RF6[0:1024]可存ln1/ln2 gamma/beta。
- 有WO的WG0..31每组13312word（RF0..5 +RF6[0:1024]）。
- RF6[1024:1536]为32x16 acc；RF6[1536:1568]存W1、W2bias；RF7整lane为临时A、LN/attention/GELU。

RF权重packing：QKV两个128K chunk各1536；WO用512word chunk填RF0/1空洞；W1/W2用32K*16=512word chunk first-fit填满，保留RF6>=1024和整个RF7。
QKV临时A按m16,k128=2048，acc192；WO m16,k128=2048，acc128；W1/W2固定m32,k32=1024，acc512，不合并K chunk，以保持各WG相同指令时序。
LN WG32..63每组4行，参数RF6，临时RF7。W1整tileGELU只用RF7临时和RF6acc，不覆盖常驻权重。
attention用WB group中16/32组，但RF0..6不可用，必须重写RF7-only attention，或在attention前关闭WB WG、单独attention WG then重新恢复WB（后者失去RF缓存，不采用）。
可在QKV/attention之间使用专用8或16 WG，必须保持每SM<=4：64个WB已占满，需WB WG自己执行RF7-only。块attentionm8,n8,k32，score单row/8row compact至少512word，所有临时放RF7需要layout；初步采用逐query m1，K/V 16keys每次512 RF7[64:576]，score65 RF7[600:665]，ctxRF7[700:732]，query[0:32]。prompt逐query很慢，建议专用blockedRF7-only m8 scores8*64=512、query256、K256、P64、V256、ctx256，共1600word可用。

初始化按矩阵种类全WG barrier对齐。QKV各组相同tile形状，WB常驻消除加载漂移，输入A在不同SM同cycle，期望multicast把每波16个SM的source cache查询合并。
理论目标：P1约25~30万，D1约5~6万，公式分约4.7~5.7万。这是未验证假设。先精确一个dense段和功能检查，若未显著改善则停止完整精评。

## 实现更新

代码位于candidates/rf-stationary/project/rf_prompt.py。
WO改为32个8列切片，以降低TC尾部浪费；仅WG0..31拥有WO权重。
无WO的WG32..63执行LN，各组4行，并将四个256元素LN参数向量缓存在RF6[0:1024]。
权重packing按512word粒度，WG0..31共13312word，最后RF6[0:1024]放权重，acc从1024起；WG32..63权重11264word，RF6全空可放LN和acc/bias。
attention已重写为RF7-only的8x8 score/context tile，Q256、K/V256、score512、scoretile64、ctx256、两个scalar总布局不超过2048word，保留RF0..6的全部权重。
FOR压缩连续RF weight chunk与标识符后，P1源码7369784字节，低于8MiB。
此版本功能检查及W1孤立阶段精评在运行。暂不进行全程序精评，待局部收益验证。

## 筛选结论

功能通过，P1最大绝对误差3.01e-6。
孤立W1阶段33343cycles、8.770227W、HBM274624bytes；比vector8-block W1 41519快19.70%。
当前层四种权重+参数RF初始化29211cycles、15.223685W、HBM3554560bytes，cache bank忙约28782cycles，仍受固定cache lookup吞吐限制。
孤立W2阶段39587cycles、9.873658W、HBM1554688bytes，慢于原resident2 W2 31808。K分组与输入读取时序漂移未能稳定保持multicast。
三项每层合计102141cycles，三层约306423cycles；尚未计其他阶段及首个新位置。以D1 62630计算，5万分要求P1<=257762.29。隔离阶段与完整程序缓存不同，因此这是筛选粗估，不是该方案的完整分数或不可能性证明。
按此筛选不开展RF-stationary全程序grade；保留生成器和功能/局部时序报告，未把它当作有效成绩。
后续若重试，应先改善W2输入对齐与RF初始化/计算重叠，重新筛选，再全程序精评。
