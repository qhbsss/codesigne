# 上传兼容性修复记录（2026-10-02）

官网第一阶段说明：https://linux-slai.tail6d76d1.ts.net:8443/statement/

## 已核实的事实

1. 官网 Phase One 当前仍发布 v0.7.3；Phase Two 使用另一个入口、程序和分数，不应混用。
2. 官网要求 ZIP 根路径为 hardware.json、programs/M1_P1.asm、programs/M2_D1.asm、原始 local-grade.json；每包 ≤25 MiB，最终提交附自己的 agent trace。
3. 旧完整包具备上述文件，不能把服务器的版本不匹配提示解释成缺报告或目录错误。
4. 新下载官网 starter 中，所有参与旧报告 provenance 的源码文件与旧报告记录的 SHA-256 逐项一致。
5. 原版 runner.py 用 str(path.relative_to(root)) 作为源码哈希字典的键，再对字典计算 simulator_sha256；它没有规范化路径分隔符。
6. 在原版 Ubuntu 评分器中复核，硬件/ASM/成本模型/工作负载指纹均不变，只有 source_sha256 的路径表示和 simulator_sha256 不同：
   - Windows: 187d59f2be81c7e5817d208fc04574e5bc7d68e2b850117917c6dfd93525df81
   - Linux: 09b3ba4005c529694ab122c9a47162b011203038f2d402ab39498b249bc92c7b

## 采取的措施

使用官网直接下载并解压的官方评分器，在已有 Ubuntu 环境重新完整运行 seed 7 grade。硬件与 ASM 从旧有效包逐字节复制。没有改评分器源码，没有手改报告中的路径、版本、哈希、分数或计时数据。

新包采用这次完整运行的原始 local-grade.json。对照官网下载源文件验证 Linux source_sha256、simulator_sha256、trust_policy_sha256 和冻结基准；另验证硬件与 ASM 哈希、正确性/面积/功耗/延迟门槛及 ZIP 字节完整性。

## 证据边界

服务端上传校验实现不在公开 starter 中，也没有用上传反复试错。跨平台指纹差异已实测确认，是该报错的高度可疑原因；没有服务器回执之前，不宣称已被服务器接受或获得官方 Verified 成绩。

官网 BEFORE YOU UPLOAD 是通用规则，意为附上当前官方 grade 生成的原始报告、不要以上传代替本地验证、遵守 10 分钟间隔、使用一致真实身份。它本身不是这次失败的错误信息。
