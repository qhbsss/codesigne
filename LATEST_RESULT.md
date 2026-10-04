# 当前最高分

完整本地公开评分器实验分数 **55811.66099137171**，`eligible=true`；八万分目标尚未达到。

可复现版本：[final-optimized-55811-20261004](final-optimized-55811-20261004/README.md)。未经编辑的完整评分：[local-grade.json](final-optimized-55811-20261004/local-grade.json)。本轮[数学分析](optimization-80k-20261004/MODEL_80K.md)与[实验记录](optimization-80k-20261004/README.md)。

P1：346276 cycles，18.097779683 W；D1：37417 cycles，18.819380588 W；面积23.36637760512 mm²。相较51742分提高7.8651%。完整公共精评 seed7 通过，另有 seed19 数值与竞争检查。生成器逐字节重生成受评程序，公开评估器、成本模型及冻结baseline未修改。没有服务端官方验证结果，官方 `score=null`。

可直接上传的阶段性 ZIP：[homework-submit-55811-20261004.zip](homework-submit-55811-20261004.zip)，[SHA256](homework-submit-55811-20261004.zip.sha256)。ZIP 根目录和完整原始评分符合 Phase One 格式，解压后可执行 `python -m project.verify_artifacts`。

本轮中间完整合格结果54705分、之前51742/51409/50944分目录和50944分提交ZIP继续保留，各自分数以未经编辑的完整grade为准。未将局部计时当作完整分数，也没有证明八万分全局不可达。
