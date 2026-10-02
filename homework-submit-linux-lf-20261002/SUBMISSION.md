# Phase One 官方 Linux 评分提交包

此包针对上传报错 `The local report uses a different scorer or workload release.` 重新生成。原硬件和两份 ASM 保持不变，`local-grade.json` 来自 Ubuntu / Python 3.12 / NumPy 2.3.5 中对官网当前原版评分器进行的完整 seed 7 `grade`，报告没有编辑。

## 上传入口与文件结构

请在 https://linux-slai.tail6d76d1.ts.net:8443/phase-one/#submit 上传本 ZIP，并选择 **Phase One**；不能将本作业提交到单独的 Phase Two 入口。

ZIP 根目录包含 `hardware.json`、`programs/M1_P1.asm`、`programs/M2_D1.asm`、`local-grade.json`，以及生成器、迭代日志、完整原始会话快照和官方评分器源码。没有额外目录外壳。

使用本人真实学号与一致的显示名。同一学号每次上传至少间隔 10 分钟。本包只在本地验证，未替用户提交，也没有服务器接受回执；服务器验证成绩以实际回执为准。

## 结果与校验

本次精评分数、功能/面积/功耗门槛和运行环境见 `project/linux-grade-summary.json`，权威原始结果在根目录 `local-grade.json`。`project/file-manifest.json` 记录各文件 SHA-256。

打包时比对了官网新下载 starter 的全部评分相关源码原始字节；报告中的源码路径和汇总指纹是原版评分器在 Linux 下自然生成的，没有修改评分器或伪造版本标识。问题分析见 `project/COMPATIBILITY.md`。旧包仍单独保留。

可在解压目录执行轻量只读校验：

```text
python -m project.verify_submission
```

复现完整报告请在 Linux 中安装 Python 3.12 与 `numpy==2.3.5`，使用新输出文件名，避免覆盖本包原始报告：

```text
python challenge.py grade --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --baseline baseline_manifest.json --seed 7 --report recheck-grade.json
```

`project/iteration-log.md` 保留之前 28885 分方案的迭代过程，提及的 Windows 打包检查属于旧包历史记录；本次新增 Linux 重评与兼容性检查见上述新文档。
