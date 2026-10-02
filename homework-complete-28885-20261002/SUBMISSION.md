# 完整作业提交包

本包提交的是已经完成 seed 7 完整本地评分的 **28885.37935367514 分**版本，不是后续尚未通过验收的冲分实验。此分数是本地 `experimental_score`，不是服务器验证成绩。

## 文件与要求对应

- `hardware.json`：已评分硬件。
- `programs/M1_P1.asm`、`programs/M2_D1.asm`：与评分报告对应的两份程序。
- `local-grade.json`：评分器原始完整输出，未经编辑；其中历史运行路径按原样保留。
- `project/compiler.py`、`project/schedule.py`：可复现提交 ASM 的生成器。
- `project/functional_check.py`、`project/rough_model.py`、`project/exact_case.py`：功能检查、粗估与单场景精评工具。
- `project/iteration-log.md`：依据留存文件整理的迭代记录。
- `project/reports/`：后续实验的原始报告，不作为本包提交成绩。
- `agent-trace/`：本地可找到的本作业会话原始 JSONL，包括主会话和子会话。
- `challenge.py`、`codesign/`、`baseline_manifest.json`：原始本地评分器及基准，用于复验。
- 作业公告、README、ISA、ABI 等：提交要求和接口参考。
- `project/package-validation.json`、`project/file-manifest.json`：打包检查结果与文件 SHA-256 清单。

ZIP 直接包含上述根目录文件，没有额外外层目录。上传上限为 25 MiB。无需把 ASM 再复制一份到 ZIP 根目录；公告和 README 指定的是 `programs/` 路径。

## 复现与验证

原评分环境为 Python 3.12.14、NumPy 2.3.5、Windows。解压后在包根目录运行：

```powershell
python -m pip install -r requirements.txt
python -m project.verify_package
```

验证脚本只检查，不修改提交 ASM 或评分报告；它会验证生成器输出、ISA 限制、seed、评分报告来源哈希及 agent trace 格式。生成器复现比较忽略操作系统文本换行差异。

若要重新生成 ASM，建议先复制一份目录再运行 `python -m project.compiler`。若要重新进行耗时完整评分，必须使用新的报告文件名：

```powershell
python challenge.py grade --hardware hardware.json --program-p1 programs/M1_P1.asm --program-d1 programs/M2_D1.asm --baseline baseline_manifest.json --seed 7 --report recheck-grade.json
```

不要修改或覆盖附带的 `local-grade.json`。本次重新打包没有再次运行完整评分；提交件未变，已核对报告与硬件、两份 ASM、评分器源码的完整 provenance。

## 已有完整评分结果

| 项目 | P1 | D1 |
| --- | ---: | ---: |
| 功能检查 | 通过 | 通过 |
| Cycles | 680994 | 71030 |
| 峰值滚动窗口功耗 W | 19.5951230135 | 18.0734086558 |

面积为 23.8958883123 mm²，面积、功耗和延迟门槛均通过。报告 `eligible=true`。

学生姓名、学号等由提交者在上传页面按要求填写，本包不猜测或编造身份信息。
