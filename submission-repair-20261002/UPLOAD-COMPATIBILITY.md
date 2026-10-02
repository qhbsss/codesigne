# ZIP 程序字节兼容性修复

服务器在接受 Linux scorer 指纹后返回：`The local report must describe the hardware and programs in this ZIP.`

检查发现，官方 `grade` 使用 Python 文本模式读取程序；输入文件中的 CRLF 会被标准化为 LF，再由 `runner.provenance()` 对 `program.encode()` 计算 SHA-256。因此 Linux 完整报告记录的是 LF 文本摘要。此前 ZIP 却保留了从 Windows 工作目录复制的 CRLF 原始字节。如果上传层直接对 ZIP 条目字节求摘要，二者不一致。

本包将两份 ASM 的 CRLF 机械转换为 LF，并验证转换后的 ZIP 原始条目 SHA-256 与原始 Linux `local-grade.json` 中的 `program_sha256` 完全一致。没有改变任何汇编字符、指令、硬件、报告字段或分数；报告文件保持完整评分器输出的原始字节。

硬件摘要是官方评分器对解析后的规范 JSON 求哈希，已同时核对。ZIP 根目录和 Linux scorer/workload 指纹也继续满足此前检查。

上传服务实现不在公开 starter 中，因此仍以服务器实际回执为准。本修复基于报错顺序和可复现的唯一 artifact 摘要差异，不通过修改报告绕过校验。
