# Transformer codesign 优化续跑

原始版本保留在 ../final-optimized-30394-20261002，本目录为独立实验空间。

- ANALYSIS.md：历史核对、50k预算、优化方向及局部筛选结果。
- RESULTS.md：只列完整官方grade结果。
- candidates/：每个候选的独立硬件、生成器、ASM。
- reports/：原始功能检查、局部/单项时序及完整grade；日志与报告分别保存。
- project/：本轮诊断和生成辅助脚本。

公开评分器 codesign/challenge 与历史30394提交完全一致，未作修改。
使用 Python3.12、NumPy2.x；建议 OPENBLAS_NUM_THREADS=1 避免线程争抢。

复现某候选：

```sh
python -m project.generate_candidate merged128
OPENBLAS_NUM_THREADS=1 python challenge.py grade \
  --hardware candidates/merged128/hardware.json \
  --program-p1 candidates/merged128/M1_P1.asm \
  --program-d1 candidates/merged128/M2_D1.asm \
  --baseline baseline_manifest.json --seed 7 \
  --report reports/merged128-grade-rerun.json
```

请始终使用新报告名，并保留评分器产生的原始JSON，不手改分数或门槛。
