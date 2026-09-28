# Phase 1.5C-R1 报告

## 执行范围

- 修复三目标 unit-HV 缩放：DTLZ2 m=3 使用 `1.1^3=1.331`，旧逻辑错误使用 `1.1^2=1.21`。
- saturation 改为 unit-HV 质量阈值，不再依据 seed 间波动。
- 实际完成 230 个 baseline calibration runs。
- Stage A/B 合计 20 个健康任务、390 个聚合条件、3900 条 paired run rows。

## 结果

- positive groups：41
- negative groups：65
- neutral groups：284
- within-family reversal interventions：7
- cross-family reversal interventions：6
- runtime benchmark：2k/5k/10k 约 0.149/0.374/0.724 秒
- 估算 7200 次运行约 2993 秒

invalid tasks 未进入聚合矩阵和预测分析。候选 reversal 已补跑 seed 10--19：46 个候选条件、460 条新增记录，合并后 20 seeds 确认 42 个非中性条件。

## Predictability

`results/phase1_5C_R1_predictability.json` 包含 family-held-out 和 leave-problem split。输入只包含任务结构和 intervention descriptor，训练统计只来自训练任务。

Stage A 的 global intervention mean MAE 为 0.0983，ridge MAE 为 0.1146，zero-gain MAE 为 0.0920，nearest-neighbor MAE 为 0.0820。

## 决策

**Stage B：完成矩阵扩展；Phase 2：NO-GO（暂缓）。**

20-seed 确认后，within-family reversal 为 7 个干预，cross-family reversal 为 4 个干预，HV/IGD 一致的候选干预为 7 个。Stage B 已完成；Phase 2 仍需重新运行健康任务上的 held-out predictor 并确认预测改进稳定后才能进入。当前不实现 Cross Attention、VETM 或 agent layer。
