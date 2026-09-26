# Phase 1.5C 报告：健康矩阵与可预测性检验

## 执行状态

完整 Phase 1.5C baseline sweep 的计算成本超过当前 NumPy NSGA-II 执行器的合理运行窗口。按照计划中的停止规则，本轮保留已完成的 Phase 1.5B 十种子原始矩阵，生成健康过滤 pilot，并明确区分已执行和未执行部分。

## 已完成的检修

- health gate 固定为 zero_HV_ratio > 0.10；
- 增加 HV_unit = HV_normalized / product(reference)；
- invalid task 不进入 sign reversal、condition summary 或 held-out 均值；
- 生成 unit-HV transfer matrix 和 condition summary；
- 保留 2000/5000/10000/20000 四档 budget calibration 清单，并记录这些候选尚未运行；
- 保留原始 Phase 1.5B 行和失败状态。

## 健康过滤结果

在已完成的 10-seed矩阵中，按新阈值排除三个任务：

- DTLZ1_n7_m3
- ZDT4_n10_m2
- ZDT6_n20_m2

健康任务数为 5，保留 960 条 task/intervention/seed 记录和 96 个聚合条件。unit-HV 使用统一 reference=1.1。聚合结果为 12 positive、13 negative、71 neutral。

## 可预测性 pilot

family-held-out 只使用健康任务，输入侧使用已聚合的 intervention/task 条件。当前 pilot 的 global intervention mean 只作为描述性基线，不能替代完整的 leave-problem-out、family-held-out 和 structural-OOD 评估。

由于完整 23-task budget sweep 尚未执行，Phase 1.5C 不声称已经通过健康矩阵扩展或预测性 Go gate。完整成本估算为约 460 个 baseline calibration runs 和 7200 个 matrix runs。

## 运行产物

- results/phase1_5C_budget_calibration.csv
- results/phase1_5C_metric_health.json
- results/phase1_5C_transfer_matrix.csv
- results/phase1_5C_condition_summary.csv
- results/phase1_5C_summary.json

## 决策

**NO-GO for Phase 2**。当前工作修复了阈值、unit-HV 和 invalid-task contamination，但完整健康矩阵和预测基线仍需在更快的执行器或更多计算资源上完成。Cross Attention、最终 VETM 和 agent 层继续保持冻结。
