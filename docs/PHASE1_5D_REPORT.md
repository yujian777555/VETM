# Phase 1.5D 报告：Early Dynamic Task Context 预测鲁棒性

## 目标与数据

本阶段使用 20 个健康任务的 390 个 aggregated Task × Intervention conditions。probe 使用专用 seeds 100–104，未使用 transfer seeds 0–19。probe budget 按预注册规则计算为 min(1000, max(400, 0.2B)) 并按 population size=50 对齐。

输出：

- results/phase1_5D_task_context.csv
- results/phase1_5D_static_features.csv
- results/phase1_5D_predictability.json
- results/phase1_5D_bootstrap.json
- results/phase1_5D_permutation.json

特征没有使用 benchmark name、最终目标结果、transfer label 或未来轨迹。干预特征来自 Intervention.descriptor，kNN 使用训练集均值和标准差。

## 主 leave-one-problem-out 结果

预注册主模型为 static + dynamic task context + intervention descriptor 的 ridge。task-block bootstrap 结果：

- macro task-level MAE difference（model - zero）：+0.0156
- 95% task-block CI：[-0.0044, 0.0442]
- context-shuffled MAE mean：0.1208
- context-shuffled MAE 95th percentile：0.1585

CI 上界没有小于 0，且真实模型没有稳定优于 zero-gain baseline。因此主 predictive gate 未通过。

## Secondary family-held-out

ZDT train → DTLZ test 的 negative-transfer detection：

- standardized kNN AUROC：0.6387
- balanced accuracy：按 results/phase1_5D_predictability.json 记录
- MLP AUROC：0.3398

该结果只能作为 secondary evidence，不能替代主 leave-one-problem-out gate。

## Independent transfer evidence

Phase 1.5C-R1 的 20-seed candidate confirmation 已保留：46 个候选条件、460 条额外记录，42 个条件保持非中性，within-family reversal 7 个，cross-family reversal 4 个，HV/IGD agreement 7 个。

这说明 transfer effect 现象仍存在，但 Phase 1.5D 没有证明 early dynamic context 能稳定预测该效应。

## 决策

**NO-GO for Phase 2。**

当前不实现 Cross Attention、最终 VETM 或 agent 层。下一步应分析任务结构特征为何无法预测 transfer utility，或重新设计更具因果性的 pre-decision probe。
