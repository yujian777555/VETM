# Phase 1.5E-R1 报告：经验尺度修复与最终预测门槛

## 修复

- causal fingerprint 的 empirical mean 和 objective spread 改为 initial population median/IQR normalized space；
- empirical HV 继续使用同一初始 population-derived box；
- 保存 seed-200/201 原始 fingerprint 和 task-level aggregate；
- 增加 DTLZ1 scale audit、fingerprint stability、negative-transfer 输出；
- 维持 20 tasks、probe seeds 200/201、8 个 anchors、one-generation、leave-one-problem-out、5000 bootstrap、1000 permutation。

DTLZ1 原始 raw response 最大绝对值约为 20.06（m2）和 17.40（m3）；修复后 normalized response 最大绝对值为 0.4914 和 0.3690，且非有限值为 0。

## Fingerprint stability

20 个任务的两 seed fingerprint stability 已计算，median cosine similarity 为 0.8997，median Pearson 为 0.8844，median Spearman 为 0.4148。碰撞 anchor 保留 availability mask，不被静默当作有效 probe。

Probe fraction：

- min：0.05
- median：0.425
- max：0.50
- probe fraction >0.25 的任务：14
- probe fraction >=0.50 的任务：7

这说明 probe 不是可以忽略的免费特征，当前不作效率声明。

## Primary prediction gate

20 个 task configuration blocks 的 leave-one-problem-out 结果：

- zero-gain MAE：0.0526
- global intervention mean MAE：0.0644
- intervention-only MAE：0.0643
- causal fingerprint MAE：0.0920
- passive + causal MAE：0.0807

Task-block bootstrap 5000 resamples：

- causal - zero 95% CI：[0.0147, 0.0754]
- causal - intervention-only 95% CI：[0.0021, 0.0660]
- causal - global 95% CI：[0.0018, 0.0655]

macro Spearman 为 0.2763，低于预注册的 0.30；context permutation 1000 次的 p=0.375。

## Negative-transfer detection

负迁移标签为 delta_HV_unit < -0.01。kNN、logistic 和 5-seed MLP 的 held-out 分类结果写入 results/phase1_5E_R1_negative_transfer.json；该结果只能作为 secondary endpoint，不能改变失败的主回归 gate。

## 决策

**NO-GO for Phase 2。**

经验尺度修复显著降低了 DTLZ1 特征幅度，但 causal fingerprint 仍没有超过 zero-gain 或 intervention-only baseline，且 bootstrap CI 全部高于 0。按 Planner 的最终 fork，停止 predictor rescue，转向 mechanistic operator sensitivity、negative-transfer structure 和 intervention-response phase diagrams。
