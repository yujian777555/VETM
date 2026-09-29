# Phase 1.5E 报告：Oracle-Free Causal Response Fingerprint

## 运行内容

- 20 个健康任务、390 个 aggregated conditions；
- dedicated probe seeds 200、201；
- baseline 与八个 off-grid probe anchors 共享 byte-identical initial population 与 initial objectives；
- one-generation causal response fingerprint；
- probe features 不调用 reference_front、calibrate_task、IGD，也不使用 benchmark name 或最终目标；
- leave-one-problem-out，task configuration 作为 bootstrap block；
- 5000 task-block bootstrap、1000 完整 task-level context permutation。

Probe cost 见 results/phase1_5E_probe_cost.json。每个任务的 probe fraction 为 calibrated budget 的 5%–50%，取决于 anchor collision 过滤和预算。

## Primary results

Macro task MAE：

- zero-gain：0.05261
- global intervention mean：0.06439
- intervention-only ridge：0.06426
- passive + intervention：0.08304
- causal fingerprint + intervention：0.20878
- passive + causal + intervention：0.18035

causal fingerprint 明显没有超过 trivial baseline。task-block bootstrap 中：

- causal - zero 95% CI：[0.02365, 0.39178]
- causal - intervention-only 95% CI：[0.00889, 0.38297]
- causal - global 95% CI：[0.00828, 0.38391]

macro task Spearman：0.2763，低于预注册 0.30；context permutation p=0.307，未通过 p<0.05。

## Audit 与稳定性

results/phase1_5D_R1_taskblock_audit.json 记录了旧 Phase 1.5D 聚合无法恢复完整 20 task-block predictions 的限制。Phase 1.5E 使用新生成的 20 task configuration blocks。

固定初始 population 注入、probe seeds 分离和 oracle 搜索断言均由 tests/test_phase1_5e.py 覆盖。probe collision 被显式记录并排除，不修改固定 anchor。

## 决策

**NO-GO for Phase 2。**

最后的 oracle-free causal fingerprint 仍未超过 zero/intervention-only baseline。按 Planner 的最终 fork，不继续尝试更深 predictor；项目转向 mechanistic operator sensitivity、negative-transfer boundary 和 intervention-response phase diagram 分析。当前不实现 Cross Attention、最终 VETM 或 agent layer。
