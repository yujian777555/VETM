# Phase 1.5E-R2 报告：Mean-Score Consistency Repair

## 修复

causal fingerprint 的经验 mean score 统一为：

    empirical_mean_score(F) = mean(sum(F_norm, axis=1))
    delta_empirical_mean = score(probe) - score(baseline)

baseline 和 probe 使用同一 initial population median/IQR normalization。新增 m=2/m=3 identical-front invariant tests，39 项测试全部通过。

## Scale audit

DTLZ1 raw response 最大绝对值：

- R1 m=2：20.06
- R1 m=3：17.40
- R2 m=2：0.4914
- R2 m=3：0.3690

R2 fingerprint stability median cosine=0.8997，median Pearson=0.8844，median Spearman=0.4148。probe fraction 的 min/median/max 为 0.05/0.425/0.50。

## Primary gate

R2 macro task MAE：

- zero-gain：0.0526
- global intervention mean：0.0644
- intervention-only：0.0643
- causal fingerprint：0.0835
- passive + causal：0.0761

5000 task-block bootstrap：

- causal - zero 95% CI：[0.0120, 0.0569]
- causal - intervention-only 95% CI：[-0.0001, 0.0465]
- causal - global 95% CI：[-0.0004, 0.0464]

macro Spearman=0.2763，低于预注册 0.30；1000 permutation p=0.201。causal fingerprint 仍未超过 zero-gain，primary gate 失败。

negative-transfer secondary output 位于 results/phase1_5E_R2_negative_transfer.json。

## 最终决策

**NO-GO for Phase 2。**

R2 完成了最后的数学一致性修复，但 primary gate 仍失败。预测器 rescue 路径关闭，项目转向 mechanistic operator sensitivity、negative-transfer structure 和 intervention-response phase diagrams；不再尝试更深模型、Cross Attention 或最终 VETM。
