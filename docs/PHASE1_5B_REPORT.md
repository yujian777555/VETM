# Phase 1.5B：指标校准与迁移信号检修报告

## 结论

当前为 **No-Go for Phase 2**。校准后的 HV 解决了一部分零值问题，且在健康任务上观察到两个干预的稳定正负翻转；但 family-held-out 迁移预测明显劣于零增益基线，另有 3 个任务仍触发 metric invalid。不能据此声称经验有效性可稳定预测。

## 实验配置

- 8 个任务配置，覆盖 ZDT1/ZDT2/ZDT3/ZDT4/ZDT6 与 DTLZ1/DTLZ2；
- 10 个配对种子；
- population size=50；
- 39 代后代演化，加初始种群，总函数评价预算=2000；
- 20 个确定性干预注册项，按任务过滤与 baseline 完全相同的 no-op；
- baseline 与 intervention 使用相同目标任务、种子和评价预算；
- 统一 normalized HV reference=1.1，per-task ideal/nadir 由固定参考 Pareto 前沿确定；
- practical effect tolerance=0.01，在运行前写入配置；
- 辅助指标：IGD。

原始数据见 results/phase1_5B_transfer_matrix.csv。每行包含 raw_HV、normalized_HV、IGD、配对种子、预算一致性、状态和配置哈希。实验脚本会保留失败行；本次 1540 行均为成功状态。

## Before/After 指标

旧 Phase 1.5 pilot 的零 HV 比率为 **0.6667**。本次 calibrated normalized HV 的零值比率为 **0.3750**。这两组实验的预算、种子数和干预集合不同，因此这个 before/after 比较仅用于健康检查，不能把差异全归因于归一化。

本次 8 个任务中，ZDT4_n10_m2、ZDT6_n20_m2、DTLZ1_n7_m3 的 normalized HV 全为零，被标记 invalid。任务校准公式本身对 8 个任务均有效；异常来自这些任务在当前预算下未进入固定参考点内的有效 HV 区域。其余 5 个任务的 zero_HV_ratio 为零。

## Transfer Matrix

正式运行得到 **154** 个 Task × Intervention 条件、**1540** 条配对记录。聚合条件中：**14 positive、14 negative、126 neutral**。两个干预在健康任务间出现稳定正负翻转：

- mutation_prob_04：ZDT1_n20_m2 与 ZDT3_n20_m2 为 positive，DTLZ2_n7_m3 为 negative；
- mutation_eta_01：ZDT1_n20_m2、ZDT2_n10_m2、ZDT3_n20_m2 为 positive，DTLZ2_n7_m3 为 negative。

该翻转通过 10 个配对种子的 bootstrap 95% 区间与 ±0.01 tolerance 判定，说明值得继续研究，但不足以通过 Go gate。

## Held-out 预测

以 ZDT 条件的每个干预平均效应预测 DTLZ family-held-out 条件：

- held-out 条件数：40；
- 历史干预均值预测 MAE：**0.02687**；
- 零增益基线 MAE：**0.00621**。

历史干预均值预测比零增益基线差约 4.3 倍，没有证据支持跨 family 的可预测性。此分析是简单的全局干预基线，尚非训练模型；更完整的 held-out 模型比较仍需在健康任务扩充后进行。

## 运行与验证

    .\.venv\Scripts\python.exe -m pytest -q tests
    .\.venv\Scripts\python.exe experiments\phase1_5b_experiment.py
    .\.venv\Scripts\python.exe experiments\phase1_5b_analyze.py

当前测试：33 passed。代码编译、完整预算检查及 CSV/JSON 产物读取均通过。

## 局限和后续

1. pymoo/scikit-learn 安装受当前环境 SSL 证书错误阻断；本轮二维/三维 HV 使用项目内精确算法与已知前沿数值测试。三维以上未验证。
2. 3 个任务的 HV 全零，不能用于正负信号结论。应提高预算或调整预注册参考区域，并重新做配对实验。
3. 虽有健康任务上的稳定符号反转，family-held-out 简单预测失败；需增加任务结构特征与样本数量，再考察更强基线。
4. 目前未实现最终 VETM/Cross Attention，也不应进入 Phase 2。
