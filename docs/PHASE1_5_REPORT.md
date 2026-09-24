# Phase 1.5 报告：Transfer Matrix Pilot

## 目标

Phase 1.5 检修 Phase 1 的科学有效性问题：固定多目标质量指标、控制任务与干预、使用配对种子和相同函数评价预算，再观察 transfer effect 是否跨任务改变符号。

本阶段没有实现 Cross Attention、VETM 最终架构或 LLM agent。

## Pilot 协议

本次 pilot 使用：

- 8 个任务配置：ZDT1/ZDT2/ZDT3/ZDT4/ZDT6 的两个维度配置，以及 DTLZ1/DTLZ2 的三目标配置；
- 8 个确定性干预；
- 3 个配对随机种子；
- population size=20，5 代，函数评价预算=120；
- baseline 与 intervention 使用相同目标任务、随机种子和预算；
- 干预只改变 mutation probability、mutation eta、crossover probability、crossover eta，或 tournament size。

完整矩阵由以下命令生成：

    .\.venv\Scripts\python.exe experiments\phase1_5_pilot.py

输出：

- results/phase1_5_transfer_matrix.csv
- results/phase1_5_summary.json

## 指标

主指标是固定参考点下的归一化 HV 差值：

    delta_HV = HV(intervention) - HV(baseline)

参考点写入任务配置并在 paired runs 中固定。二维和三维 HV 使用项目内可测试的精确切片实现；更高维任务暂不进入 pilot。IGD 使用由问题定义生成的固定参考 Pareto 前沿作为辅助指标。

每个 Task × Intervention 组记录 mean、median、standard deviation 和 bootstrap 95% 区间。正/负/中性判定使用运行前固定的 practical tolerance=0.01。

## Pilot 结果

- 条件数：64
- 配对运行数：192
- 任务数：8
- 干预数：8
- 配对种子数：3
- 预算一致性：所有记录为 true
- 当前 3 seed pilot 中没有条件的 bootstrap 区间稳定越过 ±0.01，因此不能宣称稳定的正/负 transfer boundary。

原始矩阵和汇总文件已保留。由于 pilot 只有 3 个种子，结果只用于检修管线和估算完整运行成本。

## Go/No-Go 判断

当前为 **No-Go for Phase 2**，理由是：

1. 还没有完成计划要求的 10–20 个配对种子；
2. 还没有 30–50 个任务配置；
3. 还没有 family-held-out、structural OOD 和完整预测基线；
4. pilot 尚未显示可稳定支持的正负符号反转；
5. pymoo/scikit-learn 依赖安装因当前环境证书错误失败，本 pilot 使用项目内实现和 NumPy baseline。

下一步应先扩展任务配置和配对种子，再根据聚合 transfer matrix 判断研究假设是否成立。当前结果不构成 VETM effectiveness 或新颖性声明。
