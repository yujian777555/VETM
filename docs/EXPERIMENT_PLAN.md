# VETM 实验计划

## 1. 研究问题与阶段边界

第一阶段只验证一个基础假设：**在演化优化中，学习经验有效性的门控，能否在保持或提高最终优化性能的同时降低负迁移？** 本阶段不把结果外推为通用 RSI 能力，也不比较未经实现的编码代理系统。

## 2. Benchmark

使用多目标演化优化基准：

- **ZDT**：ZDT1、ZDT2、ZDT3、ZDT4、ZDT6。
- **DTLZ**：DTLZ1、DTLZ2。

每个问题固定变量维度、目标数、函数评价上限、停止条件和随机种子集合。任务划分应覆盖同族迁移（例如 ZDT 内部）和跨族迁移（ZDT 到 DTLZ），并按任务实例分层，避免同一实例的轨迹泄漏到训练和测试。

## 3. Experience source

经验来自 NSGA-II 运行轨迹。每条记录至少包含：

- population size；
- mutation 参数；
- crossover 参数；
- selection 规则；
- 每代 population 摘要和 convergence history；
- 最终前沿质量、评价次数、运行时间、success/failure 和随机种子。

经验库应同时保留成功、失败、早熟收敛和停滞轨迹。只收集成功经验会造成选择偏差，使有效性估计器无法学习何时拒绝经验。

## 4. Baselines

- **B0 — No transfer**：每个目标任务使用固定的 NSGA-II 初始化和参数，不读取历史经验。
- **B1 — Nearest-neighbor retrieval**：按任务特征距离检索最近经验并直接迁移，不学习 validity gate。
- **B2 — Naive memory reuse**：从可用记忆中按固定规则复用（例如最近成功经验或最高历史性能），不显式估计目标任务适用性。
- **B3 — VETM**：使用 Task Encoder、Experience Encoder、Cross Attention 和 Validity Prediction Head，路由 Transfer、Reject 或 Adapt。

所有基线共享算法实现、函数评价预算、硬件环境、随机种子集合和经验可见性规则。B3 的阈值只能用训练/验证数据确定。

## 5. Protocol

1. 在源任务上运行 NSGA-II，生成包含成功与失败的经验库。
2. 按任务实例划分 train/validation/test；测试任务和其轨迹不得参与编码器训练或阈值选择。
3. 对每个测试任务执行 B0–B3，记录动作、迁移内容、适应步数和完整评价轨迹。
4. 以相同种子重复多次，报告均值、标准差和置信区间；同时按任务族、维度、任务相似度和经验年龄分层。
5. 对 VETM 做消融：无在线信号、无轨迹、无 Cross Attention、只二元标签、去除 Adapt 动作。
6. 做校准检查和阈值敏感性分析，避免只报告单一阈值下的最好结果。

## 6. Metrics

必须报告以下指标，并预先固定方向和统计方式：

1. **Final optimization performance**：最终 hypervolume（HV，越高越好）、IGD（越低越好）或任务适用的 Pareto 前沿指标；报告最终值和达到固定质量所需函数评价次数。
2. **Negative transfer rate**：迁移后性能相对于 B0 恶化超过预设容忍度的迁移次数 / 迁移总次数。Transfer 和 Adapt 应分别统计。
3. **Transfer decision accuracy**：冻结阈值下的 Transfer/Reject/Adapt 决策与受控对照标签的一致率；同时报告 precision、recall、F1、AUROC 和概率校准误差（如 Brier score）。
4. **Adaptation cost**：Adapt 额外消耗的函数评价次数、时间、迭代步数和人工/代理操作数；应报告绝对成本及相对于 B0 的比例。

可补充报告成功迁移率、拒绝后回退率、收益-成本曲线和跨任务泛化差距，但不能替代上述四项指标。

## 7. Statistical analysis and acceptance criteria

每个任务和基线使用配对随机种子。除均值外，报告中位数、分位数和置信区间；多任务比较时使用配对检验或 bootstrap，并校正多重比较。Phase 1 的“支持假设”应同时满足：VETM 在预设任务集合上显著降低负迁移率，决策校准不劣于 B1/B2，且最终性能不因门控和适应成本而超过预设退化范围。若只改善某一任务族，应如实报告为局部结果。

## 8. Reproducibility checklist

实验记录必须包含代码版本、依赖版本、硬件、随机种子、任务参数、预算、经验库快照、阈值、失败运行和原始指标。所有标签生成规则、数据切分和排除标准应在运行前写入配置或实验日志，防止以测试结果反向调整有效迁移定义。

## 9. Out of scope for Phase 0

本阶段不实现 NSGA-II 运行器、VETM 模型、数据管线或训练脚本，不生成性能结果，也不声称已经验证任何贡献。后续实现应先依据本计划冻结协议，再进行小规模可行性测试。
