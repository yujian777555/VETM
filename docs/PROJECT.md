# VETM 项目说明

## 1. Motivation：为什么需要经验有效性验证

递归自我改进（Recursive Self-Improvement, RSI）和 Self-Evolving Agent 会把历史任务中的策略、轨迹、参数和失败教训保存为可复用经验。持久化记忆可以减少重复探索，但“检索到”不等于“当前仍然适用”：任务分布、约束、目标数量、预算或搜索景观发生变化时，直接复用可能把旧的偏差带到新任务中，造成负迁移；过于保守地拒绝所有经验又会失去迁移带来的效率收益。

因此，经验系统需要在调用旧经验前回答一个独立问题：**给定旧经验和目标任务，旧经验在可接受成本下是否会改善目标任务？** 这一步是经验复用的验证门，而不是另一个经验数据库。

## 2. Problem Definition：Experience Transfer Problem

设历史任务为 $\tau_o$，目标任务为 $\tau_n$，历史经验为 $e_o$，动作集合为 $a \in \{\text{Transfer},\text{Reject},\text{Adapt}\}$。Experience Transfer Problem 定义为学习一个决策函数：

\[
q_\phi(e_o, \tau_n) = P(\text{valid transfer}\mid e_o, \tau_n),
\]

并根据阈值和适应成本选择动作：

- **Transfer**：直接将经验中的策略、初始化、参数或算子配置用于目标任务。
- **Reject**：不使用该经验，从通用初始化或基线策略开始。
- **Adapt**：只迁移可识别的子结构，并在受控预算内重新调节参数、算子或策略。

“有效迁移”必须由目标任务上的预先登记指标定义，而不能由任务相似度或旧任务成功与否替代。一个可操作的标签是：在相同随机种子协议、函数评价预算和时间预算下，迁移方案相对于 B0 No transfer 达到预设的性能增益阈值，且没有超过允许的额外适应成本；否则标记为无效。若迁移使性能显著恶化，则记录为 negative transfer。

本阶段只固定研究问题、表示和评估协议，不实现估计器、不训练模型、不运行实验。

## 3. Relation to adjacent areas

| 方向 | 主要回答的问题 | 与 VETM 的边界 |
|---|---|---|
| RSIAgent | 如何探索环境并产生递归自我改进的经验或策略 | VETM 不复制其经验产生机制，而是在经验被调用前判断是否适用 |
| Self-Evolving Coding Agents | 如何更新代码代理的框架、记忆、技能、工具、模型或协作流程 | VETM 可作为其中的经验调用门，当前 Phase 0 不研究代码代理本身 |
| Memory Agent | 如何存储、索引、检索和组织长期记忆 | 检索只提供候选经验；VETM 额外估计迁移有效性 |
| Experience Replay | 在训练或控制过程中重放历史转移以改善学习 | Replay 通常关注采样和学习稳定性；VETM 关注跨任务复用前的有效性与负迁移 |
| Transfer Learning | 如何把源任务知识迁移到目标任务并提高目标性能 | VETM 将经验迁移视为带 Transfer/Reject/Adapt 动作的决策问题，并显式计量适应成本 |

这些方向可以组合：RSI 或 Memory Agent 产生候选经验，VETM 负责验证和路由，Transfer Learning 提供迁移评价视角，Experience Replay 可作为训练估计器的数据来源。

## 4. Expected Contribution（保守表述）

本项目的预期贡献是：

1. 给出面向 Self-Evolving Agent 的 **experience validity estimation** 问题定义，将“是否复用”从普通检索中分离出来。
2. 设计包含 Transfer、Reject、Adapt 的可评估决策协议，并把负迁移率、决策准确率和适应成本作为一等指标。
3. 在演化优化基准上验证一个初始的 Task/Experience 编码与有效性预测架构是否比无门控的复用更可靠。

这些是待验证的研究假设，不宣称提出全新的 RSI 范式，也不宣称复制或超越 RSIAgent。

## 5. Possible paper positioning

论文可以定位为“Self-Evolving Agent 中的经验迁移有效性估计”或“面向经验复用的 validity-aware routing”。核心卖点应是可检验的迁移决策和负迁移控制，而不是重新定义 RSI。若实验不能稳定降低负迁移或适应成本，贡献应收缩为基准、数据协议和失败分析。

## 6. Scope for Phase 0

本阶段交付三份研究基础文档：项目问题定义、方法设计和实验计划。实现、训练、超参数搜索和实验结果均留到后续阶段；任何性能或新颖性结论必须以实际实验为依据。
