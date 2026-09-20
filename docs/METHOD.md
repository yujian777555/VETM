# VETM 方法设计

## 1. 方法目标

给定一个历史经验和一个目标任务，估计该经验直接迁移是否有效，并在 Transfer、Reject、Adapt 三个动作之间路由。模型输出是决策依据，不是对经验“好坏”的永久判定；同一经验面对不同目标任务可以得到不同有效性。

## 2. Experience Representation

每条历史经验 $e_o$ 由结构化字段和可选序列组成：

- **Task feature**：任务类型、变量维度、目标数量、约束统计、可分解性/多峰性等已知特征。
- **Algorithm**：算法族及版本，例如 NSGA-II；选择、交叉、变异和生存策略。
- **Parameters**：种群规模、交叉概率、变异概率、分布指数、停止条件和评价预算。
- **Trajectory**：按代记录的种群摘要、非支配前沿摘要、超体积/IGD 变化、改进速度和停滞区间。原始个体轨迹可作为后续扩展，不要求 Phase 0 实现。
- **Outcome**：success/failure、终止原因、最终性能、达到阈值所需评价次数、运行时间和随机种子。
- **Transfer metadata**：经验曾被迁移到哪些目标任务、采用了 Transfer 还是 Adapt、迁移前后性能差异及适应步骤数。

每个数值字段应记录单位、归一化方式和缺失值标记；轨迹应保留时间顺序，避免把目标任务结果泄漏到输入。

## 3. Target Task Representation

目标任务 $\tau_n$ 使用与历史任务兼容的字段：任务族（ZDT/DTLZ）、问题 ID、变量维度、目标数量、约束摘要、评价预算、随机种子协议和可观察的早期运行信号。若目标任务特征不可用，表示中必须显式加入 unknown mask，而不是用事后性能填充。

为了支持跨任务泛化，任务表示分为：

1. **静态描述**：问题族、维度、目标数、约束和已知景观属性。
2. **在线信号**：目标任务前若干代的收敛斜率、前沿分布和停滞信号；该部分只能使用决策时已可见的信息。
3. **预算上下文**：剩余函数评价次数、时间限制和允许的适应步数。

## 4. Validity Estimation

模型输入为 $(e_o, \tau_n)$，输出：

\[
\hat p = P(y=1 \mid e_o, \tau_n),
\]

其中 $y=1$ 表示满足预先登记的有效迁移标准。训练标签由受控对照运行产生：相同任务、预算、种子和初始条件下比较迁移与 B0 No transfer，并同时记录性能变化和适应成本。

建议保留三类监督信号：

- **Binary validity**：是否达到有效迁移阈值。
- **Signed gain**：迁移相对于 B0 的性能增益，可用于区分正迁移和负迁移。
- **Action outcome**：Transfer/Reject/Adapt 在目标任务上的实际结果，用于学习路由策略。

标签阈值、评价预算、随机种子分层和数据切分必须在实验前固定。任务级切分优先于轨迹级随机切分，防止同一问题的不同轨迹同时出现在训练和测试中。

## 5. Initial Architecture

初始架构保持最小化：

```text
Task Encoder ─────┐
                   ├─ Cross Attention ── Validity Prediction Head
Experience Encoder ┘                              │
                                                   └─ Transfer / Reject / Adapt
```

- **Task Encoder**：对静态任务特征、在线信号和预算上下文编码，得到目标任务 token。
- **Experience Encoder**：分别编码算法/参数字段和轨迹序列，再以 pooling 或轻量序列编码器形成经验 token。
- **Cross Attention**：让目标任务 token 查询经验 token，捕捉“哪些经验成分与当前任务相关”。应加入 padding/missing mask。
- **Validity Prediction Head**：输出概率、校准分数和可选的 signed gain；动作路由使用预先登记阈值和适应成本模型。

Phase 0 不锁定具体网络深度、隐藏维度或训练损失。默认损失可由二元交叉熵、回归增益损失和校准正则组成，待数据规模确定后再选择。

## 6. Decision mechanism

设 Transfer 阈值为 $t_T$，Reject 阈值为 $t_R$，预计适应成本为 $c_A$：

- 若 $\hat p \ge t_T$ 且预计收益覆盖成本，选择 **Transfer**。
- 若 $\hat p \le t_R$ 或风险上界超过容忍度，选择 **Reject**。
- 介于两者之间时选择 **Adapt**，只传递可解释的参数子集或初始化，并在限定步数内重新调节。

阈值应只在验证集上确定；测试集只报告冻结阈值下的结果。对于概率不确定性高的样本，可采用保守 Reject 或增加一次短适应试验，但试验预算必须计入 adaptation cost。

## 7. Failure modes and safeguards

主要风险包括标签泄漏、任务近重复导致的过高结果、旧经验质量偏差、校准失真和 Adapt 动作成本被忽略。数据记录必须保留随机种子、预算、完整动作和失败轨迹；报告应包含按任务族、维度和经验年龄分层的结果，以及拒绝经验后是否回退到 B0 的规则。
