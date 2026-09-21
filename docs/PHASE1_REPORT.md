# Phase 1 初始报告：Experience Transfer Benchmark 与基线验证器

## 实现范围

本阶段实现了：

- 仅依赖 NumPy 的 NSGA-II 经验生成器，覆盖 ZDT1、ZDT2、ZDT3、ZDT4、ZDT6、DTLZ1、DTLZ2；
- 结构化 ExperienceRecord 和 TransferExample JSONL schema；
- 按目标任务实例的 train/validation/test 切分；
- Random、Nearest-neighbor、单隐藏层 NumPy MLP 三个 validity prediction baseline；
- accuracy、precision、recall、F1、AUROC、Brier score 和负迁移率评估；
- 预算一致的受控 transfer label：源经验只提供算子参数，迁移与 B0 使用相同 population size、generation 数、随机种子和函数评价预算。

Phase 1 没有实现 Cross Attention，也没有实现 VETM 模型。

## 数据生成协议

实际运行命令：

    .\.venv\Scripts\python.exe experiments\generate_dataset.py --output-dir data\phase1 --seeds 0 1 2 --generations 5 --population-size 20
    .\.venv\Scripts\python.exe experiments\evaluate_baselines.py --data-dir data\phase1

配置：7 个 benchmark、3 个随机种子、population size=20、5 代。数据生成保留了每个 benchmark 的 3 条成功经验，并额外记录 1 条由无效 population 配置触发的失败经验，避免只保留成功样本。

生成统计：

| 项目 | 数量 |
|---|---:|
| ExperienceRecord | 28 |
| 成功经验 | 21 |
| 失败经验 | 7 |
| TransferExample | 441 |
| 正迁移标签 | 133 |
| 测试集负迁移率 | 0.3889 |

任务切分按 target problem_id 分组，保证同一目标任务的不同 seed 不会跨 train/validation/test。输入特征不含 target outcome、label、signed gain 或事后测试指标。

## Baseline 结果

冻结 test split 的结果：

| Baseline | Accuracy | F1 | AUROC | Brier |
|---|---:|---:|---:|---:|
| Random | 0.4762 | 0.3654 | 0.5184 | 0.3555 |
| Nearest-neighbor | 0.6429 | 0.4944 | 0.7195 | 0.1933 |
| MLP | 0.7222 | 0.5455 | 0.7769 | 0.1772 |

这些数值说明在当前小规模、固定预算协议下，validity label 具有一定可预测信号；它们不证明 VETM 优于基线，也不能外推到 Self-Evolving Agent。后续应增加任务实例、种子和预算，并进行置信区间、校准曲线和跨配置泛化测试。

## 已知限制

1. 当前迁移动作只复用 NSGA-II 算子参数，不迁移完整种群或轨迹状态；这是为了先隔离 validity estimation。
2. 目标函数使用最终种群的平均目标值构造 signed gain，后续需要按 HV/IGD 和 Pareto 前沿质量复核标签。
3. 当前任务级切分只有 7 个任务族，测试组较小；结果方差可能较大。
4. 失败经验由受控无效配置生成，后续应增加自然停滞、早熟收敛和运行异常。
5. 还没有适应动作、adaptation cost 或三分类路由训练；这些留到后续阶段。

## 结论

Phase 1 的初始目标是建立可复现数据与 baseline validator。当前运行支持“先验证 validity prediction 是否可学习”的方向，但不支持任何 VETM effectiveness 或新颖性声明。
