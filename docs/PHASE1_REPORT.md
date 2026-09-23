# Phase 1 初始报告：Experience Transfer Benchmark 与基线验证器

本阶段实现了七个 ZDT/DTLZ benchmark 的 NumPy NSGA-II 生成器、结构化 ExperienceRecord/TransferExample、任务级切分，以及 Random、Nearest-neighbor、NumPy MLP validity validators。没有实现 Cross Attention 或 VETM 模型。

正式轻量运行使用独立环境 `D:\VETM\.venv\Scripts\python.exe`：

    .\.venv\Scripts\python.exe experiments\generate_dataset.py --output-dir data\phase1 --seeds 0 1 2 --generations 5 --population-size 20
    .\.venv\Scripts\python.exe experiments\evaluate_baselines.py --data-dir data\phase1

最新运行统计为 28 条经验（21 成功、7 失败）、252 条迁移样本、76 条正迁移。source bank 使用与 train split 相同的四个任务：ZDT1、ZDT2、ZDT3、ZDT6；验证和测试任务不进入 source bank。

当前已验证：`pytest -q tests` 为 24 passed。限制仍包括样本量小、任务组少、标签使用简化的最终目标均值、未实现 Adapt/适应成本、未报告置信区间。任何结果均不构成 VETM effectiveness 或新颖性声明。
