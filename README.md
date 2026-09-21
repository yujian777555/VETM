# VETM Phase 1

Phase 1 先建立演化优化经验转移基准与 validity prediction baseline，不实现 Cross Attention。

## 环境

使用项目独立环境：

    D:\VETM\.venv\Scripts\python.exe

安装依赖：

    D:\VETM\.venv\Scripts\python.exe -m pip install -e .
    D:\VETM\.venv\Scripts\python.exe -m pip install pytest==7.4.4

## 运行

    .\.venv\Scripts\python.exe experiments\generate_dataset.py --output-dir data\phase1 --seeds 0 1 2 --generations 5 --population-size 20
    .\.venv\Scripts\python.exe experiments\evaluate_baselines.py --data-dir data\phase1
    .\.venv\Scripts\python.exe -m pytest -q tests

生成的 data/phase1 是运行产物，不提交到 Git；统计和结果摘要见 docs/PHASE1_REPORT.md。
