"""Phase 1.5C-R1 runtime benchmark for representative baseline runs."""
from __future__ import annotations
import json
import time
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vetm.nsga2 import NSGA2Config, run_nsga2
from vetm.problems import get_problem

def main():
    problem = get_problem("ZDT1", n_var=20, n_obj=2)
    rows = []
    for budget in (2000, 5000, 10000):
        cfg = NSGA2Config(population_size=50, generations=budget // 50 - 1)
        start = time.perf_counter()
        result = run_nsga2(problem, cfg, seed=0)
        seconds = time.perf_counter() - start
        rows.append({"budget": budget, "seconds": seconds,
                     "evaluations": result["function_evaluations"],
                     "evaluations_per_second": result["function_evaluations"] / seconds})
    output = {"problem": "ZDT1_n20_m2", "rows": rows,
              "estimated_7200_runs_seconds": float(7200 * sum(row["seconds"] for row in rows) / len(rows))}
    (ROOT / "results" / "phase1_5C_R1_runtime_benchmark.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
