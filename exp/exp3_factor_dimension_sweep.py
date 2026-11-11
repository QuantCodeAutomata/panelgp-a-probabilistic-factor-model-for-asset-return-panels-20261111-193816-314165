"""Experiment 3: Effect of the number of factors H on model performance (Sec. 6.3, Fig. 3).

Sweeps H over the paper's grid (1..8, reduced here to a subset of that grid)
for all four PanelGP kernel pairs and the two IPCA benchmarks, re-using the
fixed-training pipeline of Experiment 2 for each value of H.

Run: python -m exp.exp3_factor_dimension_sweep [--full]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src import config as cfg
from exp.exp2_fixed_kernel_comparison import run as run_exp2

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run(full: bool = False):
    h_grid = cfg.APP_H_GRID_PAPER if full else cfg.APP_H_GRID_RUN
    t0 = time.time()
    all_h_results = {}
    for h in h_grid:
        print(f"[exp3] H={h} starting, elapsed={time.time()-t0:.0f}s")
        res = run_exp2(full=full, h=h, tag=f"exp3_H{h}")
        all_h_results[str(h)] = res["specifications"]
    out = {
        "config": {"full": full, "h_grid": h_grid},
        "by_h": all_h_results,
        "runtime_seconds": time.time() - t0,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp3_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] wrote {out_path} in {out['runtime_seconds']:.0f}s")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    run(full=args.full)
