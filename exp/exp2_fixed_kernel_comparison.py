"""Experiment 2: Fixed-training kernel comparison on an equity panel (Sec. 6.2-6.4).

Four PanelGP kernel pairs (affine/RBF loadings x constant/RBF factors) and two
IPCA benchmarks (with and without intercept) are estimated once on the
training window and evaluated out of sample, following Table 2 of the paper.

Run: python -m exp.exp2_fixed_kernel_comparison [--full]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from data.data_loader import load_ohlcv
from src import config as cfg
from src.metrics import holm_adjust, newey_west_diff_test
from src.pipeline import (
    average_seed_metrics, evaluate_ipca, evaluate_panelgp, fit_ipca_on_panel,
    fit_panelgp_seed, load_universe_and_panel, portfolio_series,
)
from src.universe import CANDIDATE_TICKERS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

KERNEL_PAIRS = [
    ("affine", "constant"), ("affine", "rbf"), ("rbf", "constant"), ("rbf", "rbf"),
]
SPEC_NAMES = {
    ("affine", "constant"): "affine_const", ("affine", "rbf"): "affine_rbf",
    ("rbf", "constant"): "rbf_const", ("rbf", "rbf"): "rbf_rbf",
}


def months_between(panel, start, end):
    return [m for m in panel.months if start <= m <= end]


def run(full: bool = False, h: int | None = None, seeds: int | None = None, tag: str = "exp2"):
    use_paper = full
    n_characteristics = cfg.APP_N_CHARACTERISTICS_PAPER if use_paper else cfg.APP_N_CHARACTERISTICS_RUN
    n_context = cfg.APP_N_CONTEXT_PAPER if use_paper else cfg.APP_N_CONTEXT_RUN
    n_inducing = cfg.APP_N_INDUCING_PAPER if use_paper else cfg.APP_N_INDUCING_RUN
    n_epochs = cfg.APP_N_EPOCHS_PAPER if use_paper else cfg.APP_N_EPOCHS_RUN
    universe_size = len(CANDIDATE_TICKERS) if use_paper else cfg.APP_UNIVERSE_SIZE_RUN
    train_start = cfg.APP_TRAIN_START_PAPER if use_paper else cfg.APP_TRAIN_START_RUN
    train_end = cfg.APP_TRAIN_END_PAPER if use_paper else cfg.APP_TRAIN_END_RUN
    valid_start = cfg.APP_VALID_START_PAPER if use_paper else cfg.APP_VALID_START_RUN
    valid_end = cfg.APP_VALID_END_PAPER if use_paper else cfg.APP_VALID_END_RUN
    eval_start = cfg.APP_EVAL_START_PAPER if use_paper else cfg.APP_EVAL_START_RUN
    eval_end = cfg.APP_EVAL_END_PAPER if use_paper else cfg.APP_EVAL_END_RUN
    n_factors = h if h is not None else cfg.APP_H_RUN
    n_seeds = seeds if seeds is not None else cfg.APP_N_SEEDS_RUN

    t0 = time.time()
    bars, meta = load_ohlcv(CANDIDATE_TICKERS, train_start, eval_end, timeframe="1d")
    print(f"[data] rows={len(bars)} tickers_loaded={len(meta['ticker_details'])} "
          f"failed={meta['failed']} non_massive_source={meta['non_massive_source']}")
    panel = load_universe_and_panel(
        bars, train_start, train_end, valid_start, valid_end, eval_start, eval_end,
        universe_size, n_characteristics, n_context,
    )
    print(f"[panel] months={len(panel.months)} train={len(panel.train_months)} "
          f"eval={len(panel.eval_months)} universe={len(panel.tickers)}")

    train_months = panel.train_months
    eval_months = panel.eval_months

    all_series = {}   # name -> {"portfolio": aggregated portfolio arrays, "scalar": averaged metrics}
    monthly_logscore = {}
    monthly_rankcorr = {}

    for loading_kernel, factor_kernel in KERNEL_PAIRS:
        name = SPEC_NAMES[(loading_kernel, factor_kernel)]
        seed_metrics, seed_scores, seed_rank = [], [], []
        last_ps = None
        for seed in range(n_seeds):
            fit, s_r = fit_panelgp_seed(
                panel, train_months, loading_kernel, factor_kernel, n_factors,
                n_inducing, n_epochs, seed,
            )
            per_month = evaluate_panelgp(fit, s_r, panel, eval_months)
            ps = portfolio_series(per_month)
            seed_metrics.append(ps)
            seed_scores.append(ps["log_scores_by_month"])
            last_ps = ps
            print(f"[fit] {name} seed={seed} rank_corr={ps['rank_correlation']:.4f} "
                  f"tangency_sr={ps['tangency_sharpe']:.3f} elapsed={time.time()-t0:.0f}s")
        avg = average_seed_metrics(seed_metrics)
        all_series[name] = {"scalar": avg, "last_portfolio": {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in last_ps.items()
            if k not in ("log_scores_by_month",)
        }}
        monthly_logscore[name] = np.mean(np.vstack(seed_scores), axis=0)
        monthly_rankcorr[name] = np.mean([m["rank_correlation"] for m in seed_metrics])

    for with_intercept, name in [(False, "ipca"), (True, "ipca_intercept")]:
        fit = fit_ipca_on_panel(panel, train_months, n_factors, with_intercept)
        per_month = evaluate_ipca(fit, panel.s_r, panel, eval_months)
        ps = portfolio_series(per_month)
        all_series[name] = {"scalar": {
            "rank_correlation_mean": ps["rank_correlation"], "rank_correlation_sd": 0.0,
            "log_score_mean": ps["log_score"], "log_score_sd": 0.0,
            "predictive_r2_mean": ps["predictive_r2"], "predictive_r2_sd": 0.0,
            "tangency_sharpe_mean": ps["tangency_sharpe"], "tangency_sharpe_sd": 0.0,
            "long_short_sharpe_mean": ps["long_short_sharpe"], "long_short_sharpe_sd": 0.0,
            "gmv_vol_mean": ps["gmv_vol"], "gmv_vol_sd": 0.0,
        }, "last_portfolio": {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in ps.items()
            if k not in ("log_scores_by_month",)
        }}
        monthly_logscore[name] = ps["log_scores_by_month"]
        print(f"[fit] {name} rank_corr={ps['rank_correlation']:.4f} "
              f"tangency_sr={ps['tangency_sharpe']:.3f} elapsed={time.time()-t0:.0f}s")

    # Pairwise Newey-West tests with Holm adjustment, 15 comparisons for log score.
    names = list(all_series.keys())
    pairs = list(itertools.combinations(names, 2))
    pvals = []
    for a, b in pairs:
        n = min(len(monthly_logscore[a]), len(monthly_logscore[b]))
        _, p = newey_west_diff_test(monthly_logscore[a][:n], monthly_logscore[b][:n], lags=12)
        pvals.append(p)
    pvals_adj = holm_adjust(pvals)
    pairwise_tests = [
        {"pair": [a, b], "p_value": pv, "p_value_holm": pa, "significant_5pct": pa < 0.05}
        for (a, b), pv, pa in zip(pairs, pvals, pvals_adj)
    ]

    result = {
        "config": {
            "full": use_paper, "n_factors": n_factors, "n_seeds": n_seeds,
            "n_inducing": n_inducing, "n_epochs": n_epochs, "universe_size": len(panel.tickers),
            "train_window": [train_start, train_end], "eval_window": [eval_start, eval_end],
            "n_train_months": len(train_months), "n_eval_months": len(eval_months),
        },
        "specifications": {k: v["scalar"] for k, v in all_series.items()},
        "pairwise_log_score_tests": pairwise_tests,
        "runtime_seconds": time.time() - t0,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{tag}_results.json"
    out_path.write_text(json.dumps(result, indent=2, default=float))
    print(f"[done] wrote {out_path} in {result['runtime_seconds']:.0f}s")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    run(full=args.full)
