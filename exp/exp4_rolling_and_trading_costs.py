"""Experiment 4: Annual re-estimation and trading-cost analysis (Sec. 6.5, Tables 3-4).

Compares fixed-training and annually rolling re-estimation of PanelGP
(RBF/RBF, H=5) against both IPCA benchmarks, then applies the paper's
proportional trading-cost model to the tangency and long-short portfolios.

Run: python -m exp.exp4_rolling_and_trading_costs [--full]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data.data_loader import load_ohlcv
from src import config as cfg
from src.metrics import (
    annualized_sharpe, gmv_weights, holm_adjust, long_short_weights,
    newey_west_diff_test, tangency_weights,
)
from src.pipeline import (
    break_even_cost_bps, cached_rolling_panelgp, evaluate_ipca, evaluate_panelgp,
    fit_ipca_on_panel, fit_panelgp_seed, load_universe_and_panel, portfolio_series,
    rolling_ipca_forecasts, rolling_panelgp_forecasts, trading_cost_series,
)
from src.universe import CANDIDATE_TICKERS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def months_between(panel, start, end):
    return [m for m in panel.months if start <= m <= end]


def run(full: bool = False, tag: str = "exp4"):
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
    window_months = cfg.APP_WINDOW_MONTHS_PAPER if use_paper else cfg.APP_WINDOW_MONTHS_RUN
    n_factors = cfg.APP_H_RUN
    n_seeds = cfg.APP_N_SEEDS_RUN

    t0 = time.time()
    bars, meta = load_ohlcv(CANDIDATE_TICKERS, train_start, eval_end, timeframe="1d")
    panel = load_universe_and_panel(
        bars, train_start, train_end, valid_start, valid_end, eval_start, eval_end,
        universe_size, n_characteristics, n_context,
    )
    all_months = panel.months
    train_months = panel.train_months
    eval_months = panel.eval_months
    print(f"[panel] months={len(all_months)} train={len(train_months)} eval={len(eval_months)}")

    # --- Fixed training ---
    fixed_pg_seeds = []
    for s in range(n_seeds):
        fit, s_r = fit_panelgp_seed(panel, train_months, "rbf", "rbf", n_factors, n_inducing, n_epochs, s)
        fixed_pg_seeds.append(evaluate_panelgp(fit, s_r, panel, eval_months))
        print(f"[fixed-panelgp] seed={s} elapsed={time.time()-t0:.0f}s")
    fixed_ipca = evaluate_ipca(fit_ipca_on_panel(panel, train_months, n_factors, False), panel.s_r, panel, eval_months)
    fixed_ipca_int = evaluate_ipca(fit_ipca_on_panel(panel, train_months, n_factors, True), panel.s_r, panel, eval_months)

    # --- Rolling re-estimation (Section 6.5); cached so exp5/exp6 reuse it ---
    cache_key = f"{tag}_{universe_size}_{n_factors}_{n_epochs}_{n_seeds}_{window_months}"
    rolling_pg_seeds = cached_rolling_panelgp(
        cache_key,
        lambda: rolling_panelgp_forecasts(panel, all_months, eval_months, window_months,
                                           n_factors, n_inducing, n_epochs, n_seeds),
    )
    print(f"[rolling-panelgp] done elapsed={time.time()-t0:.0f}s")
    rolling_ipca = rolling_ipca_forecasts(panel, all_months, eval_months, window_months,
                                           n_factors, False)
    rolling_ipca_int = rolling_ipca_forecasts(panel, all_months, eval_months, window_months,
                                               n_factors, True)

    specs = {
        "panelgp_fixed": fixed_pg_seeds, "panelgp_rolling": rolling_pg_seeds,
        "ipca_fixed": [fixed_ipca], "ipca_rolling": [rolling_ipca],
        "ipca_intercept_fixed": [fixed_ipca_int], "ipca_intercept_rolling": [rolling_ipca_int],
    }
    summary = {}
    monthly_logscore = {}
    for name, seed_list in specs.items():
        seed_ps = [portfolio_series(pm) for pm in seed_list]
        avg = {k + "_mean": float(np.mean([p[k] for p in seed_ps]))
               for k in ("rank_correlation", "log_score", "predictive_r2",
                         "tangency_sharpe", "long_short_sharpe", "gmv_vol")}
        avg.update({k + "_sd": float(np.std([p[k] for p in seed_ps], ddof=1)) if len(seed_ps) > 1 else 0.0
                    for k in ("rank_correlation", "log_score", "predictive_r2",
                              "tangency_sharpe", "long_short_sharpe", "gmv_vol")})
        summary[name] = avg
        monthly_logscore[name] = np.mean(np.vstack([p["log_scores_by_month"] for p in seed_ps]), axis=0)

    # Subperiod split (Table S3): first half vs second half of the evaluation window.
    mid = eval_months[len(eval_months) // 2]
    subperiods = {}
    for name, seed_list in specs.items():
        early = [[r for r in pm if r["month"] < mid] for pm in seed_list]
        late = [[r for r in pm if r["month"] >= mid] for pm in seed_list]
        early_ps = [portfolio_series(pm) for pm in early if len(pm) > 2]
        late_ps = [portfolio_series(pm) for pm in late if len(pm) > 2]
        subperiods[name] = {
            "early": {k: float(np.mean([p[k] for p in early_ps])) for k in
                      ("tangency_sharpe", "long_short_sharpe", "rank_correlation")} if early_ps else {},
            "late": {k: float(np.mean([p[k] for p in late_ps])) for k in
                     ("tangency_sharpe", "long_short_sharpe", "rank_correlation")} if late_ps else {},
        }

    # Trading costs for rolling PanelGP and rolling IPCA+intercept (Table 4).
    trading_costs = {}
    for name, seed_list in [("panelgp_rolling", rolling_pg_seeds), ("ipca_intercept_rolling", [rolling_ipca_int])]:
        tan_tc, ls_tc, be_tan, be_ls = [], [], [], []
        for pm in seed_list:
            tan = trading_cost_series(pm, lambda r: tangency_weights(r["Sigma"], r["mu"]), cost_bps=cfg.TRADING_COST_BPS)
            ls = trading_cost_series(pm, lambda r: long_short_weights(r["mu"]), cost_bps=cfg.TRADING_COST_BPS)
            tan_tc.append(tan)
            ls_tc.append(ls)
            be_tan.append(break_even_cost_bps(pm, lambda r: tangency_weights(r["Sigma"], r["mu"])))
            be_ls.append(break_even_cost_bps(pm, lambda r: long_short_weights(r["mu"])))
        trading_costs[name] = {
            "tangency": {
                "avg_turnover": float(np.mean([np.mean(t["turnover"]) for t in tan_tc])),
                "gross_sharpe": float(np.mean([annualized_sharpe(t["gross_returns"]) for t in tan_tc])),
                "net_sharpe": float(np.mean([annualized_sharpe(t["net_returns"]) for t in tan_tc])),
                "break_even_bps": float(np.mean(be_tan)),
            },
            "long_short": {
                "avg_turnover": float(np.mean([np.mean(t["turnover"]) for t in ls_tc])),
                "gross_sharpe": float(np.mean([annualized_sharpe(t["gross_returns"]) for t in ls_tc])),
                "net_sharpe": float(np.mean([annualized_sharpe(t["net_returns"]) for t in ls_tc])),
                "break_even_bps": float(np.mean(be_ls)),
            },
        }

    names = list(specs.keys())
    pairs_of_interest = [("panelgp_fixed", "panelgp_rolling"), ("panelgp_rolling", "ipca_intercept_rolling")]
    pvals = []
    for a, b in pairs_of_interest:
        n = min(len(monthly_logscore[a]), len(monthly_logscore[b]))
        _, p = newey_west_diff_test(monthly_logscore[a][:n], monthly_logscore[b][:n], lags=12)
        pvals.append(p)
    pvals_adj = holm_adjust(pvals)

    result = {
        "config": {
            "full": use_paper, "n_factors": n_factors, "n_seeds": n_seeds,
            "window_months": window_months, "cost_bps": cfg.TRADING_COST_BPS,
        },
        "specifications": summary,
        "subperiods": subperiods,
        "trading_costs": trading_costs,
        "pairwise_tests": [
            {"pair": list(p), "p_value": pv, "p_value_holm": pa}
            for p, pv, pa in zip(pairs_of_interest, pvals, pvals_adj)
        ],
        "runtime_seconds": time.time() - t0,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{tag}_results.json"
    out_path.write_text(json.dumps(result, indent=2, default=float))
    print(f"[done] wrote {out_path} in {result['runtime_seconds']:.0f}s")
    return result, rolling_pg_seeds, rolling_ipca_int


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    run(full=args.full)
