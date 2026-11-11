"""Shared data/model/evaluation pipeline for Experiments 2-4 (Section 6).

Every application experiment script calls these functions with the same
signatures so the universe, panel construction, PanelGP/IPCA estimation and
evaluation metrics are computed identically across experiments.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import MINIBATCH_ROWS
from src.ipca_model import fit_ipca, ipca_predict_month
from src.metrics import (
    annualized_sharpe, annualized_vol, gaussian_log_score, gmv_weights,
    long_short_weights, monthly_rank_correlation, predictive_r2, tangency_weights,
)
from src.panel_data import PanelData, build_panel_dataset, stack_training_rows
from src.panelgp import PanelGPConfig, PanelGPModel
from src.prediction import fit_summary, predict_month
from src.training import train_panelgp
from src.universe import CANDIDATE_TICKERS, select_liquid_universe


def load_universe_and_panel(
    bars, train_start: str, train_end: str, valid_start: str, valid_end: str,
    eval_start: str, eval_end: str, universe_size: int,
    n_characteristics: int, n_context: int,
) -> PanelData:
    """Build the fixed cross-section (liquidity-ranked over the first
    training year) and the month-indexed characteristics/context/return
    panel used by every specification in Experiments 2-4."""
    first_year_end = str(np.datetime64(train_start) + np.timedelta64(365, "D"))[:10]
    tickers = select_liquid_universe(bars, train_start, first_year_end, universe_size)
    return build_panel_dataset(
        bars, tickers, train_start, train_end, valid_start, valid_end,
        eval_start, eval_end, n_characteristics, n_context,
    )


@dataclass
class SpecResult:
    name: str
    model_type: str          # "panelgp" or "ipca"
    seed_results: list        # list of per-seed per-month prediction dicts


def fit_panelgp_seed(
    panel: PanelData, train_months: list, loading_kernel: str, factor_kernel: str,
    n_factors: int, n_inducing: int, n_epochs: int, seed: int,
):
    """Train one PanelGP specification/seed on ``train_months`` and return a
    frozen ``FittedPanelGP`` plus the training return scale ``s_r``."""
    data = stack_training_rows(panel, train_months)
    cfg = PanelGPConfig(
        n_characteristics=panel.n_characteristics, n_context=panel.n_context,
        n_factors=n_factors, n_inducing=n_inducing,
        loading_kernel=loading_kernel, factor_kernel=factor_kernel,
        isotropic_innovations=False,
    )
    model = PanelGPModel(cfg, train_x=data["x"], train_c=data["c_seq"], seed=seed)
    train_panelgp(
        model, x=data["x"], t_idx=data["t_idx"], r=data["r"],
        n_epochs=n_epochs, minibatch_rows=MINIBATCH_ROWS, seed=seed, log_every=0,
    )
    fit = fit_summary(model)
    return fit, panel.s_r


def evaluate_panelgp(fit, s_r: float, panel: PanelData, eval_months: list) -> list:
    """Predictive mean/covariance/variance-decomposition for every month in
    ``eval_months``, in original return units (Eq. 19-27)."""
    out = []
    for m in eval_months:
        X = panel.chars_by_month[m]
        c = panel.context_by_month[m].reshape(1, -1)
        r = panel.returns_by_month[m]
        pred = predict_month(fit, X, c)
        out.append({
            "month": m, "tickers": panel.tickers_active_by_month[m],
            "mu": s_r * pred["mu"], "Sigma": (s_r ** 2) * pred["Sigma"], "r": r,
            "V0": (s_r ** 2) * pred["V0"], "V_phi": (s_r ** 2) * pred["V_phi"],
            "V_beta": (s_r ** 2) * pred["V_beta"], "tr_DbDf": (s_r ** 2) * pred["tr_DbDf"],
            "sigma_r2": (s_r ** 2) * pred["sigma_r2"],
            "m_beta": pred["m_beta"], "m_phi": pred["m_phi"],
            "SBB_tilde": pred["SBB_tilde"], "SFF_tilde": pred["SFF_tilde"], "SBP": pred["SBP"],
            "s_r": s_r,
        })
    return out


def fit_ipca_on_panel(panel: PanelData, train_months: list, n_factors: int, with_intercept: bool):
    rets = [panel.returns_by_month[m] / panel.s_r for m in train_months]
    chars = [panel.chars_by_month[m] for m in train_months]
    # Only characteristics for the tickers with a valid training return are used.
    chars = [c[:r.shape[0]] if c.shape[0] != r.shape[0] else c for c, r in zip(chars, rets)]
    fit = fit_ipca(rets, chars, n_factors=n_factors, with_intercept=with_intercept)
    return fit


def evaluate_ipca(fit, s_r: float, panel: PanelData, eval_months: list) -> list:
    out = []
    for m in eval_months:
        X = panel.chars_by_month[m]
        r = panel.returns_by_month[m]
        pred = ipca_predict_month(fit, X)
        out.append({
            "month": m, "tickers": panel.tickers_active_by_month[m],
            "mu": s_r * pred["mu"], "Sigma": (s_r ** 2) * pred["Sigma"], "r": r,
        })
    return out


def portfolio_series(per_month: list) -> dict:
    """Monthly tangency, GMV and long-short portfolio returns, plus monthly
    rank correlation, log score and predictive-R^2 components (Sec. 5.2/6)."""
    tan_rets, gmv_rets, ls_rets = [], [], []
    rank_pred, rank_real, log_scores = [], [], []
    resid_all, real_all = [], []
    months = []
    for rec in per_month:
        mu, Sigma, r = rec["mu"], rec["Sigma"], rec["r"]
        if len(r) < 3:
            continue
        w_tan = tangency_weights(Sigma, mu)
        w_gmv = gmv_weights(Sigma)
        w_ls = long_short_weights(mu)
        tan_rets.append(float(w_tan @ r))
        gmv_rets.append(float(w_gmv @ r))
        ls_rets.append(float(w_ls @ r))
        rank_pred.append(mu)
        rank_real.append(r)
        log_scores.append(gaussian_log_score(r - mu, Sigma))
        resid_all.append(r - mu)
        real_all.append(r)
        months.append(rec["month"])
    resid_all = np.concatenate(resid_all)
    real_all = np.concatenate(real_all)
    return {
        "months": months,
        "tangency_returns": np.array(tan_rets), "gmv_returns": np.array(gmv_rets),
        "long_short_returns": np.array(ls_rets),
        "rank_correlation": monthly_rank_correlation(rank_pred, rank_real),
        "log_score": float(np.mean(log_scores)), "log_scores_by_month": np.array(log_scores),
        "predictive_r2": predictive_r2(resid_all, real_all),
        "tangency_sharpe": annualized_sharpe(np.array(tan_rets)),
        "long_short_sharpe": annualized_sharpe(np.array(ls_rets)),
        "gmv_vol": annualized_vol(np.array(gmv_rets)),
    }


def trading_cost_series(per_month: list, weight_fn, cost_bps: float = 10.0, risk_free: float = 0.0) -> dict:
    """Turnover and net-of-cost portfolio returns for one weight rule, using
    the drift-then-rebalance identity of Section 6.5:

        w~_{i,t-1} = w_{i,t-1} (1 + r_f + r_{i,t}) / (1 + r_f + w_{t-1}^T r_t)
        TO_t = sum_i |w_{i,t} - w~_{i,t-1}|
        r^net_{p,t+1} = w_t^T r_{t+1} - c * TO_t

    The risk-free rate is held at zero (documented deviation): no funding-rate
    series is available from the data path used here.
    """
    cost = cost_bps / 10000.0
    prev_w: dict = {}
    turnovers, gross_rets, net_rets = [], [], []
    for rec in per_month:
        tickers = rec["tickers"]
        w = weight_fn(rec)
        w_dict = dict(zip(tickers, w))
        r_dict = dict(zip(tickers, rec["r"]))
        if prev_w:
            port_r_prev = sum(prev_w.get(t, 0.0) * r_dict.get(t, 0.0) for t in prev_w)
            denom = 1.0 + risk_free + port_r_prev
            drifted = {t: wt * (1.0 + risk_free + r_dict.get(t, 0.0)) / denom for t, wt in prev_w.items()}
            all_t = set(w_dict) | set(drifted)
            turnover = sum(abs(w_dict.get(t, 0.0) - drifted.get(t, 0.0)) for t in all_t)
        else:
            turnover = sum(abs(v) for v in w_dict.values())
        gross = float(w @ rec["r"])
        turnovers.append(turnover)
        gross_rets.append(gross)
        net_rets.append(gross - cost * turnover)
        prev_w = w_dict
    return {
        "turnover": np.array(turnovers), "gross_returns": np.array(gross_rets),
        "net_returns": np.array(net_rets),
    }


def annual_windows(eval_months: list, window_months: int) -> dict:
    """Map each calendar year in ``eval_months`` to the first evaluation
    month of that year, used to decide when to re-fit (Section 6.5)."""
    years = sorted({m.year for m in eval_months})
    return {y: (min(m for m in eval_months if m.year == y),
                [m for m in eval_months if m.year == y]) for y in years}


def rolling_panelgp_forecasts(panel: PanelData, all_months: list, eval_months: list,
                               window_months: int, n_factors: int, n_inducing: int,
                               n_epochs: int, n_seeds: int) -> list:
    """Annual re-estimation of the RBF/RBF PanelGP model (Section 6.5): for
    every calendar year in ``eval_months``, fit on the latest ``window_months``
    input months before that year and forecast every month of the year."""
    windows = annual_windows(eval_months, window_months)
    per_seed_months: list = [[] for _ in range(n_seeds)]
    for year, (first_month, year_months) in sorted(windows.items()):
        idx = all_months.index(first_month)
        train_window = all_months[max(0, idx - window_months):idx]
        if len(train_window) < window_months // 2:
            continue
        for s in range(n_seeds):
            fit, s_r = fit_panelgp_seed(panel, train_window, "rbf", "rbf", n_factors,
                                         n_inducing, n_epochs, s)
            per_seed_months[s].extend(evaluate_panelgp(fit, s_r, panel, year_months))
    return per_seed_months


def rolling_ipca_forecasts(panel: PanelData, all_months: list, eval_months: list,
                            window_months: int, n_factors: int, with_intercept: bool) -> list:
    """Annual re-estimation of the IPCA benchmark, same windows as above."""
    windows = annual_windows(eval_months, window_months)
    out: list = []
    for year, (first_month, year_months) in sorted(windows.items()):
        idx = all_months.index(first_month)
        train_window = all_months[max(0, idx - window_months):idx]
        if len(train_window) < window_months // 2:
            continue
        fit = fit_ipca_on_panel(panel, train_window, n_factors, with_intercept)
        out.extend(evaluate_ipca(fit, panel.s_r, panel, year_months))
    return out


def cached_rolling_panelgp(cache_key: str, builder) -> list:
    """Reuse the annually re-estimated PanelGP forecasts (Section 6.5) across
    Experiments 4-6, which all evaluate the same rolling RBF/RBF, H=5 model.
    Cached outside the repository (``data/cache``); recomputed if absent."""
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"rolling_panelgp_{cache_key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    result = builder()
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def break_even_cost_bps(per_month: list, weight_fn) -> float:
    """Trading cost (in basis points per dollar traded) at which the average
    net excess return of the portfolio is zero."""
    tc = trading_cost_series(per_month, weight_fn, cost_bps=0.0)
    avg_gross = float(np.mean(tc["gross_returns"]))
    avg_turnover = float(np.mean(tc["turnover"]))
    if avg_turnover <= 0:
        return float("nan")
    return 10000.0 * avg_gross / avg_turnover


def average_seed_metrics(seed_metrics: list) -> dict:
    """Mean and standard deviation across seeds for each scalar metric."""
    keys = ["rank_correlation", "log_score", "predictive_r2", "tangency_sharpe",
            "long_short_sharpe", "gmv_vol"]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in seed_metrics])
        out[k + "_mean"] = float(np.mean(vals))
        out[k + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out
