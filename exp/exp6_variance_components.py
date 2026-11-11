"""Experiment 6: predicted-variance calibration and component analysis
(paper Section 6.6, Equations 26-27).

Using the annually re-estimated RBF/RBF, H=5 PanelGP forecasts of Experiment 4
(Section 6.5), decompose the predicted variance of every stock-month into the
factor-innovation term V^phi, the loading-innovation term V^beta, the shared
term tr(D_beta D_phi), and the observation-noise variance sigma_r^2, and
relate V^beta and V^phi to squared forecast errors by Spearman correlation and
by quintile comparisons of realized RMS error against predicted risk.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import src.config as cfg
from data.data_loader import load_ohlcv
from src.pipeline import cached_rolling_panelgp, load_universe_and_panel, rolling_panelgp_forecasts
from src.universe import CANDIDATE_TICKERS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def months_between(panel, start, end):
    return [m for m in panel.months if start <= m <= end]


def quintile_table(group_key: np.ndarray, sq_err: np.ndarray, pred_var: np.ndarray,
                    weight_by_month: np.ndarray | None = None) -> list:
    """RMS realized error vs. predicted risk (sqrt variance) by quintile of
    ``group_key`` (Section 6.6: temporal quintiles of V^beta weight months
    equally; cross-sectional quintiles of V^phi pool stock-months)."""
    order = np.argsort(group_key)
    n = len(group_key)
    edges = np.linspace(0, n, 6).astype(int)
    out = []
    for q in range(5):
        idx = order[edges[q]:edges[q + 1]]
        if weight_by_month is None:
            rms_err = float(np.sqrt(np.mean(sq_err[idx])))
            rms_pred = float(np.sqrt(np.mean(pred_var[idx])))
        else:
            w = weight_by_month[idx]
            rms_err = float(np.sqrt(np.average(sq_err[idx], weights=w)))
            rms_pred = float(np.sqrt(np.average(pred_var[idx], weights=w)))
        out.append({"quintile": q + 1, "n": int(len(idx)),
                    "realized_rms_error": rms_err, "predicted_risk": rms_pred})
    return out


def run(full: bool = False, tag: str = "exp6"):
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
    eval_months = months_between(panel, eval_start, eval_end)

    cache_key = f"exp4_{universe_size}_{n_factors}_{n_epochs}_{n_seeds}_{window_months}"
    rolling_pg_seeds = cached_rolling_panelgp(
        cache_key,
        lambda: rolling_panelgp_forecasts(panel, all_months, eval_months, window_months,
                                           n_factors, n_inducing, n_epochs, n_seeds),
    )
    print(f"[rolling-panelgp] ready elapsed={time.time()-t0:.0f}s")

    rho_beta_list, rho_phi_list = [], []
    avg_components = {"V_phi": [], "V_beta": [], "tr_DbDf": [], "sigma_r2": []}
    quint_temporal_seeds, quint_cross_seeds = [], []
    for seed_records in rolling_pg_seeds:
        month_Vbeta, month_sq_err_avg = [], []
        cross_Vphi, cross_sq_err, cross_month_weight = [], [], []
        for rec in seed_records:
            e2 = (rec["r"] - rec["mu"]) ** 2
            V_phi = rec["V_phi"]            # (N,) varies across stocks
            V_beta = float(np.mean(rec["V_beta"])) if np.ndim(rec["V_beta"]) else float(rec["V_beta"])
            month_Vbeta.append(V_beta)
            month_sq_err_avg.append(float(np.mean(e2)))
            cross_Vphi.append(V_phi)
            cross_sq_err.append(e2)
            cross_month_weight.append(np.full(e2.shape[0], 1.0 / e2.shape[0]))
            avg_components["V_phi"].append(np.mean(V_phi))
            avg_components["V_beta"].append(V_beta)
            avg_components["tr_DbDf"].append(float(np.mean(rec["tr_DbDf"])))
            avg_components["sigma_r2"].append(float(np.mean(rec["sigma_r2"])))
            rho, _ = spearmanr(V_phi, e2)
            if np.isfinite(rho):
                rho_phi_list.append(rho)
        rho_b, _ = spearmanr(month_Vbeta, month_sq_err_avg)
        if np.isfinite(rho_b):
            rho_beta_list.append(rho_b)

        pred_var_by_month = [rec["Sigma"].diagonal() for rec in seed_records]
        quint_temporal_seeds.append(quintile_table(
            np.array(month_Vbeta),
            np.array(month_sq_err_avg),
            np.array([np.mean(v) for v in pred_var_by_month]),
        ))
        vphi_all = np.concatenate(cross_Vphi)
        sq_err_all = np.concatenate(cross_sq_err)
        pred_var_all = np.concatenate(pred_var_by_month)
        quint_cross_seeds.append(quintile_table(vphi_all, sq_err_all, pred_var_all))

    def avg_quintiles(seed_tables):
        out = []
        for q in range(5):
            out.append({
                "quintile": q + 1,
                "realized_rms_error": float(np.mean([t[q]["realized_rms_error"] for t in seed_tables])),
                "predicted_risk": float(np.mean([t[q]["predicted_risk"] for t in seed_tables])),
            })
        return out

    result = {
        "config": {"full": use_paper, "n_factors": n_factors, "n_seeds": n_seeds},
        "spearman_V_beta_vs_monthly_sq_error": float(np.mean(rho_beta_list)),
        "spearman_V_phi_vs_sq_error_within_month": float(np.mean(rho_phi_list)),
        "average_components": {k: float(np.mean(v)) for k, v in avg_components.items()},
        "temporal_quintiles_by_V_beta": avg_quintiles(quint_temporal_seeds),
        "cross_sectional_quintiles_by_V_phi": avg_quintiles(quint_cross_seeds),
        "runtime_seconds": time.time() - t0,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{tag}_results.json"
    out_path.write_text(json.dumps(result, indent=2, default=float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, table, title in zip(
        axes, [result["temporal_quintiles_by_V_beta"], result["cross_sectional_quintiles_by_V_phi"]],
        ["Temporal quintiles (V^beta)", "Cross-sectional quintiles (V^phi)"],
    ):
        q = [t["quintile"] for t in table]
        ax.plot(q, [t["realized_rms_error"] for t in table], marker="o", label="realized RMS error")
        ax.plot(q, [t["predicted_risk"] for t in table], marker="s", label="predicted risk")
        ax.set_yscale("log"); ax.set_xlabel("quintile"); ax.set_title(title); ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{tag}_variance_calibration.png", dpi=150)
    plt.close(fig)

    print(f"[done] wrote {out_path} in {result['runtime_seconds']:.0f}s")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    run(full=args.full)
