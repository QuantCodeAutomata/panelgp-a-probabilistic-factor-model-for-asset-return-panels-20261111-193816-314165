"""Experiment 5: higher moments of the bilinear predictive distribution
(paper Section 6.6).

Using the annually re-estimated RBF/RBF, H=5 PanelGP forecasts of Experiment 4
(Section 6.5), this script computes the per-observation log-score gain of the
exact bilinear predictive density over the Gaussian density with the same
first two moments (Appendix B.3, Proposition 1), averaged over stock-months
and five seeds, and its breakdown by the size of the standardized forecast
error and by calendar month.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import src.config as cfg
from data.data_loader import load_ohlcv
from src.bilinear import gaussian_log_density, log_density, pairing_matrix
from src.pipeline import cached_rolling_panelgp, load_universe_and_panel, rolling_panelgp_forecasts
from src.universe import CANDIDATE_TICKERS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
ERROR_BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, np.inf)]


def months_between(panel, start, end):
    return [m for m in panel.months if start <= m <= end]


def per_observation_gain(records: list, H: int) -> dict:
    """Log-score gain ``g_it`` (Section 6.6) for every stock-month record of
    one rolling-forecast seed, plus its standardized error and month."""
    A = pairing_matrix(H)
    gains, abs_z, months = [], [], []
    for rec in records:
        m_beta = rec["m_beta"]            # (N, H), in scaled training units
        m_phi = rec["m_phi"].reshape(-1)  # (H,)
        SBB = rec["SBB_tilde"]            # (H, H, N)
        SFF = rec["SFF_tilde"]            # (H, H)
        SBP = rec["SBP"]                  # (H, H, N)
        s_r = rec["s_r"]
        r_scaled = rec["r"] / s_r
        mu = rec["mu"] / s_r
        var = np.clip(np.diag(rec["Sigma"]), 1e-12, None) / s_r ** 2
        N = m_beta.shape[0]
        m_star = np.concatenate([m_beta, np.broadcast_to(m_phi, (N, H))], axis=1)
        S_star = np.zeros((N, 2 * H, 2 * H))
        S_star[:, :H, :H] = np.moveaxis(SBB, -1, 0)
        S_star[:, H:, H:] = SFF[None, :, :]
        S_star[:, :H, H:] = np.moveaxis(SBP, -1, 0)
        S_star[:, H:, :H] = np.moveaxis(SBP, -1, 0).transpose(0, 2, 1)
        sigma_r2_scaled = rec["sigma_r2"] / s_r ** 2
        log_p_bil = log_density(r_scaled, m_star, S_star, A, sigma_r2_scaled) - np.log(s_r)
        log_p_gauss = gaussian_log_density(r_scaled, mu, var) - np.log(s_r)
        g = log_p_bil - log_p_gauss
        gains.append(g)
        abs_z.append(np.abs((rec["r"] - rec["mu"]) / np.sqrt(np.diag(rec["Sigma"]))))
        months.extend([rec["month"]] * N)
    return {
        "gain": np.concatenate(gains), "abs_z": np.concatenate(abs_z),
        "month": np.array(months),
    }


def run(full: bool = False, tag: str = "exp5"):
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

    # Same cache key convention as exp/exp4_rolling_and_trading_costs.py so the
    # rolling PanelGP forecasts computed there are reused here without refitting.
    cache_key = f"exp4_{universe_size}_{n_factors}_{n_epochs}_{n_seeds}_{window_months}"
    rolling_pg_seeds = cached_rolling_panelgp(
        cache_key,
        lambda: rolling_panelgp_forecasts(panel, all_months, eval_months, window_months,
                                           n_factors, n_inducing, n_epochs, n_seeds),
    )
    print(f"[rolling-panelgp] ready elapsed={time.time()-t0:.0f}s")

    all_gain, all_abs_z, all_month = [], [], []
    for seed_records in rolling_pg_seeds:
        out = per_observation_gain(seed_records, n_factors)
        all_gain.append(out["gain"]); all_abs_z.append(out["abs_z"]); all_month.append(out["month"])
    gain = np.concatenate(all_gain)
    abs_z = np.concatenate(all_abs_z)
    month = np.concatenate(all_month)

    bin_stats = []
    for lo, hi in ERROR_BINS:
        mask = (abs_z >= lo) & (abs_z < hi)
        bin_stats.append({
            "range": f"[{lo},{hi})", "n": int(mask.sum()),
            "mean_gain": float(np.mean(gain[mask])) if mask.any() else None,
            "contribution": float(np.sum(gain[mask]) / gain.size),
        })

    order = np.argsort(month)
    month_sorted, gain_sorted = month[order], gain[order]
    unique_months, inv = np.unique(month_sorted, return_inverse=True)
    monthly_mean = np.bincount(inv, weights=gain_sorted) / np.bincount(inv)

    result = {
        "config": {"full": use_paper, "n_factors": n_factors, "n_seeds": n_seeds,
                    "n_observations": int(gain.size)},
        "average_log_score_gain": float(np.mean(gain)),
        "error_bins": bin_stats,
        "monthly_mean_gain": {str(m): float(v) for m, v in zip(unique_months, monthly_mean)},
        "runtime_seconds": time.time() - t0,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{tag}_results.json"
    out_path.write_text(json.dumps(result, indent=2, default=float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    trailing = np.convolve(monthly_mean, np.ones(12) / 12, mode="valid")
    ax.plot(unique_months, monthly_mean, alpha=0.4, label="monthly mean")
    ax.plot(unique_months[11:], trailing, label="trailing 12-month average")
    ax.set_xlabel("month"); ax.set_ylabel("log-score gain")
    ax.set_title("Bilinear vs. Gaussian log-score gain (Exp. 5, Sec. 6.6)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{tag}_log_score_gain.png", dpi=150)
    plt.close(fig)

    print(f"[done] wrote {out_path} in {result['runtime_seconds']:.0f}s, "
          f"average gain={result['average_log_score_gain']:.5f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's full configuration")
    args = parser.parse_args()
    run(full=args.full)
