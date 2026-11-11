"""Shared evaluation metrics: predictive R^2, rank correlation, Gaussian log
score, portfolio construction, and Newey-West inference (paper Section 5.2 /
Appendix D). Spearman correlation and the Newey-West long-run variance use
scipy.stats and statsmodels, which implement these standard estimators;
portfolio formulas and the Gaussian log score are coded from the paper's
equations since no library matches this exact bilinear-GP evaluation design.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def function_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 = 1 - SS_res/SS_tot for recovering a deterministic function on a
    grid (paper Section 5.1)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def predictive_r2(residuals: np.ndarray, realized: np.ndarray) -> float:
    """Predictive R^2 = 1 - sum(residual^2) / sum(realized^2), paper Sec. 5.2
    (no centering, since realized excess returns have mean close to zero)."""
    return float(1.0 - np.sum(residuals ** 2) / np.sum(realized ** 2))


def monthly_rank_correlation_series(pred_by_month: list[np.ndarray], real_by_month: list[np.ndarray]) -> np.ndarray:
    """Monthly Spearman rank correlation between predicted and realized
    returns, one value per month (paper Section 5.2)."""
    vals = []
    for p, r in zip(pred_by_month, real_by_month):
        if len(p) < 2:
            vals.append(np.nan)
            continue
        rho, _ = spearmanr(p, r)
        vals.append(rho)
    return np.array(vals, dtype=float)


def monthly_rank_correlation(pred_by_month: list[np.ndarray], real_by_month: list[np.ndarray]) -> float:
    """Average monthly Spearman rank correlation between predicted and
    realized returns (paper Section 5.2)."""
    vals = monthly_rank_correlation_series(pred_by_month, real_by_month)
    return float(np.nanmean(vals))


def gaussian_log_score(resid: np.ndarray, Sigma: np.ndarray) -> float:
    """Per-observation Gaussian log score for one month (paper Section 5.2):
    ell_t = -(N_t log(2 pi) + log|Sigma_t| + resid^T Sigma_t^{-1} resid) / (2 N_t).
    """
    n = resid.shape[0]
    sign, logdet = np.linalg.slogdet(Sigma)
    sol = np.linalg.solve(Sigma, resid)
    quad = float(resid @ sol)
    return float(-(n * np.log(2 * np.pi) + logdet + quad) / (2 * n))


def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """Global minimum variance weights w = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)."""
    ones = np.ones(Sigma.shape[0])
    inv_ones = np.linalg.solve(Sigma, ones)
    return inv_ones / inv_ones.sum()


def tangency_weights(Sigma: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Tangency weights w = Sigma^{-1} mu / ||Sigma^{-1} mu||_1 (unit gross exposure)."""
    inv_mu = np.linalg.solve(Sigma, mu)
    gross = np.sum(np.abs(inv_mu))
    return inv_mu / gross


def long_short_weights(mu: np.ndarray, gross_each_side: float = 0.5) -> np.ndarray:
    """Equal-weighted top-decile minus bottom-decile portfolio with gross
    exposure ``gross_each_side`` on each side (paper Section 5.2)."""
    n = len(mu)
    n_decile = max(1, n // 10)
    order = np.argsort(mu)
    weights = np.zeros(n)
    short_idx = order[:n_decile]
    long_idx = order[-n_decile:]
    weights[long_idx] = gross_each_side / n_decile
    weights[short_idx] = -gross_each_side / n_decile
    return weights


def annualized_sharpe(monthly_returns: np.ndarray) -> float:
    mean, std = monthly_returns.mean(), monthly_returns.std(ddof=1)
    return float(np.sqrt(12) * mean / std) if std > 0 else float("nan")


def annualized_vol(monthly_returns: np.ndarray) -> float:
    return float(np.sqrt(12) * monthly_returns.std(ddof=1))


def newey_west_se(x: np.ndarray, lags: int = 12) -> float:
    """Newey-West standard error of the sample mean of ``x`` (Bartlett
    kernel, paper Appendix D: "pairwise Newey-West tests with 12 lags")."""
    from statsmodels.stats.sandwich_covariance import cov_hac
    import statsmodels.api as sm
    n = len(x)
    xc = x - x.mean()
    model = sm.OLS(x, np.ones(n))
    res = model.fit()
    cov = cov_hac(res, nlags=lags)
    return float(np.sqrt(cov[0, 0]))


def newey_west_diff_test(x: np.ndarray, y: np.ndarray, lags: int = 12) -> tuple[float, float]:
    """Newey-West t-test for a zero mean difference d = x - y. Returns
    (t-statistic, two-sided p-value)."""
    from scipy.stats import norm
    d = x - y
    se = newey_west_se(d, lags=lags)
    if se == 0 or not np.isfinite(se):
        return 0.0, 1.0
    t_stat = d.mean() / se
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))
    return float(t_stat), float(p_value)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment for a family of p-values."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted.tolist()
