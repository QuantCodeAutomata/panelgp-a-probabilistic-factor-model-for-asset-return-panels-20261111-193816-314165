"""Instrumented Principal Component Analysis benchmark (Kelly, Pruitt and Su,
2019-style alternating least squares), used as in paper Section 5.2 /
Appendix D: "Both estimated by alternating least squares."

Custom implementation: no maintained library offers this exact IPCA
alternating-least-squares estimator with the paper's with/without-intercept
variants, so it is implemented directly from the normal equations of the
bilinear characteristic-managed-portfolio model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IPCAFit:
    gamma: np.ndarray          # (P1, H) loadings on characteristics
    gamma_alpha: np.ndarray | None  # (P1,) static intercept loading, or None
    factors: np.ndarray        # (T, H) estimated factors f_t
    factor_mean: np.ndarray    # (H,) time-series average factor, used for forecasting
    factor_cov: np.ndarray     # (H, H) time-series covariance of factors
    resid_var: float           # sigma_e^2, average idiosyncratic residual variance


def _append_const(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=-1)


def fit_ipca(returns_by_month: list[np.ndarray], chars_by_month: list[np.ndarray],
             n_factors: int, with_intercept: bool,
             max_iter: int = 200, tol: float = 1e-6) -> IPCAFit:
    """Alternating least squares estimation of the IPCA model

        r_it = x_it' Gamma f_t [+ x_it' gamma_alpha] + e_it

    ``returns_by_month`` and ``chars_by_month`` hold one array per training
    month (N_t,) and (N_t, P), respectively; a constant column is appended
    internally to form [X_t, 1] as in the paper's loading definitions.
    """
    T = len(returns_by_month)
    H = n_factors
    X_list = [_append_const(x) for x in chars_by_month]
    P1 = X_list[0].shape[1]

    XtX = [X.T @ X for X in X_list]
    Xtr = [X.T @ r for X, r in zip(X_list, returns_by_month)]

    # Deterministic PCA initialization on characteristic-managed portfolios.
    Q = np.stack(Xtr, axis=0)  # (T, P1)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    gamma = Vt[:H].T.copy()  # (P1, H)
    gamma, _ = np.linalg.qr(gamma)
    gamma_alpha = None
    if with_intercept:
        sum_XtX = sum(XtX)
        sum_Xtr = sum(Xtr)
        scale0 = max(float(np.trace(sum_XtX)) / max(P1, 1), 1e-8)
        gamma_alpha = np.linalg.lstsq(sum_XtX + 1e-8 * scale0 * np.eye(P1), sum_Xtr, rcond=None)[0]

    factors = np.zeros((T, H))
    prev_gamma = gamma.copy()
    for iteration in range(max_iter):
        # F-step: cross-sectional regression per month.
        for t in range(T):
            resid = returns_by_month[t] - (X_list[t] @ gamma_alpha if with_intercept else 0.0)
            W = X_list[t] @ gamma  # (N_t, H)
            scale = max(float(np.trace(W.T @ W)) / max(H, 1), 1e-8)
            WtW = W.T @ W + 1e-6 * scale * np.eye(H)
            factors[t] = np.linalg.lstsq(WtW, W.T @ resid, rcond=None)[0]

        # Gamma-step (and gamma_alpha-step): pooled normal equations, accumulated per month.
        dim = P1 * H + (P1 if with_intercept else 0)
        AtA = np.zeros((dim, dim))
        Atr = np.zeros(dim)
        for t in range(T):
            f = factors[t]
            block_gg = np.kron(np.outer(f, f), XtX[t])          # (P1*H, P1*H)
            AtA[:P1 * H, :P1 * H] += block_gg
            Atr[:P1 * H] += np.kron(f, Xtr[t])
            if with_intercept:
                block_ga = np.kron(f.reshape(-1, 1), XtX[t])      # (P1*H, P1)
                AtA[:P1 * H, P1 * H:] += block_ga
                AtA[P1 * H:, :P1 * H] += block_ga.T
                AtA[P1 * H:, P1 * H:] += XtX[t]
                Atr[P1 * H:] += Xtr[t]
        diag_scale = max(float(np.trace(AtA)) / max(dim, 1), 1e-8)
        theta = np.linalg.lstsq(AtA + 1e-8 * diag_scale * np.eye(dim), Atr, rcond=None)[0]
        gamma = theta[:P1 * H].reshape(P1, H)
        if with_intercept:
            gamma_alpha = theta[P1 * H:]

        change = np.linalg.norm(gamma - prev_gamma) / (np.linalg.norm(prev_gamma) + 1e-12)
        prev_gamma = gamma.copy()
        if change < tol:
            break

    resid_sq, n_obs = 0.0, 0
    for t in range(T):
        pred = X_list[t] @ gamma @ factors[t]
        if with_intercept:
            pred = pred + X_list[t] @ gamma_alpha
        resid = returns_by_month[t] - pred
        resid_sq += float((resid ** 2).sum())
        n_obs += len(resid)
    resid_var = resid_sq / max(n_obs, 1)

    return IPCAFit(
        gamma=gamma, gamma_alpha=gamma_alpha, factors=factors,
        factor_mean=factors.mean(axis=0), factor_cov=np.cov(factors.T) if H > 1 else np.array([[factors.var()]]),
        resid_var=resid_var,
    )


def ipca_predict_month(fit: IPCAFit, x_month: np.ndarray) -> dict:
    """Forecast mean and covariance for one evaluation month (paper Section
    5.2): mu_hat = [X_t,1] Gamma f_bar, Sigma_hat = [X_t,1] Gamma Sigma_f
    Gamma' [X_t,1]' + sigma_e^2 I."""
    X = _append_const(x_month)
    loading = X @ fit.gamma  # (N, H)
    mu = loading @ fit.factor_mean
    if fit.gamma_alpha is not None:
        mu = mu + X @ fit.gamma_alpha
    Sigma = loading @ fit.factor_cov @ loading.T + fit.resid_var * np.eye(X.shape[0])
    return {"mu": mu, "Sigma": Sigma}
