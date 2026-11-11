"""Simulated panels for the PanelGP recovery study (paper Section 5.1).

Generates a balanced panel of 80 assets and 50 months from the model's own
data-generating process, with the smooth loading and factor maps given in
Section 5.1 and isotropic innovation/noise variances of 0.05.
"""

import numpy as np

from src.config import SIM_N_ASSETS, SIM_N_MONTHS, SIM_H, SIM_SIGMA_BETA2, SIM_SIGMA_PHI2, SIM_SIGMA_R2


def characteristic_grid(n_assets: int = SIM_N_ASSETS) -> np.ndarray:
    """Asset characteristics x_i on a regular 9x9 grid in [0,1]^2 with the
    upper-right corner omitted, giving 80 points (paper Section 5.1)."""
    side = 9
    grid = np.array([[i / (side - 1), j / (side - 1)] for i in range(side) for j in range(side)])
    grid = grid[~((grid[:, 0] == 1.0) & (grid[:, 1] == 1.0))]  # drop the upper-right corner
    assert grid.shape[0] == n_assets, f"expected {n_assets} grid points, got {grid.shape[0]}"
    return grid


def context_sequence(n_months: int = SIM_N_MONTHS) -> np.ndarray:
    """Scalar context c_t = (t-1)/(T-1) for t = 1..T (paper Section 5.1)."""
    return np.arange(n_months) / (n_months - 1)


def true_phi(c: np.ndarray) -> np.ndarray:
    """True smooth factor maps phi_1(c), phi_2(c) (paper Section 5.1)."""
    phi1 = 0.5 + np.sin(2 * np.pi * c) + 0.5 * np.cos(4 * np.pi * c) + (1 / 3) * np.sin(6 * np.pi * c)
    phi2 = -0.5 + np.cos(2 * np.pi * c) - 0.5 * np.sin(4 * np.pi * c) + (1 / 3) * np.cos(6 * np.pi * c)
    return np.stack([phi1, phi2], axis=-1)


def true_beta(x: np.ndarray) -> np.ndarray:
    """True smooth loading maps beta_1(x), beta_2(x) (paper Section 5.1)."""
    x1, x2 = x[..., 0], x[..., 1]
    beta1 = np.sin(2 * np.pi * x1) * np.cos(np.pi * x2) + 0.5 * np.cos(2 * np.pi * (x1 + x2)) \
        + (1 / 3) * np.sin(4 * np.pi * x1) * np.sin(2 * np.pi * x2)
    beta2 = np.cos(2 * np.pi * x2) * np.sin(np.pi * x1) - 0.5 * np.sin(2 * np.pi * (x1 - x2)) \
        + (1 / 3) * np.cos(2 * np.pi * x1) * np.cos(4 * np.pi * x2)
    return np.stack([beta1, beta2], axis=-1)


def simulate_panel(seed: int) -> dict:
    """Simulate one balanced panel under the PanelGP data-generating process.

    Returns a dict with the panel in long form (one row per asset-month) plus
    the true smooth maps and latent realized loadings/factors needed to
    evaluate recovery.
    """
    rng = np.random.default_rng(seed)
    x = characteristic_grid()            # (N, 2)
    c = context_sequence()                # (T,)
    n_assets, n_months = x.shape[0], c.shape[0]

    beta_x = true_beta(x)                 # (N, H) smooth loadings at training characteristics
    phi_c = true_phi(c)                   # (T, H) smooth factors at training contexts

    beta_innov = rng.normal(0.0, np.sqrt(SIM_SIGMA_BETA2), size=(n_assets, SIM_H))
    phi_innov = rng.normal(0.0, np.sqrt(SIM_SIGMA_PHI2), size=(n_months, SIM_H))
    beta_tilde = beta_x + beta_innov       # realized loadings (Eq. 2-ish of the model)
    phi_tilde = phi_c + phi_innov          # realized factors

    mean = beta_tilde @ phi_tilde.T        # (N, T) conditional mean per asset-month
    noise = rng.normal(0.0, np.sqrt(SIM_SIGMA_R2), size=(n_assets, n_months))
    returns = mean + noise                 # r_{i,t+1} = beta_tilde_i^T phi_tilde_t + sigma_r eps

    asset_idx, month_idx = np.meshgrid(np.arange(n_assets), np.arange(n_months), indexing="ij")
    panel = {
        "asset_id": asset_idx.ravel(),
        "month_id": month_idx.ravel(),
        "x": x[asset_idx.ravel()],
        "c": c[month_idx.ravel()],
        "r": returns.ravel(),
    }
    return {
        "panel": panel,
        "x_grid": x,
        "c_seq": c,
        "beta_x_true": beta_x,
        "phi_c_true": phi_c,
        "beta_tilde_true": beta_tilde,
        "phi_tilde_true": phi_tilde,
        "returns_matrix": returns,
    }
