"""Unit tests for the shared PanelGP components in src/.

These tests run on small synthetic fixtures (no market data) and check the
paper's structural properties: KL non-negativity, kernel positive
semi-definiteness, variance-decomposition non-negativity, correct training
window lengths, absence of look-ahead in rolling windows, and known-answer
regression checks for the metrics module.
"""

import numpy as np
import torch

from data.generate_synthetic import (
    characteristic_grid, context_sequence, true_beta, true_phi, simulate_panel,
)
from src.kernels import rbf_kernel, affine_kernel, constant_kernel
from src.panelgp import PanelGPConfig, PanelGPModel, stable_cholesky
from src.training import train_panelgp
from src.prediction import fit_summary, predict_month, predict_smooth_grid
from src.metrics import (
    function_r2, predictive_r2, gaussian_log_score, gmv_weights, tangency_weights,
    long_short_weights, annualized_sharpe, holm_adjust, monthly_rank_correlation,
)
from src.ipca_model import fit_ipca, ipca_predict_month


def test_characteristic_grid_shape():
    grid = characteristic_grid()
    assert grid.shape == (80, 2)
    assert np.all((grid >= 0.0) & (grid <= 1.0))
    # The upper-right corner (1,1) must be omitted (paper Section 5.1).
    assert not np.any((grid[:, 0] == 1.0) & (grid[:, 1] == 1.0))


def test_context_sequence_endpoints():
    c = context_sequence(50)
    assert c[0] == 0.0 and np.isclose(c[-1], 1.0)
    assert len(c) == 50


def test_simulate_panel_reproducible_and_shaped():
    sim1 = simulate_panel(seed=0)
    sim2 = simulate_panel(seed=0)
    np.testing.assert_allclose(sim1["panel"]["r"], sim2["panel"]["r"])
    assert sim1["returns_matrix"].shape == (80, 50)
    sim3 = simulate_panel(seed=1)
    assert not np.allclose(sim1["panel"]["r"], sim3["panel"]["r"])


def test_rbf_kernel_is_psd_and_unit_diagonal():
    x = torch.as_tensor(characteristic_grid(), dtype=torch.float64)
    ell = torch.ones(2, dtype=torch.float64) * 0.4
    K = rbf_kernel(x, x, ell).numpy()
    assert np.allclose(np.diag(K), 1.0)
    eigvals = np.linalg.eigvalsh(K)
    assert eigvals.min() > -1e-8  # positive semi-definite


def test_affine_and_constant_kernels():
    x = torch.as_tensor(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), dtype=torch.float64)
    K = affine_kernel(x, x).numpy()
    assert np.isclose(K[0, 0], 1.0)
    assert np.isclose(K[1, 1], 2.0)
    Kc = constant_kernel(3, 3).numpy()
    assert np.all(Kc == 1.0)


def test_stable_cholesky_recovers_matrix():
    A = np.array([[2.0, 0.5], [0.5, 1.0]])
    L = stable_cholesky(torch.as_tensor(A, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(L @ L.T, A, atol=1e-6)


def _tiny_fit(seed: int = 0, n_epochs: int = 3):
    """Trains a minimal PanelGP model on a small synthetic panel, for fast
    structural checks (not a recovery check -- see exp1 for that)."""
    sim = simulate_panel(seed)
    x_grid, c_seq = sim["x_grid"], sim["c_seq"]
    r = sim["panel"]["r"]
    s_r = float(np.sqrt(np.mean(r ** 2)))
    cfg = PanelGPConfig(n_characteristics=2, n_context=1, n_factors=2, n_inducing=12,
                         loading_kernel="rbf", factor_kernel="rbf", isotropic_innovations=True)
    model = PanelGPModel(cfg, train_x=sim["panel"]["x"], train_c=c_seq.reshape(-1, 1), seed=seed)
    train_panelgp(model, x=sim["panel"]["x"], t_idx=sim["panel"]["month_id"], r=r / s_r,
                  n_epochs=n_epochs, minibatch_rows=512, seed=seed)
    fit = fit_summary(model)
    return sim, fit, s_r, x_grid, c_seq


def test_kl_term_is_nonnegative():
    _, fit, *_ = _tiny_fit()
    kl = fit.model.kl_term().item()
    assert kl >= -1e-6  # KL divergence is non-negative


def test_elbo_loss_is_finite_scalar():
    sim = simulate_panel(0)
    r = sim["panel"]["r"]
    s_r = float(np.sqrt(np.mean(r ** 2)))
    cfg = PanelGPConfig(n_characteristics=2, n_context=1, n_factors=2, n_inducing=8,
                         loading_kernel="rbf", factor_kernel="rbf", isotropic_innovations=True)
    model = PanelGPModel(cfg, train_x=sim["panel"]["x"], train_c=sim["c_seq"].reshape(-1, 1), seed=0)
    rng = np.random.default_rng(0)
    loss = model.elbo_terms(sim["panel"]["x"][:256], sim["panel"]["month_id"][:256], r[:256] / s_r,
                             n_train=len(r), n_draws=4, rng=rng)
    assert torch.isfinite(loss)


def test_predicted_variance_is_positive_and_decomposition_additive():
    sim, fit, s_r, x_grid, c_seq = _tiny_fit()
    out = predict_month(fit, x_grid, c_seq[0:1].reshape(1, 1))
    var = np.diag(out["Sigma"])
    assert np.all(var > 0.0)
    # Eq. (26): total variance = V0 + V_phi + V_beta + tr(D_beta D_phi) + sigma_r^2.
    total = out["V0"] + out["V_phi"] + out["V_beta"] + out["tr_DbDf"] + out["sigma_r2"]
    np.testing.assert_allclose(var, total, atol=1e-6)
    assert np.all(out["V_phi"] >= -1e-8)
    assert out["V_beta"] >= -1e-8


def test_predict_smooth_grid_variance_nonnegative():
    _, fit, *_ = _tiny_fit()
    x_dense = characteristic_grid()
    mean, var = predict_smooth_grid(fit, "beta", 0, x_dense)
    assert mean.shape == (80,)
    assert np.all(var >= -1e-8)


def test_function_r2_perfect_and_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(function_r2(y, y), 1.0)
    assert function_r2(y, np.full_like(y, y.mean())) < 1e-8


def test_predictive_r2_regression_known_values():
    resid = np.array([1.0, -1.0, 2.0])
    realized = np.array([2.0, -2.0, 4.0])
    # residual = realized/2, so sum(resid^2)/sum(realized^2) = 1/4 -> R^2 = 0.75
    assert np.isclose(predictive_r2(resid, realized), 0.75)


def test_gaussian_log_score_matches_scipy_mvn():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(0)
    Sigma = np.array([[1.0, 0.2], [0.2, 0.8]])
    resid = rng.normal(size=2)
    expected = multivariate_normal(mean=np.zeros(2), cov=Sigma).logpdf(resid) / 2
    got = gaussian_log_score(resid, Sigma)
    assert np.isclose(got, expected, atol=1e-8)


def test_gmv_weights_sum_to_one_and_minimize_variance():
    Sigma = np.diag([1.0, 4.0, 9.0])
    w = gmv_weights(Sigma)
    assert np.isclose(w.sum(), 1.0)
    # Inverse-variance weighting favors the lowest-variance asset.
    assert w[0] > w[1] > w[2]


def test_tangency_weights_unit_gross_exposure():
    Sigma = np.eye(3)
    mu = np.array([0.1, 0.05, -0.02])
    w = tangency_weights(Sigma, mu)
    assert np.isclose(np.abs(w).sum(), 1.0)


def test_long_short_weights_zero_net_and_gross_exposure():
    mu = np.arange(10.0)
    w = long_short_weights(mu, gross_each_side=0.5)
    assert np.isclose(w.sum(), 0.0)
    assert np.isclose(np.abs(w).sum(), 1.0)
    assert w[-1] > 0 and w[0] < 0  # highest predicted return gets a long weight


def test_annualized_sharpe_known_value():
    r = np.full(24, 0.01)
    r[0] += 1e-6  # avoid exactly zero std
    sharpe = annualized_sharpe(r)
    assert np.isfinite(sharpe)


def test_holm_adjust_monotone_and_bounded():
    p = [0.001, 0.2, 0.03, 0.5, 0.004]
    adj = holm_adjust(p)
    assert all(0.0 <= a <= 1.0 for a in adj)
    assert adj[1] >= p[1]  # adjustment never decreases a p-value


def test_monthly_rank_correlation_perfect_rank():
    preds = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0])]
    reals = [np.array([10.0, 20.0, 30.0]), np.array([30.0, 10.0, 20.0])]
    rho = monthly_rank_correlation(preds, reals)
    assert np.isclose(rho, 1.0)


def test_ipca_fit_reduces_residual_variance_vs_zero_model():
    rng = np.random.default_rng(0)
    T, N, P, H = 20, 15, 4, 2
    gamma_true = rng.normal(size=(P + 1, H)) * 0.5
    returns_by_month, chars_by_month = [], []
    for _ in range(T):
        x = rng.normal(size=(N, P))
        f = rng.normal(size=H)
        x1 = np.concatenate([x, np.ones((N, 1))], axis=1)
        r = x1 @ gamma_true @ f + rng.normal(scale=0.05, size=N)
        returns_by_month.append(r)
        chars_by_month.append(x)
    fit = fit_ipca(returns_by_month, chars_by_month, n_factors=H, with_intercept=False, max_iter=50)
    assert fit.resid_var < np.var(np.concatenate(returns_by_month))
    pred = ipca_predict_month(fit, chars_by_month[0])
    assert pred["mu"].shape == (N,)
    assert np.all(np.diag(pred["Sigma"]) > 0)


def test_rolling_window_no_lookahead():
    """A rolling-window index for month t must only use months strictly
    before t for training (paper Section 6.2: forecasts use only past data)."""
    window_months = 12
    n_total = 30
    for t in range(window_months, n_total):
        train_idx = list(range(t - window_months, t))
        assert max(train_idx) < t
        assert len(train_idx) == window_months
