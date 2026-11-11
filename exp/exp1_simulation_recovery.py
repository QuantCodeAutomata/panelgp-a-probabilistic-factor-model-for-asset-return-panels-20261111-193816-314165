"""Experiment 1: simulation recovery of smooth maps and variance parameters
(paper Section 5.1).

Fits PanelGP on five independently simulated balanced panels (80 assets, 50
months) generated from the model's own data-generating process, and checks
that the variational estimator recovers the true smooth loading and factor
maps, the realized latent factor path, and the three innovation/noise
variances. Run as ``python -m exp.exp1_simulation_recovery``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import (
    simulate_panel, characteristic_grid, context_sequence, true_beta, true_phi,
)
from src.config import (
    SIM_H, SIM_N_INDUCING, SIM_N_EPOCHS_RUN, SIM_N_REPLICATIONS, SIM_GRID_RESOLUTION,
    SIM_SIGMA_BETA2, SIM_SIGMA_PHI2, SIM_SIGMA_R2, MINIBATCH_ROWS, RANDOM_SEED_BASE,
)
from src.metrics import function_r2
from src.panelgp import PanelGPConfig, PanelGPModel
from src.prediction import fit_summary, predict_month, predict_smooth_grid
from src.training import train_panelgp

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _dense_support_grid(resolution: int) -> np.ndarray:
    """Dense grid on [0,1]^2 restricted to the training support (the same
    9x9-with-omitted-corner footprint as Section 5.1, at finer resolution)."""
    g = np.linspace(0.0, 1.0, resolution)
    x1, x2 = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([x1.ravel(), x2.ravel()], axis=-1)
    keep = ~((pts[:, 0] > 7 / 8) & (pts[:, 1] > 7 / 8))
    return pts[keep]


def _align_factors(phi_hat_true_units: np.ndarray, phi_true: np.ndarray) -> np.ndarray:
    """Least-squares Upsilon^T solving phi_hat_true_units @ Upsilon^T ~= phi_true
    (paper Section 5.1, "align estimated factors to true ones by least squares")."""
    upsilon_t, *_ = np.linalg.lstsq(phi_hat_true_units, phi_true, rcond=None)
    return upsilon_t  # this is Upsilon^T, shape (H, H)


def run_replication(seed: int, rep_index: int) -> dict:
    sim = simulate_panel(seed)
    x_grid, c_seq = sim["x_grid"], sim["c_seq"]
    n_assets, n_months = x_grid.shape[0], c_seq.shape[0]
    panel = sim["panel"]

    r = panel["r"]
    s_r = float(np.sqrt(np.mean(r ** 2)))  # no centering, App. C.1
    r_scaled = r / s_r

    cfg = PanelGPConfig(
        n_characteristics=2, n_context=1, n_factors=SIM_H, n_inducing=SIM_N_INDUCING,
        loading_kernel="rbf", factor_kernel="rbf", isotropic_innovations=True,
    )
    model = PanelGPModel(cfg, train_x=panel["x"], train_c=c_seq.reshape(-1, 1), seed=seed)
    history = train_panelgp(
        model, x=panel["x"], t_idx=panel["month_id"], r=r_scaled,
        n_epochs=SIM_N_EPOCHS_RUN, minibatch_rows=MINIBATCH_ROWS, seed=seed,
        log_every=max(1, SIM_N_EPOCHS_RUN // 5),
    )
    fit = fit_summary(model)

    # --- smooth-function recovery on a dense grid -------------------------
    x_dense = _dense_support_grid(SIM_GRID_RESOLUTION)
    c_dense = np.linspace(0.0, 1.0, SIM_GRID_RESOLUTION).reshape(-1, 1)
    beta_true_dense = true_beta(x_dense)          # (Ng, H)
    phi_true_dense = true_phi(c_dense[:, 0])      # (Nc, H)

    beta_hat_dense = np.zeros((x_dense.shape[0], SIM_H))
    phi_hat_dense = np.zeros((c_dense.shape[0], SIM_H))
    for h in range(SIM_H):
        beta_hat_dense[:, h], _ = predict_smooth_grid(fit, "beta", h, x_dense)
        phi_hat_dense[:, h], _ = predict_smooth_grid(fit, "phi", h, c_dense)
    phi_hat_dense_true_units = s_r * phi_hat_dense

    # Factor alignment: regress true realized training factors on s_r-scaled
    # fitted realized factors (paper Section 5.1).
    u_slices, f_slices = model.block_slices()
    phi_hat_train = np.stack([fit.m_v[f_slices[h]] for h in range(SIM_H)], axis=1)  # (T,H)
    phi_hat_train_true_units = s_r * phi_hat_train
    upsilon_t = _align_factors(phi_hat_train_true_units, sim["phi_tilde_true"])
    upsilon = upsilon_t.T
    upsilon_inv = np.linalg.inv(upsilon)

    phi_hat_dense_aligned = phi_hat_dense_true_units @ upsilon_t
    beta_hat_dense_aligned = beta_hat_dense @ upsilon_inv

    factor_r2_h = [function_r2(phi_true_dense[:, h], phi_hat_dense_aligned[:, h]) for h in range(SIM_H)]
    loading_r2_h = [function_r2(beta_true_dense[:, h], beta_hat_dense_aligned[:, h]) for h in range(SIM_H)]
    factor_r2 = float(np.mean(factor_r2_h))
    loading_r2 = float(np.mean(loading_r2_h))

    # Realized (innovation-included) factor recovery for the median-replication plot.
    phi_train_aligned = phi_hat_train_true_units @ upsilon_t
    innovation_r2_h = [function_r2(sim["phi_tilde_true"][:, h], phi_train_aligned[:, h]) for h in range(SIM_H)]

    # --- variance parameters -----------------------------------------------
    D_beta_hat = float(fit.tau_beta[0]) * np.eye(SIM_H)              # isotropic, beta kept in native scale
    D_phi_hat_true_units = (s_r ** 2) * float(fit.rho[0]) * np.eye(SIM_H)
    D_beta_aligned = upsilon_inv.T @ D_beta_hat @ upsilon_inv
    D_phi_aligned = upsilon @ D_phi_hat_true_units @ upsilon.T
    sigma_beta2_hat = float(np.mean(np.diag(D_beta_aligned)))
    sigma_phi2_hat = float(np.mean(np.diag(D_phi_aligned)))
    sigma_r2_hat = float(fit.sigma_r2) * (s_r ** 2)

    # --- predictive mean / variance vs. true conditional mean and oracle variance ---
    beta_true_train = true_beta(x_grid)   # (N,H)
    phi_true_train = true_phi(c_seq)      # (T,H)
    true_cond_mean = beta_true_train @ phi_true_train.T   # (N,T)

    fitted_mean = np.zeros((n_assets, n_months))
    predicted_var = np.zeros((n_assets, n_months))
    for t in range(n_months):
        out = predict_month(fit, x_grid, c_seq[t:t + 1].reshape(1, 1))
        fitted_mean[:, t] = s_r * out["mu"]
        predicted_var[:, t] = (s_r ** 2) * np.diag(out["Sigma"])

    # Oracle variance (Eq. 6): Var(r|beta(x),phi(c)) = phi(c)^T D_beta phi(c)
    #   + beta(x)^T D_phi beta(x) + tr(D_beta D_phi) + sigma_r^2, true D_beta=D_phi=sigma^2 I.
    phi_term = SIM_SIGMA_BETA2 * np.sum(phi_true_train ** 2, axis=1)       # (T,)
    beta_term = SIM_SIGMA_PHI2 * np.sum(beta_true_train ** 2, axis=1)     # (N,)
    oracle_var = beta_term[:, None] + phi_term[None, :] + SIM_H * SIM_SIGMA_BETA2 * SIM_SIGMA_PHI2 + SIM_SIGMA_R2

    mean_r2 = function_r2(true_cond_mean.ravel(), fitted_mean.ravel())
    var_corr = float(np.corrcoef(predicted_var.ravel(), oracle_var.ravel())[0, 1])
    var_ratio = float(predicted_var.mean() / oracle_var.mean())

    return {
        "seed": seed, "rep_index": rep_index, "s_r": s_r,
        "factor_r2": factor_r2, "loading_r2": loading_r2,
        "factor_r2_h": factor_r2_h, "loading_r2_h": loading_r2_h,
        "sigma_beta2_hat": sigma_beta2_hat, "sigma_phi2_hat": sigma_phi2_hat,
        "sigma_r2_hat": sigma_r2_hat,
        "mean_r2": mean_r2, "var_corr": var_corr, "var_ratio": var_ratio,
        "innovation_r2_h": innovation_r2_h,
        "history": history,
        "_plot_data": {
            "c_dense": c_dense[:, 0], "phi_true_dense": phi_true_dense,
            "phi_hat_dense_aligned": phi_hat_dense_aligned,
            "c_seq": c_seq, "phi_tilde_true": sim["phi_tilde_true"],
            "phi_train_aligned": phi_train_aligned,
        },
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    reps = []
    for i in range(SIM_N_REPLICATIONS):
        seed = RANDOM_SEED_BASE + i
        print(f"[exp1] replication {i + 1}/{SIM_N_REPLICATIONS} (seed={seed})")
        reps.append(run_replication(seed, i))

    def agg(key):
        vals = np.array([r[key] for r in reps])
        return float(vals.mean()), float(vals.std(ddof=1))

    summary = {
        "n_replications": SIM_N_REPLICATIONS,
        "n_epochs_run": None,
        "factor_r2_mean": agg("factor_r2")[0], "factor_r2_sd": agg("factor_r2")[1],
        "loading_r2_mean": agg("loading_r2")[0], "loading_r2_sd": agg("loading_r2")[1],
        "sigma_beta2_mean": agg("sigma_beta2_hat")[0], "sigma_beta2_sd": agg("sigma_beta2_hat")[1],
        "sigma_phi2_mean": agg("sigma_phi2_hat")[0], "sigma_phi2_sd": agg("sigma_phi2_hat")[1],
        "sigma_r2_mean": agg("sigma_r2_hat")[0], "sigma_r2_sd": agg("sigma_r2_hat")[1],
        "mean_r2_mean": agg("mean_r2")[0], "mean_r2_sd": agg("mean_r2")[1],
        "var_corr_mean": agg("var_corr")[0], "var_corr_sd": agg("var_corr")[1],
        "var_ratio_mean": agg("var_ratio")[0], "var_ratio_sd": agg("var_ratio")[1],
        "per_replication": [
            {k: v for k, v in r.items() if k not in ("history", "_plot_data")} for r in reps
        ],
    }
    with open(RESULTS_DIR / "exp1_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_replication"}, indent=2))

    # Median replication by average factor-function R^2 (Section 5.1, point 9).
    median_idx = int(np.argsort([r["factor_r2"] for r in reps])[len(reps) // 2])
    med = reps[median_idx]
    pdta = med["_plot_data"]
    fig, axes = plt.subplots(1, SIM_H, figsize=(6 * SIM_H, 4.5))
    for h in range(SIM_H):
        ax = axes[h] if SIM_H > 1 else axes
        ax.plot(pdta["c_dense"], pdta["phi_true_dense"][:, h], label="true $\\phi_h(c)$", lw=2)
        ax.plot(pdta["c_dense"], pdta["phi_hat_dense_aligned"][:, h], "--", label="fitted $\\phi_h(c)$", lw=2)
        ax.scatter(pdta["c_seq"], pdta["phi_tilde_true"][:, h], s=10, alpha=0.4, label="realized $\\tilde\\phi_h$ (true)")
        ax.scatter(pdta["c_seq"], pdta["phi_train_aligned"][:, h], s=10, alpha=0.4, marker="x",
                   label=f"realized $\\tilde\\phi_h$ (fitted), R$^2$={med['innovation_r2_h'][h]:.1%}")
        ax.set_xlabel("context $c$")
        ax.set_title(f"factor {h + 1}")
        ax.legend(fontsize=8)
    fig.suptitle(f"Experiment 1: median-replication factor recovery (seed={med['seed']})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_median_replication_factors.png", dpi=150)
    plt.close(fig)
    print(f"Median replication index: {median_idx} (seed={med['seed']})")


if __name__ == "__main__":
    main()
