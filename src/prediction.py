"""Post-training prediction for PanelGP: predictive moments and the variance
decomposition of Section 4.3 (Equations 19-27).

Custom implementation of the paper's bilinear predictive-distribution
formulas; no GP library exposes this exact combination of exact (smooth)
Gaussian-process conditioning with a bilinear likelihood integrated in closed
form over one factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.panelgp import PanelGPModel


@dataclass
class FittedPanelGP:
    """Detached, numpy-friendly snapshot of a trained PanelGPModel."""

    model: PanelGPModel
    m_v: np.ndarray
    S_v: np.ndarray
    L_uu: list
    L_ff: list
    a: np.ndarray
    b: np.ndarray
    rho: np.ndarray
    tau_beta: np.ndarray
    sigma_r2: float
    Z: np.ndarray
    train_c: np.ndarray

    @property
    def H(self) -> int:
        return self.model.H


def fit_summary(model: PanelGPModel) -> FittedPanelGP:
    """Freeze a trained model's variational and kernel parameters."""
    model.eval()
    with torch.no_grad():
        m_v, S_v, L_uu, L_ff, _, _ = model.variational_moments()
        a, b, rho = model.shares()
        tau_beta = model.tau_beta()
        sigma_r2 = model.sigma_r2().item()
    return FittedPanelGP(
        model=model,
        m_v=m_v.numpy(), S_v=S_v.numpy(),
        L_uu=[L.numpy() for L in L_uu], L_ff=[L.numpy() for L in L_ff],
        a=a.numpy(), b=b.numpy(), rho=rho.numpy(), tau_beta=tau_beta.numpy(),
        sigma_r2=sigma_r2, Z=model.Z.detach().numpy(), train_c=model.train_c.numpy(),
    )


def _solve_lower(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve (L L^T) X = B via forward/back substitution, L lower-triangular."""
    import scipy.linalg as sla
    y = sla.solve_triangular(L, B, lower=True)
    x = sla.solve_triangular(L.T, y, lower=False)
    return x


def _smooth_beta_kernel(fit: FittedPanelGP, h: int, X: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
    model = fit.model
    with torch.no_grad():
        xt = torch.as_tensor(X, dtype=torch.float64)
        xt2 = xt if X2 is None else torch.as_tensor(X2, dtype=torch.float64)
        return model.k_beta(h, xt, xt2).numpy()


def _smooth_beta_diag(fit: FittedPanelGP, h: int, X: np.ndarray) -> np.ndarray:
    model = fit.model
    with torch.no_grad():
        return model.k_beta_diag(h, torch.as_tensor(X, dtype=torch.float64)).numpy()


def _smooth_phi_kernel(fit: FittedPanelGP, h: int, C: np.ndarray, C2: np.ndarray | None = None) -> np.ndarray:
    model = fit.model
    with torch.no_grad():
        ct = torch.as_tensor(C, dtype=torch.float64)
        ct2 = ct if C2 is None else torch.as_tensor(C2, dtype=torch.float64)
        a_h, b_h = float(fit.a[h]), float(fit.b[h])
        return model.k_phi_smooth(h, ct, ct2, a_h, b_h).numpy()


def predict_month(fit: FittedPanelGP, X_month: np.ndarray, c_t: np.ndarray) -> dict:
    """Predictive mean, full covariance, and the Eq. (26)-(27) variance
    decomposition for every stock present in one month.

    X_month: (N, P) characteristics of the stocks active in the month.
    c_t: (1, Q) context vector of the month (already standardized as used
    in training).
    Returns a dict with mu (N,), Sigma (N,N), V0 (N,), V_phi (N,),
    V_beta (scalar), tr_DbDf (scalar), sigma_r2 (scalar).
    """
    H = fit.H
    N = X_month.shape[0]
    M = fit.Z.shape[0]

    m_beta = np.zeros((H, N))
    R_full = np.zeros((H, N, N))
    m_phi = np.zeros(H)
    R_phi = np.zeros(H)
    u_slices = fit.model.block_slices()[0]
    f_slices = fit.model.block_slices()[1]

    Wb = np.zeros((H, N, M))
    Wp = np.zeros((H, fit.train_c.shape[0]))
    for h in range(H):
        Kxz = _smooth_beta_kernel(fit, h, X_month, fit.Z)        # (N,M)
        Wb[h] = _solve_lower(fit.L_uu[h], Kxz.T).T                # (N,M)
        Kxx = _smooth_beta_kernel(fit, h, X_month, X_month)       # (N,N)
        R_full[h] = Kxx - Wb[h] @ Kxz.T
        m_beta[h] = Wb[h] @ fit.m_v[u_slices[h]]

        Kcz = _smooth_phi_kernel(fit, h, c_t, fit.train_c)        # (1,T)
        Wp[h] = _solve_lower(fit.L_ff[h], Kcz.T).ravel()          # (T,)
        diag_phi = fit.a[h] + fit.b[h]
        R_phi[h] = diag_phi - Wp[h] @ Kcz.ravel()
        m_phi[h] = Wp[h] @ fit.m_v[f_slices[h]]

    SBB = np.zeros((H, H, N, N))
    SFF = np.zeros((H, H))
    SBP = np.zeros((H, H, N))
    for h in range(H):
        for hp in range(H):
            Suu = fit.S_v[u_slices[h], :][:, u_slices[hp]]
            SBB[h, hp] = Wb[h] @ Suu @ Wb[hp].T
            if h == hp:
                SBB[h, hp] = SBB[h, hp] + R_full[h]

            Sff = fit.S_v[f_slices[h], :][:, f_slices[hp]]
            SFF[h, hp] = float(Wp[h] @ Sff @ Wp[hp])
            if h == hp:
                SFF[h, hp] = SFF[h, hp] + R_phi[h]

            Suf = fit.S_v[u_slices[h], :][:, f_slices[hp]]
            SBP[h, hp] = Wb[h] @ Suf @ Wp[hp]

    mu = np.einsum("h,hj->j", m_phi, m_beta) + np.einsum("hhj->j", SBP)  # trace over h=h'

    tau = fit.tau_beta
    rho = fit.rho
    SBB_tilde = SBB.copy()
    for h in range(H):
        SBB_tilde[h, h] += tau[h] * np.eye(N)
    SFF_tilde = SFF + np.diag(rho)

    V0 = (np.einsum("h,hgjj,g->j", m_phi, SBB, m_phi)
          + np.einsum("hj,hg,gj->j", m_beta, SFF, m_beta)
          + np.einsum("hgjj,gh->j", SBB, SFF)
          + np.einsum("hgj,hgj->j", SBP, SBP))
    V_phi = np.einsum("hj,h,hj->j", m_beta, rho, m_beta) + np.einsum("hhjj,h->j", SBB, rho)
    V_beta = float(m_phi @ np.diag(tau) @ m_phi + np.trace(np.diag(tau) @ SFF))
    tr_DbDf = float(np.sum(tau * rho))

    diag_jj = np.einsum("hgjj->j", SBB_tilde)  # unused helper, kept for clarity
    Sigma = (np.einsum("h,hgjk,g->jk", m_phi, SBB_tilde, m_phi)
             + np.einsum("hj,hg,gk->jk", m_beta, SFF_tilde, m_beta)
             + np.einsum("hgjk,gh->jk", SBB_tilde, SFF_tilde)
             + np.einsum("hgj,hgk->jk", SBP, SBP))
    Sigma = Sigma + fit.sigma_r2 * np.eye(N)

    return {
        "mu": mu, "Sigma": Sigma, "V0": V0, "V_phi": V_phi,
        "V_beta": V_beta, "tr_DbDf": tr_DbDf, "sigma_r2": fit.sigma_r2,
    }


def predict_smooth_grid(fit: FittedPanelGP, which: str, h: int, new_inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and variance of the smooth map (beta_h or phi_h) at new
    inputs, used for the function-R^2 recovery checks of Section 5.1."""
    u_slices, f_slices = fit.model.block_slices()[0], fit.model.block_slices()[1]
    if which == "beta":
        Ksz = _smooth_beta_kernel(fit, h, new_inputs, fit.Z)
        W = _solve_lower(fit.L_uu[h], Ksz.T).T
        diag = _smooth_beta_diag(fit, h, new_inputs)
        R = diag - (W * Ksz).sum(-1)
        block = u_slices[h]
    else:
        Ksz = _smooth_phi_kernel(fit, h, new_inputs, fit.train_c)
        W = _solve_lower(fit.L_ff[h], Ksz.T).T
        diag = fit.a[h] + fit.b[h]
        R = diag - (W * Ksz).sum(-1)
        block = f_slices[h]
    mean = W @ fit.m_v[block]
    Sblock = fit.S_v[block, :][:, block]
    var = np.einsum("nm,mk,nk->n", W, Sblock, W) + R
    return mean, var
