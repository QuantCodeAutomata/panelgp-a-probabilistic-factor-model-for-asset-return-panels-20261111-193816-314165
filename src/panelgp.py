"""PanelGP: sparse variational Gaussian-process factor model (Sections 3-4).

Custom implementation of the model and inference scheme of Ruglioni and
Filipovic (2026); no published library implements this bilinear
loading/factor variational GP with a dense whitened posterior over inducing
loadings and realized factors, so it is coded from the paper's equations
(Section 4.1, Appendix A-C) with PyTorch autodiff for optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.config import (
    JITTER_INITIAL, JITTER_GROWTH, WHITENED_MEAN_INIT_STD, INIT_SIGMA_R2,
    INIT_RHO_H, INIT_B_H, INIT_A_H, UNIFORM_VARIANCE,
)
from src.kernels import rbf_kernel, affine_kernel, affine_diag


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


def inv_softplus(y: float) -> float:
    """Inverse of softplus, for initializing raw parameters at a target value."""
    y = max(y, 1e-6)
    return float(np.log(np.expm1(y)))


def stable_cholesky(mat: torch.Tensor) -> torch.Tensor:
    """Cholesky factorization with a diagonal jitter that grows tenfold until
    it succeeds, as in Appendix C.3 ("Numerical stabilization")."""
    n = mat.shape[-1]
    jitter = JITTER_INITIAL
    eye = torch.eye(n, device=mat.device, dtype=mat.dtype)
    base = mat
    for _ in range(10):
        try:
            return torch.linalg.cholesky(base + jitter * eye)
        except RuntimeError:
            jitter *= JITTER_GROWTH
    raise RuntimeError("Cholesky factorization failed even with large jitter")


@dataclass
class PanelGPConfig:
    n_characteristics: int
    n_context: int
    n_factors: int
    n_inducing: int
    loading_kernel: str = "rbf"   # 'rbf' or 'affine'
    factor_kernel: str = "rbf"    # 'rbf' or 'constant'
    isotropic_innovations: bool = False  # shared D_beta, D_phi across h (simulation only)


class PanelGPModel(nn.Module):
    """Variational PanelGP model for a fixed training panel.

    All quantities use scaled returns r_n^circ = r_n / s_r (paper App. C.1);
    the scaling factor s_r is supplied externally and only used to rescale
    predictive moments back to return units.
    """

    def __init__(self, cfg: PanelGPConfig, train_x: np.ndarray, train_c: np.ndarray, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(seed)
        H, P, Q, M = cfg.n_factors, cfg.n_characteristics, cfg.n_context, cfg.n_inducing
        T = train_c.shape[0]
        self.H, self.P, self.Q, self.M, self.T = H, P, Q, M, T

        self.register_buffer("train_c", torch.as_tensor(train_c, dtype=torch.float64))

        # Inducing inputs: initialized at distinct training rows (App. C.3), optimized jointly.
        rng = np.random.default_rng(seed)
        idx = rng.choice(train_x.shape[0], size=M, replace=train_x.shape[0] < M)
        self.Z = nn.Parameter(torch.as_tensor(train_x[idx], dtype=torch.float64).clone())

        # Lengthscale initialization, Appendix C.3, Eq. (C.7): l0 = sqrt(sum_p var_p).
        ell0_beta = float(np.sqrt(P * UNIFORM_VARIANCE))
        ell0_phi = float(np.sqrt(Q * UNIFORM_VARIANCE))
        if cfg.loading_kernel == "rbf":
            self.raw_ell_beta = nn.Parameter(torch.full((H, P), np.log(np.expm1(ell0_beta)), dtype=torch.float64))
        if cfg.factor_kernel == "rbf":
            self.raw_ell_phi = nn.Parameter(torch.full((H, Q), np.log(np.expm1(ell0_phi)), dtype=torch.float64))

        # Output scale eta_h = 0.9 / (H*(d+0.1)), d=1 for RBF, d=1+mean||x||^2 for affine.
        d = 1.0 if cfg.loading_kernel == "rbf" else 1.0 + float((train_x ** 2).sum(-1).mean())
        eta0 = 0.9 / (H * (d + 0.1))
        self.raw_eta = nn.Parameter(torch.full((H,), inv_softplus(eta0), dtype=torch.float64))

        # Factor kernel shares a_h, b_h, rho_h with a_h+b_h+rho_h=1 (Eq. C.4).
        if cfg.factor_kernel == "rbf":
            logits0 = torch.log(torch.tensor([INIT_A_H, INIT_B_H, INIT_RHO_H], dtype=torch.float64))
            if cfg.isotropic_innovations:
                self.raw_shares = nn.Parameter(logits0.clone())  # (3,) shared across h
            else:
                self.raw_shares = nn.Parameter(logits0.unsqueeze(0).repeat(H, 1).clone())  # (H,3)
        else:
            rho0 = INIT_RHO_H
            if cfg.isotropic_innovations:
                self.raw_rho = nn.Parameter(torch.tensor(float(inv_softplus_logit(rho0)), dtype=torch.float64))
            else:
                self.raw_rho = nn.Parameter(torch.full((H,), inv_softplus_logit(rho0), dtype=torch.float64))

        # Loading-innovation variance tau_beta,h = [D_beta]_hh (softplus transform).
        tau0 = inv_softplus(INIT_SIGMA_R2 / 2.0) if False else inv_softplus(0.05)
        if cfg.isotropic_innovations:
            self.raw_tau_beta = nn.Parameter(torch.tensor(inv_softplus(0.05), dtype=torch.float64))
        else:
            self.raw_tau_beta = nn.Parameter(torch.full((H,), inv_softplus(0.05), dtype=torch.float64))

        self.raw_sigma_r2 = nn.Parameter(torch.tensor(inv_softplus(INIT_SIGMA_R2), dtype=torch.float64))

        # Whitened variational state omega ~ N(0, I) under the prior; d_v = (M+T)*H.
        d_v = (M + T) * H
        self.d_v = d_v
        self.m_omega = nn.Parameter(WHITENED_MEAN_INIT_STD * torch.randn(d_v, dtype=torch.float64))
        self.L_omega_raw = nn.Parameter(torch.zeros(d_v, d_v, dtype=torch.float64))  # strictly lower part
        self.L_omega_diag_raw = nn.Parameter(torch.zeros(d_v, dtype=torch.float64))  # softplus -> positive diag

    # ---- helper accessors -------------------------------------------------
    def eta(self) -> torch.Tensor:
        return softplus(self.raw_eta)

    def tau_beta(self) -> torch.Tensor:
        val = softplus(self.raw_tau_beta)
        return val.expand(self.H) if self.cfg.isotropic_innovations else val

    def shares(self):
        """Returns (a, b, rho), each a length-H tensor."""
        H = self.H
        if self.cfg.factor_kernel == "rbf":
            logits = self.raw_shares
            if self.cfg.isotropic_innovations:
                s = torch.softmax(logits, dim=-1)
                a, b, rho = s[0].expand(H), s[1].expand(H), s[2].expand(H)
            else:
                s = torch.softmax(logits, dim=-1)  # (H,3)
                a, b, rho = s[:, 0], s[:, 1], s[:, 2]
        else:
            if self.cfg.isotropic_innovations:
                rho = torch.sigmoid(self.raw_rho).expand(H)
            else:
                rho = torch.sigmoid(self.raw_rho)
            a = torch.zeros(H, dtype=torch.float64)
            b = 1.0 - rho
        return a, b, rho

    def sigma_r2(self) -> torch.Tensor:
        return softplus(self.raw_sigma_r2)

    def L_omega(self) -> torch.Tensor:
        diag = softplus(self.L_omega_diag_raw)
        tril = torch.tril(self.L_omega_raw, diagonal=-1)
        return tril + torch.diag(diag)

    # ---- kernels ------------------------------------------------------
    def k_beta(self, h: int, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.cfg.loading_kernel == "rbf":
            ell = softplus(self.raw_ell_beta[h])
            base = rbf_kernel(x1, x2, ell)
        else:
            base = affine_kernel(x1, x2)
        return self.eta()[h] * base

    def k_beta_diag(self, h: int, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.loading_kernel == "rbf":
            base = torch.ones(x.shape[0], dtype=torch.float64)
        else:
            base = affine_diag(x)
        return self.eta()[h] * base

    def k_phi_smooth(self, h: int, c1: torch.Tensor, c2: torch.Tensor, a_h, b_h) -> torch.Tensor:
        if self.cfg.factor_kernel == "rbf":
            ell = softplus(self.raw_ell_phi[h])
            rbf = rbf_kernel(c1, c2, ell)
            return a_h * rbf + b_h
        else:
            n1, n2 = c1.shape[0], c2.shape[0]
            return b_h * torch.ones(n1, n2, dtype=torch.float64)

    # ---- prior covariance blocks ---------------------------------------
    def prior_blocks(self):
        """Returns per-component Cholesky factors L_uu[h] (MxM) and L_ff[h] (TxT)."""
        a, b, rho = self.shares()
        L_uu, L_ff, K_uu = [], [], []
        for h in range(self.H):
            Kuu = self.k_beta(h, self.Z, self.Z)
            K_uu.append(Kuu)
            L_uu.append(stable_cholesky(Kuu))
            Kff = self.k_phi_smooth(h, self.train_c, self.train_c, a[h], b[h])
            Kff = Kff + rho[h] * torch.eye(self.T, dtype=torch.float64)
            L_ff.append(stable_cholesky(Kff))
        return L_uu, L_ff, K_uu

    def full_L_v(self, L_uu, L_ff) -> torch.Tensor:
        blocks = list(L_uu) + list(L_ff)
        return torch.block_diag(*blocks)

    def variational_moments(self):
        """Returns m_v (d_v,), S_v (d_v,d_v), and the block index ranges."""
        L_uu, L_ff, K_uu = self.prior_blocks()
        L_v = self.full_L_v(L_uu, L_ff)
        m_v = L_v @ self.m_omega
        L_omega = self.L_omega()
        S_omega = L_omega @ L_omega.T
        S_v = L_v @ S_omega @ L_v.T
        return m_v, S_v, L_uu, L_ff, K_uu, L_omega

    def block_slices(self):
        M, T, H = self.M, self.T, self.H
        u_slices = [slice(h * M, (h + 1) * M) for h in range(H)]
        f_slices = [slice(H * M + h * T, H * M + (h + 1) * T) for h in range(H)]
        return u_slices, f_slices

    def kl_term(self) -> torch.Tensor:
        L_omega = self.L_omega()
        S_omega_diag_sum = (L_omega ** 2).sum()  # tr(S_omega) = sum L_ij^2
        log_det = 2.0 * torch.log(torch.diagonal(L_omega)).sum()
        quad = (self.m_omega ** 2).sum()
        return 0.5 * (S_omega_diag_sum + quad - self.d_v + (-log_det))

    def elbo_terms(self, x_batch: np.ndarray, t_batch: np.ndarray, r_batch: np.ndarray,
                   n_train: int, n_draws: int, rng: np.random.Generator):
        """Negative-ELBO loss for one minibatch (paper Eq. 17, App. A.2, C.4)."""
        H, M, T = self.H, self.M, self.T
        x_t = torch.as_tensor(x_batch, dtype=torch.float64)
        t_t = torch.as_tensor(t_batch, dtype=torch.long)
        r_t = torch.as_tensor(r_batch, dtype=torch.float64)
        B = x_t.shape[0]

        m_v, S_v, L_uu, L_ff, K_uu, _ = self.variational_moments()
        u_slices, f_slices = self.block_slices()

        # W_h = K_beta,h(x,Z) K_uu,h^{-1}  (B,M); R_h(x) = k(x,x) - sum(W_h * K_xz, axis=-1)
        W = torch.zeros(H, B, M, dtype=torch.float64)
        R = torch.zeros(H, B, dtype=torch.float64)
        m_beta = torch.zeros(B, H, dtype=torch.float64)
        for h in range(H):
            Kxz = self.k_beta(h, x_t, self.Z)                       # (B,M)
            Wh_T = torch.cholesky_solve(Kxz.T, L_uu[h])              # (M,B) = Kuu^{-1} Kzx
            Wh = Wh_T.T                                              # (B,M)
            W[h] = Wh
            diagxx = self.k_beta_diag(h, x_t)
            R[h] = diagxx - (Wh * Kxz).sum(-1)
            m_beta[:, h] = Wh @ m_v[u_slices[h]]

        m_phi = torch.zeros(B, H, dtype=torch.float64)
        for h in range(H):
            m_phi[:, h] = m_v[f_slices[h]][t_t]

        # Cross/auto covariance blocks, gathered per row.
        S_betabeta = torch.zeros(B, H, H, dtype=torch.float64)
        S_betaphi = torch.zeros(B, H, H, dtype=torch.float64)
        S_phiphi = torch.zeros(B, H, H, dtype=torch.float64)
        for h in range(H):
            for hp in range(H):
                Suu_hhp = S_v[u_slices[h], u_slices[hp]]           # (M,M)
                tmp = W[h] @ Suu_hhp                                 # (B,M)
                val = (tmp * W[hp]).sum(-1)
                if h == hp:
                    val = val + R[h]
                S_betabeta[:, h, hp] = val

                Suf_hhp = S_v[u_slices[h], f_slices[hp]]             # (M,T)
                col = Suf_hhp[:, t_t]                                # (M,B)
                S_betaphi[:, h, hp] = (W[h] * col.T).sum(-1)

                Sff_hhp = S_v[f_slices[h], f_slices[hp]]             # (T,T)
                diag_hhp = torch.diagonal(Sff_hhp)                  # (T,)
                S_phiphi[:, h, hp] = diag_hhp[t_t]

        jitter_h = 1e-6 * torch.eye(H, dtype=torch.float64)
        Sphiphi_j = S_phiphi + jitter_h
        L_phiphi = torch.linalg.cholesky(Sphiphi_j)

        # 32 independent draws + their negatives -> n_draws*2 factor samples, shared by date.
        n_dates = int(t_t.max().item()) + 1
        e = torch.as_tensor(rng.standard_normal((n_dates, n_draws, H)), dtype=torch.float64)
        e = torch.cat([e, -e], dim=1)                                # (n_dates, 2*n_draws, H)
        e_rows = e[t_t]                                              # (B, 2*n_draws, H)
        varphi = m_phi.unsqueeze(1) + (L_phiphi.unsqueeze(1) @ e_rows.unsqueeze(-1)).squeeze(-1)
        # varphi: (B, K, H)
        K = varphi.shape[1]

        diff = varphi - m_phi.unsqueeze(1)                           # (B,K,H)
        Sinv_diff = torch.linalg.solve(Sphiphi_j.unsqueeze(1).expand(-1, K, -1, -1),
                                        diff.unsqueeze(-1)).squeeze(-1)  # (B,K,H)
        beta_bar = m_beta.unsqueeze(1) + torch.einsum("bhg,bkg->bkh", S_betaphi, Sinv_diff)  # (B,K,H)

        Sphibeta = S_betaphi.transpose(-1, -2)
        Sinv_Sphibeta = torch.linalg.solve(Sphiphi_j, Sphibeta)       # (B,H,H)
        V = S_betabeta - S_betaphi @ Sinv_Sphibeta                    # (B,H,H)

        tau_beta = self.tau_beta()
        nu = self.sigma_r2() + (varphi ** 2 * tau_beta).sum(-1)       # (B,K)
        mean_pred = (beta_bar * varphi).sum(-1)                       # (B,K)
        quad_V = torch.einsum("bkh,bhg,bkg->bk", varphi, V, varphi)   # (B,K)

        resid2 = (r_t.unsqueeze(1) - mean_pred) ** 2
        loglik_sample = -0.5 * (torch.log(2 * np.pi * nu) + (resid2 + quad_V) / nu)
        loglik = loglik_sample.mean(-1)                               # Monte Carlo average over K samples

        scale = n_train / B
        neg_elbo = -(scale * loglik.sum() - self.kl_term())
        return neg_elbo


def inv_softplus_logit(p: float) -> float:
    """logit(p), for parameters passed through a sigmoid."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))
