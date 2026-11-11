"""Bilinear predictive density via Fourier inversion of the scalar CGF
(paper Appendix B.3, Proposition 1, used in Section 6.6 / Experiment 5).

For one stock j at date t, the predictive state z = [beta_j, phi_t] is a
2H-dimensional Gaussian N(m_*, S_tilde_*) (the blocks SBB_tilde, SFF_tilde,
SBP returned by ``predict_month``). The return r_{j,t+1} = beta_j^T phi_t +
sigma_r eps is a Gaussian quadratic form plus independent noise. Writing
A = [[0, I/2], [I/2, 0]] so that z^T A z = beta_j^T phi_t, the scalar
cumulant generating function is (Eq. 22 of Proposition 1):

    K(s) = -1/2 log det(I - 2 s S_tilde_* A)
           + s * m_*^T A (I - 2 s S_tilde_* A)^{-1} m_*
           + 1/2 sigma_r^2 s^2

Custom implementation: no library provides this bilinear-Gaussian CGF or its
Fourier inversion; everything below follows Proposition 1 directly.
"""

from __future__ import annotations

import numpy as np
import torch

N_BISECTION_ITERS = 60
DOMAIN_GROWTH_CAP = 1000.0


def pairing_matrix(H: int) -> np.ndarray:
    """A such that z^T A z = beta^T phi for z = [beta; phi] in R^{2H}."""
    A = np.zeros((2 * H, 2 * H))
    A[:H, H:] = 0.5 * np.eye(H)
    A[H:, :H] = 0.5 * np.eye(H)
    return A


def build_states(m_beta_j: np.ndarray, m_phi: np.ndarray, SBB_tilde_jj: np.ndarray,
                  SFF_tilde: np.ndarray, SBP_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stack the per-stock blocks of ``predict_month`` into the state mean
    m_* (N,2H) and covariance S_tilde_* (N,2H,2H)."""
    N, H = m_beta_j.shape
    m_star = np.concatenate([m_beta_j, np.broadcast_to(m_phi, (N, H))], axis=1)
    S_star = np.zeros((N, 2 * H, 2 * H))
    S_star[:, :H, :H] = np.moveaxis(SBB_tilde_jj, -1, 0)      # (N,H,H)
    S_star[:, H:, H:] = SFF_tilde[None, :, :]
    S_star[:, :H, H:] = np.moveaxis(SBP_j, -1, 0)
    S_star[:, H:, :H] = np.moveaxis(SBP_j, -1, 0).transpose(0, 2, 1)
    return m_star, S_star


def _cgf_torch(s: torch.Tensor, m: torch.Tensor, S: torch.Tensor, A: torch.Tensor,
               sigma_r2: float) -> torch.Tensor:
    """Batched real-valued CGF K(s) (one scalar s per batch row)."""
    D = m.shape[-1]
    I = torch.eye(D, dtype=torch.float64)
    sA = s.view(-1, 1, 1) * A
    M = I.unsqueeze(0) - 2.0 * torch.matmul(S, sA)
    sign, logdet = torch.linalg.slogdet(M)
    Minv_m = torch.linalg.solve(M, m.unsqueeze(-1)).squeeze(-1)
    Am = torch.einsum("ij,nj->ni", A, m)
    quad = s * torch.sum(Am * Minv_m, dim=-1)
    return -0.5 * logdet + quad + 0.5 * sigma_r2 * s ** 2


def _domain_ok(s: np.ndarray, S: np.ndarray, A: np.ndarray) -> np.ndarray:
    D = S.shape[-1]
    M = np.eye(D)[None] - 2.0 * s[:, None, None] * (S @ A[None])
    sign, logdet = np.linalg.slogdet(M)
    return (sign > 0) & np.isfinite(logdet)


def _domain_bounds(S: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-observation safe (s_min, s_max) interior to the CGF domain
    {s : S_tilde_*^{-1} - 2 s A is positive definite}."""
    N = S.shape[0]
    hi = np.full(N, 0.05)
    for _ in range(30):
        ok = _domain_ok(hi, S, A)
        hi = np.where(ok & (hi < DOMAIN_GROWTH_CAP), hi * 1.5, hi)
        if np.all(~ok | (hi >= DOMAIN_GROWTH_CAP)):
            break
    lo_hi = np.zeros(N)
    for _ in range(N_BISECTION_ITERS):
        mid = 0.5 * (lo_hi + hi)
        ok = _domain_ok(mid, S, A)
        lo_hi = np.where(ok, mid, lo_hi)
        hi = np.where(ok, hi, mid)
    s_max = lo_hi

    lo = np.full(N, -0.05)
    for _ in range(30):
        ok = _domain_ok(lo, S, A)
        lo = np.where(ok & (-lo < DOMAIN_GROWTH_CAP), lo * 1.5, lo)
        if np.all(~ok | (-lo >= DOMAIN_GROWTH_CAP)):
            break
    hi_lo = np.zeros(N)
    for _ in range(N_BISECTION_ITERS):
        mid = 0.5 * (lo + hi_lo)
        ok = _domain_ok(mid, S, A)
        hi_lo = np.where(ok, mid, hi_lo)
        lo = np.where(ok, lo, mid)
    s_min = hi_lo
    return 0.999 * s_min, 0.999 * s_max


def solve_tilt(r: np.ndarray, m_star: np.ndarray, S_star: np.ndarray, A: np.ndarray,
               sigma_r2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve K'(s) = r for every observation (vectorized bisection, K' strictly
    increasing on the CGF domain), and return (s, K(s), d = sqrt(K''(s)))."""
    s_min, s_max = _domain_bounds(S_star, A)
    m_t = torch.as_tensor(m_star, dtype=torch.float64)
    S_t = torch.as_tensor(S_star, dtype=torch.float64)
    A_t = torch.as_tensor(A, dtype=torch.float64)

    def k_prime(s_np: np.ndarray) -> np.ndarray:
        s_t = torch.tensor(s_np, dtype=torch.float64, requires_grad=True)
        K = _cgf_torch(s_t, m_t, S_t, A_t, sigma_r2).sum()
        g, = torch.autograd.grad(K, s_t)
        return g.numpy()

    lo, hi = s_min.copy(), s_max.copy()
    for _ in range(N_BISECTION_ITERS):
        mid = 0.5 * (lo + hi)
        g = k_prime(mid) - r
        lo = np.where(g < 0, mid, lo)
        hi = np.where(g < 0, hi, mid)
    s = 0.5 * (lo + hi)

    s_t = torch.tensor(s, dtype=torch.float64, requires_grad=True)
    K_val = _cgf_torch(s_t, m_t, S_t, A_t, sigma_r2)
    g, = torch.autograd.grad(K_val.sum(), s_t, create_graph=True)
    kpp, = torch.autograd.grad(g.sum(), s_t)
    d = torch.sqrt(torch.clamp(kpp, min=1e-10)).detach().numpy()
    return s, K_val.detach().numpy(), d


def _cgf_complex(s: np.ndarray, m: np.ndarray, S: np.ndarray, A: np.ndarray,
                  sigma_r2: float) -> np.ndarray:
    """K(s) for complex ``s``, one value per observation, using a phase
    continuous along the integration path (see ``log_density`` below)."""
    D = S.shape[-1]
    I = np.eye(D, dtype=complex)
    sA = s[:, None, None] * A[None].astype(complex)
    M = I[None] - 2.0 * np.matmul(S.astype(complex), sA)
    det = np.linalg.det(M)
    Minv_m = np.linalg.solve(M, m.astype(complex)[:, :, None])[:, :, 0]
    Am = np.einsum("ij,nj->ni", A, m).astype(complex)
    quad = s * np.sum(Am * Minv_m, axis=-1)
    return det, quad + 0.5 * sigma_r2 * s ** 2


def log_density(r: np.ndarray, m_star: np.ndarray, S_star: np.ndarray, A: np.ndarray,
                 sigma_r2: float, w_max: float = 40.0, n_grid: int = 2001) -> np.ndarray:
    """log p_bil(r) for every observation via the Fourier-inversion formula of
    Appendix B.3. The complex log-determinant is phase-unwrapped along a fixed
    grid in w (the branch of the matrix square root must vary continuously
    along the path, which an adaptive quadrature cannot guarantee pointwise),
    then integrated with Simpson's rule in place of ``scipy.integrate.quad_vec``.
    """
    s, K_s, d = solve_tilt(r, m_star, S_star, A, sigma_r2)
    N = r.shape[0]
    w_grid = np.linspace(0.0, w_max, n_grid)
    log_dets = np.zeros((n_grid, N), dtype=complex)
    quads = np.zeros((n_grid, N), dtype=complex)
    for k, w in enumerate(w_grid):
        s_complex = s + 1j * w / d
        det, quad = _cgf_complex(s_complex, m_star, S_star, A, sigma_r2)
        log_dets[k] = np.log(np.abs(det)) + 1j * np.angle(det)
        quads[k] = quad
    angles = np.unwrap(log_dets.imag, axis=0)
    log_det_cont = log_dets.real + 1j * angles
    K_complex = -0.5 * log_det_cont + quads                     # (n_grid, N)
    integrand = np.real(np.exp(K_complex - K_s[None, :] - 1j * w_grid[:, None] * r[None, :] / d[None, :]))
    integral = np.trapz(integrand, w_grid, axis=0)
    integral = np.clip(integral, 1e-300, None)
    return K_s - s * r - np.log(np.pi * d) + np.log(integral)


def gaussian_log_density(r: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * (r - mu) ** 2 / var
