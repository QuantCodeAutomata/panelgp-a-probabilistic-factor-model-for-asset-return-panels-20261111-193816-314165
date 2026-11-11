"""Elementary kernels used by PanelGP loading and factor maps.

RBF kernel k_RBF(x,x';l) = exp[-1/2 sum_p (x_p-x'_p)^2/l_p^2] (paper Eq. C.2),
the affine kernel 1 + x^T x', and the constant kernel 1 (paper Section 3.3 /
Appendix C.2). These are implemented from scratch following the paper's
closed-form definitions (no GP library kernel matches this exact ARD
parameterization combined with the paper's output-scale and share
parameterizations), as required by Appendix C.2.
"""

import torch


def rbf_kernel(x1: torch.Tensor, x2: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    """ARD RBF kernel matrix, paper Equation (C.2).

    x1: (n1, P), x2: (n2, P), lengthscale: (P,) positive lengthscales.
    Returns (n1, n2).
    """
    diff = (x1.unsqueeze(1) - x2.unsqueeze(0)) / lengthscale  # (n1, n2, P)
    sq_dist = (diff ** 2).sum(-1)
    return torch.exp(-0.5 * sq_dist)


def rbf_diag(n: int, device=None, dtype=None) -> torch.Tensor:
    """Diagonal of the RBF kernel: k(x,x) = 1 for every x."""
    return torch.ones(n, device=device, dtype=dtype)


def affine_kernel(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Affine kernel 1 + x^T x', generating affine functions of x."""
    return 1.0 + x1 @ x2.transpose(-1, -2)


def affine_diag(x: torch.Tensor) -> torch.Tensor:
    return 1.0 + (x ** 2).sum(-1)


def constant_kernel(n1: int, n2: int, device=None, dtype=None) -> torch.Tensor:
    return torch.ones(n1, n2, device=device, dtype=dtype)
