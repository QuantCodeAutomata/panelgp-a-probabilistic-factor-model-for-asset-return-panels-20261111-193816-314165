"""Shared Adam/minibatch training loop for PanelGPModel (Appendix C.4).

Used identically by the simulation recovery study and the application
experiments: Adam with separate learning rates for variational and model
parameters, unit gradient-norm clipping, random minibatches, factor samples
drawn as independent draws paired with their negatives, and a learning-rate
schedule that halves at fixed fractions of training.
"""

from __future__ import annotations

import numpy as np
import torch

from src.config import (
    LR_VARIATIONAL, LR_MODEL, GRAD_CLIP_NORM, N_FACTOR_DRAWS, LR_HALVING_FRACTIONS,
)
from src.panelgp import PanelGPModel


def _is_variational_param(name: str) -> bool:
    return name in ("m_omega", "L_omega_raw", "L_omega_diag_raw")


def train_panelgp(model: PanelGPModel, x: np.ndarray, t_idx: np.ndarray, r: np.ndarray,
                   n_epochs: int, minibatch_rows: int, seed: int = 0,
                   log_every: int = 0) -> list:
    """Trains ``model`` in place by minimizing the negative ELBO.

    ``x``: (N, P) characteristics, ``t_idx``: (N,) integer month index into
    the model's training context grid, ``r``: (N,) scaled returns. Returns
    the per-epoch negative-ELBO loss history.
    """
    variational_params = [p for n, p in model.named_parameters() if _is_variational_param(n)]
    model_params = [p for n, p in model.named_parameters() if not _is_variational_param(n)]
    optimizer = torch.optim.Adam([
        {"params": variational_params, "lr": LR_VARIATIONAL},
        {"params": model_params, "lr": LR_MODEL},
    ])
    milestones = [max(1, int(frac * n_epochs)) for frac in LR_HALVING_FRACTIONS]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    n_train = x.shape[0]
    rng = np.random.default_rng(seed)
    history = []
    batch_size = min(minibatch_rows, n_train)
    for epoch in range(n_epochs):
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = max(1, int(np.ceil(n_train / batch_size)))
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if len(idx) == 0:
                continue
            optimizer.zero_grad()
            loss = model.elbo_terms(x[idx], t_idx[idx], r[idx], n_train, N_FACTOR_DRAWS, rng)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), GRAD_CLIP_NORM)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(idx)
        scheduler.step()
        history.append(epoch_loss / n_train)
        if log_every and (epoch % log_every == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:5d}  neg-elbo/obs = {history[-1]:.5f}")
    return history
