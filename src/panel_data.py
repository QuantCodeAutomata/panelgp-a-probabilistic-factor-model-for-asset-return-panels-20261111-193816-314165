"""Shared monthly panel construction for Experiments 2-6 (Section 6.1-6.2).

Builds the stock-month characteristics, context and forward-return arrays
used by both PanelGP and the IPCA benchmarks, with the paper's training-only
standardization rules: within-month characteristic ranks to [-0.5, 0.5] with
missing set to zero (Section 6.1), context standardized to zero mean and unit
variance using only the training period's moments, and returns scaled without
centering by ``s_r`` computed on the training period only (Appendix C.1).
Characteristics and context of month t predict the excess return of month
t+1 (Section 6.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features import build_characteristics, build_context, monthly_returns, rank_to_unit_interval


@dataclass
class PanelData:
    months: list                       # all months with a valid t -> t+1 pair, sorted
    tickers: list                      # universe tickers, fixed cross-section
    chars_by_month: dict                # month -> (N_t, P) ranked characteristics, t
    context_by_month: dict               # month -> (Q,) standardized context, t
    returns_by_month: dict               # month -> (N_t,) realized excess return, t+1
    tickers_active_by_month: dict         # month -> list of tickers with a valid return at t+1
    train_months: list
    valid_months: list
    eval_months: list
    s_r: float                          # return scale from the training period only
    n_characteristics: int
    n_context: int


def _period(date_str: str) -> pd.Period:
    return pd.Period(date_str, freq="M")


def build_panel_dataset(
    bars: pd.DataFrame,
    tickers: list,
    train_start: str, train_end: str,
    valid_start: str, valid_end: str,
    eval_start: str, eval_end: str,
    n_characteristics: int,
    n_context: int,
) -> PanelData:
    """Build the full month-indexed panel over [train_start, eval_end].

    ``bars`` must already be restricted (or not) to ``tickers``; only
    columns for ``tickers`` are used. Context and return scale use only the
    training window [train_start, train_end].
    """
    bars = bars[bars["ticker"].isin(tickers)].copy()
    char_data = build_characteristics(bars)
    raw = char_data["raw"]
    all_months = char_data["months"]

    chosen_names = list(raw.keys())[:n_characteristics]
    ranked = {name: rank_to_unit_interval(raw[name]) for name in chosen_names}

    context_raw = build_context(bars, n_context)

    rets = monthly_returns(bars)
    rets = rets.reindex(all_months)

    train_p, valid_p = _period(train_start), _period(valid_start)
    train_end_p, valid_end_p = _period(train_end), _period(valid_end)
    eval_p, eval_end_p = _period(eval_start), _period(eval_end)

    train_months_full = [m for m in all_months if train_p <= m <= train_end_p]
    if not train_months_full:
        raise ValueError("No training months found in the loaded bars: check the date range")

    # Context standardization moments from the training period only.
    train_context = context_raw.reindex(train_months_full)
    ctx_mean = train_context.mean(axis=0)
    ctx_std = train_context.std(axis=0, ddof=1).replace(0.0, 1.0)
    context_std = (context_raw - ctx_mean) / ctx_std

    months = [m for m in all_months if train_p <= m <= eval_end_p]
    # A month t needs both characteristics/context at t and a return at t+1.
    month_to_idx = {m: i for i, m in enumerate(all_months)}
    usable_months = []
    for m in months:
        if m not in month_to_idx:
            continue
        nxt_candidates = [mm for mm in all_months if mm > m]
        if not nxt_candidates:
            continue
        nxt = min(nxt_candidates)
        if (nxt.year - m.year) * 12 + (nxt.month - m.month) != 1:
            continue  # skip months without a contiguous next month
        usable_months.append(m)

    chars_by_month, context_by_month, returns_by_month, active_by_month = {}, {}, {}, {}
    train_rows = []
    for m in usable_months:
        nxt = m + 1
        char_panel = pd.concat([ranked[name].loc[m].rename(name) for name in chosen_names], axis=1)
        ret_next = rets.loc[nxt] if nxt in rets.index else pd.Series(dtype=float)
        active = [t for t in tickers if t in char_panel.index and t in ret_next.index
                  and np.isfinite(ret_next[t])]
        if len(active) == 0:
            continue
        X = char_panel.loc[active, chosen_names].to_numpy(dtype=np.float64)
        c = context_std.loc[m].to_numpy(dtype=np.float64) if m in context_std.index else np.zeros(n_context)
        c = np.nan_to_num(c, nan=0.0)
        r = ret_next.loc[active].to_numpy(dtype=np.float64)
        chars_by_month[m] = X
        context_by_month[m] = c
        returns_by_month[m] = r
        active_by_month[m] = active
        if train_p <= m <= train_end_p:
            train_rows.append(r)

    if not train_rows:
        raise ValueError("No usable training rows after alignment")
    all_train_r = np.concatenate(train_rows)
    s_r = float(np.sqrt(np.mean(all_train_r ** 2)))

    train_months = [m for m in usable_months if train_p <= m <= train_end_p]
    valid_months = [m for m in usable_months if valid_p <= m <= valid_end_p]
    eval_months = [m for m in usable_months if eval_p <= m <= eval_end_p]

    return PanelData(
        months=usable_months, tickers=tickers,
        chars_by_month=chars_by_month, context_by_month=context_by_month,
        returns_by_month=returns_by_month, tickers_active_by_month=active_by_month,
        train_months=train_months, valid_months=valid_months, eval_months=eval_months,
        s_r=s_r, n_characteristics=n_characteristics, n_context=n_context,
    )


def stack_training_rows(panel: PanelData, months: list) -> dict:
    """Stack ``months`` into the (x, t_idx, r, c_seq) arrays PanelGPModel
    training expects: ``t_idx`` indexes into ``c_seq`` (one context row per
    training month, in order), ``r`` is scaled by ``panel.s_r``.
    """
    month_order = sorted(months)
    c_seq = np.stack([panel.context_by_month[m] for m in month_order], axis=0)
    xs, t_idx, rs = [], [], []
    for i, m in enumerate(month_order):
        X = panel.chars_by_month[m]
        r = panel.returns_by_month[m] / panel.s_r
        xs.append(X)
        t_idx.append(np.full(X.shape[0], i, dtype=np.int64))
        rs.append(r)
    return {
        "x": np.concatenate(xs, axis=0),
        "t_idx": np.concatenate(t_idx, axis=0),
        "r": np.concatenate(rs, axis=0),
        "c_seq": c_seq,
        "month_order": month_order,
    }
