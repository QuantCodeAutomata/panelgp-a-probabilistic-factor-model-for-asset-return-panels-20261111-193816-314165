"""Monthly characteristic and context panel built from daily OHLCV bars.

The paper's cross-section uses 137 of the characteristics of Jensen, Kelly
and Pedersen (2023), ranked within each month to [-0.5, 0.5] with missing
values set to zero (Section 6.1), and six context variables of Goyal,
Welch and Zafirov (2024), standardized with training-period moments. Since
neither source is reachable from this environment (see
results/RESULTS.md), the executed run computes a reduced set of
characteristics and context variables directly from price and volume, and
applies the same within-month ranking and standardization rules the paper
specifies for its own panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_MONTH = 21  # approximate, for rolling-window lengths


def _month_end_index(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M")


def monthly_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Monthly close-to-close simple returns per ticker, indexed by the
    last trading day of each month."""
    bars = bars.sort_values(["ticker", "date"]).copy()
    bars["month"] = _month_end_index(bars["date"])
    last_of_month = bars.groupby(["ticker", "month"]).tail(1)
    monthly = last_of_month.pivot(index="month", columns="ticker", values="close")
    monthly = monthly.sort_index()
    rets = monthly.pct_change()
    return rets


def _rolling_feature(daily_close: pd.DataFrame, window: int, kind: str) -> pd.DataFrame:
    """A daily rolling statistic on log returns, sampled at month end."""
    log_ret = np.log(daily_close).diff()
    if kind == "vol":
        stat = log_ret.rolling(window).std() * np.sqrt(252)
    elif kind == "mom":
        stat = daily_close.pct_change(window)
    elif kind == "skew":
        stat = log_ret.rolling(window).skew()
    elif kind == "maxret":
        stat = log_ret.rolling(window).max()
    elif kind == "minret":
        stat = log_ret.rolling(window).min()
    else:
        raise ValueError(kind)
    return stat


def build_characteristics(bars: pd.DataFrame) -> dict:
    """Build the stock-month characteristics panel from daily bars.

    Returns a dict with:
    - ``raw``: dict of {name: DataFrame[month x ticker]} unranked characteristics
    - ``months``: sorted PeriodIndex of available months
    - ``tickers``: list of tickers
    """
    bars = bars.sort_values(["ticker", "date"]).copy()
    close = bars.pivot(index="date", columns="ticker", values="close")
    volume = bars.pivot(index="date", columns="ticker", values="volume")
    dollar_vol = close * volume

    raw_daily = {
        "mom_1m": _rolling_feature(close, 21, "mom"),
        "mom_3m": _rolling_feature(close, 63, "mom"),
        "mom_6m": _rolling_feature(close, 126, "mom"),
        "mom_12m": _rolling_feature(close, 252, "mom"),
        "reversal_1w": _rolling_feature(close, 5, "mom"),
        "vol_1m": _rolling_feature(close, 21, "vol"),
        "vol_3m": _rolling_feature(close, 63, "vol"),
        "vol_12m": _rolling_feature(close, 252, "vol"),
        "skew_3m": _rolling_feature(close, 63, "skew"),
        "maxret_1m": _rolling_feature(close, 21, "maxret"),
        "minret_1m": _rolling_feature(close, 21, "minret"),
        "dollar_volume": np.log(dollar_vol.rolling(21).mean()),
        "price_level": np.log(close),
        "high52w_ratio": close / close.rolling(252).max(),
        "turnover_3m": volume.rolling(63).mean() / volume.rolling(252).mean(),
        "illiquidity": (np.log(close).diff().abs() / (dollar_vol + 1.0)).rolling(21).mean(),
    }

    months = _month_end_index(bars["date"]).unique()
    months = pd.PeriodIndex(sorted(months), freq="M")
    month_end_dates = close.index.to_series().groupby(close.index.to_period("M")).last()

    raw_monthly = {}
    for name, daily in raw_daily.items():
        sampled = daily.loc[daily.index.isin(month_end_dates.values)].copy()
        sampled.index = sampled.index.to_period("M")
        raw_monthly[name] = sampled.reindex(months)

    return {"raw": raw_monthly, "months": months, "tickers": list(close.columns)}


def rank_to_unit_interval(panel: pd.DataFrame) -> pd.DataFrame:
    """Within-month cross-sectional rank mapped to [-0.5, 0.5]; missing
    values are set to zero after ranking (Section 6.1)."""
    ranks = panel.rank(axis=1, method="average", na_option="keep")
    counts = panel.notna().sum(axis=1)
    scaled = ranks.sub(0.5).div(counts, axis=0).sub(0.5)
    return scaled.fillna(0.0)


def build_context(bars: pd.DataFrame, n_context: int) -> pd.DataFrame:
    """Context variables computed from the panel itself (Section 6.1 lists
    market return, term spread, default-yield spread, log realized market
    variance, average correlation, and lagged inflation; only the first,
    fourth and fifth can be derived from the available price panel, so a
    momentum-spread proxy is added as a fourth variable to keep the model's
    context dimension comparable to the paper's). Context must be
    standardized with training-period moments by the caller.
    """
    close = bars.pivot(index="date", columns="ticker", values="close")
    daily_ret = close.pct_change()
    monthly = daily_ret.copy()
    monthly["month"] = monthly.index.to_period("M")

    market_ret = monthly.groupby("month").apply(
        lambda g: g.drop(columns="month").mean(axis=1).add(1).prod() - 1
    )
    realized_var = np.log(
        monthly.groupby("month").apply(lambda g: g.drop(columns="month").var(axis=1).mean()) + 1e-8
    )
    corr_by_month = monthly.groupby("month").apply(
        lambda g: _average_pairwise_corr(g.drop(columns="month"))
    )
    mom_spread = close.pct_change(126)
    mom_spread_month = mom_spread.copy()
    mom_spread_month["month"] = mom_spread_month.index.to_period("M")
    mom_spread_monthly = mom_spread_month.groupby("month").apply(
        lambda g: g.drop(columns="month").std(axis=1, skipna=True).mean()
    )

    context = pd.DataFrame({
        "market_return": market_ret,
        "log_realized_variance": realized_var,
        "average_correlation": corr_by_month,
        "momentum_spread": mom_spread_monthly,
    })
    return context.iloc[:, :n_context]


def _average_pairwise_corr(frame: pd.DataFrame) -> float:
    valid = frame.dropna(axis=1, thresh=max(2, int(frame.shape[0] * 0.5)))
    if valid.shape[1] < 2:
        return np.nan
    corr = valid.corr().values
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.nanmean(corr[mask]))
