"""Substitute equity universe for Experiments 2-4 (Section 6.1).

The paper draws its cross-section from historical S&P 500 membership
(CRSP) together with the WRDS global factor characteristics of Jensen,
Kelly and Pedersen (2023). Neither point-in-time index membership nor that
characteristics panel is reachable through the QuantX Dataloader or Massive
REST API in this environment (see results/RESULTS.md, "Deviations from the
paper"). The executed run instead uses a fixed list of large, long-history
U.S. common stocks, candidates for the paper's own selection rule (the most
liquid names), and keeps only the ``APP_UNIVERSE_SIZE_RUN`` names with the
highest average dollar volume over the first training year -- the
liquidity-ranking fallback this project applies whenever a paper's universe
cannot be downloaded in full within the available time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Large-capitalization U.S. common stocks across sectors with trading
# histories reaching back to at least the executed training-window start
# (APP_TRAIN_START_RUN = 2005-01-01). This is a convenience candidate list,
# not a historical S&P 500 membership file; see the deviation note above.
CANDIDATE_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "JNJ", "V", "PG", "XOM",
    "UNH", "HD", "MA", "BAC", "DIS", "ADBE", "CSCO", "CRM", "NFLX", "PFE",
    "KO", "PEP", "ABT", "TMO", "CMCSA", "COST", "NKE", "MRK", "WMT", "MCD",
    "ORCL", "ACN", "AVGO", "TXN", "LIN", "HON", "UPS", "QCOM", "LOW", "UNP",
    "IBM", "SBUX", "GS", "CAT", "AMGN", "BA", "INTU", "ISRG", "DE", "BLK",
    "AXP", "SPGI", "GE", "MDT", "NOW", "BKNG", "PLD", "GILD", "ADI", "MMC",
    "SYK", "TJX", "MO", "CB", "ADP", "VRTX", "REGN", "SCHW", "ZTS", "CI",
    "SO", "DUK", "BDX", "PGR", "CL", "ITW", "APD", "NOC", "FDX", "EMR",
    "ETN", "AON", "SHW", "MMM", "WM", "GD", "TGT", "PNC", "USB", "CSX",
    "NSC", "ECL", "MCO", "HUM", "F", "GM", "DOW", "D", "PSA", "EXC",
    "KMB", "ALL", "AIG", "MET", "AFL", "TRV", "PRU", "STT", "BK", "COF",
    "HCA", "CVS", "ABBV", "VZ", "T", "XEL", "AEP", "WEC", "ED", "PEG",
]


def select_liquid_universe(bars: pd.DataFrame, as_of_start: str, as_of_end: str,
                            n_select: int) -> list:
    """Rank candidate tickers by average dollar volume over one window and
    keep the ``n_select`` most liquid -- the fallback ranking rule used when
    the paper's own universe is not reachable in full.
    """
    window = bars[(bars["date"] >= as_of_start) & (bars["date"] <= as_of_end)].copy()
    window["dollar_volume"] = window["close"] * window["volume"]
    ranked = (
        window.groupby("ticker")["dollar_volume"].mean().sort_values(ascending=False)
    )
    n_available = int((ranked > 0).sum())
    if n_available < n_select:
        raise ValueError(f"Only {n_available} liquid tickers available, need {n_select}")
    return ranked.index[:n_select].tolist()
