"""Shared pytest fixtures and marker registration.

Market-data tests load a small slice of real data through
``data/data_loader.py`` and are skipped automatically when
``DATALOADER_API_KEY`` is not set in the environment.
"""

import os

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: tests that load a slice of real market data"
    )


@pytest.fixture(scope="session")
def has_market_data_keys() -> bool:
    return bool(os.environ.get("DATALOADER_API_KEY")) and bool(os.environ.get("MASSIVE_API_KEY"))


@pytest.fixture(scope="session")
def market_slice(request, has_market_data_keys):
    """A small, cached slice of daily bars for a handful of tickers, used by
    market-data tests of the characteristics/context pipeline."""
    if not has_market_data_keys:
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set")
    from data.data_loader import load_ohlcv

    cache_dir = request.config.cache.mkdir("panelgp_market_slice")
    cache_file = cache_dir / "bars.csv"
    if cache_file.exists():
        bars = pd.read_csv(cache_file, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars
    tickers = ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]
    bars, meta = load_ohlcv(tickers, "2018-01-01", "2019-12-31", timeframe="1d")
    bars.to_csv(cache_file, index=False)
    return bars


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
