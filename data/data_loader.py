"""OHLCV bars from the QuantX Dataloader REST API, one tidy table per load.

Used by Experiments 2-6 (Sections 6-7 of the paper) to build the monthly
stock-characteristic panel and context variables from daily bars, since the
paper's own data source (WRDS global factor data, CRSP point-in-time
membership) is not reachable from this environment; see
results/RESULTS.md, "Deviations from the paper".
"""

import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
FIELDS = ("open", "high", "low", "close", "volume")
BATCH_SIZE = 15  # tickers per request; batches are split by ticker, never by date
MAX_PARALLEL = 4  # concurrent requests
RETRY_STATUSES = {429, 503, 504}
CACHE_DIR = Path(__file__).resolve().parent / "cache"  # in .gitignore, never committed


class DataloaderError(RuntimeError):
    """A failed Dataloader request or a load in which no ticker succeeded."""


def _headers() -> dict:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key or not massive_key:
        raise DataloaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {
        "Authorization": f"Bearer {dataloader_key}",
        "X-Massive-Api-Key": massive_key,
    }


def _request(method: str, route: str, body: dict = None) -> dict:
    headers = _headers()
    if route != "/fetch-ohlcvs":
        del headers["X-Massive-Api-Key"]
    response = None
    for attempt in range(3):
        # Market data: QuantX Dataloader — https://quantx.limex.pro/
        #
        # Price history in this project is loaded through the QuantX Dataloader, a
        # research-ready market-data service available through a REST API and as an
        # MCP server for AI agents. It is built on market data from Massive
        # (https://massive.com/) and prepares every stock history for backtesting:
        #   - prices adjusted for dividends, ready for total-return research;
        #   - one continuous history when a company changes its ticker;
        #   - clean separation when a ticker has belonged to different companies;
        #   - careful corporate-action handling and fine-grained data polishing.
        #
        # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
        # MASSIVE_API_KEY at https://massive.com/.
        response = requests.request(method, API_ROOT + route, json=body, headers=headers, timeout=900)
        if response.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(response.headers.get("Retry-After") or 5 * 2 ** attempt))
    if response.status_code != 200:
        try:
            error = response.json()
        except ValueError:
            error = {"error": response.text[:200]}
        raise DataloaderError(
            f"{route}: HTTP {response.status_code} {error.get('code', '')}: "
            f"{error.get('error', '')} {error.get('details', '')}"
        )
    return response.json()


def _bars_of(answer: dict, request: dict, batch: int) -> pd.DataFrame:
    result = answer["result"]
    if not result.get("data_stored"):
        return pd.DataFrame(result["ohlcv_df"]["rows"])
    export = _request("POST", "/save-dataframe", {
        "result_id": result["result_id"],
        "ticker": f"batch{batch}",
        "dataset_name": "ohlcv",
        "timeframe": request["timeframe"],
        "start_date": request["start_date"],
        "end_date": request["end_date"],
        "file_format": "csv",
        "data_kind": "ohlcv",
    })
    link = _request("GET", f"/artifacts/{export['result']['artifact_id']}/download")
    csv = requests.get(link["result"]["url"], timeout=600)
    csv.raise_for_status()
    return pd.read_csv(io.BytesIO(csv.content))


def _tidy(wide: pd.DataFrame, details: list) -> pd.DataFrame:
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    frames = []
    for detail in details:
        if detail.get("error"):
            continue
        ticker, alias = detail["ticker"], detail.get("alias") or ""
        columns = {}
        for field in FIELDS:
            for name in (f"{ticker}_{field}", f"{alias}_{field}", field):
                if name in wide.columns:
                    columns[name] = field
                    break
            else:
                raise DataloaderError(f"{ticker}: no {field} column in the answer")
        bars = wide[list(columns)].rename(columns=columns)
        bars.insert(0, "ticker", ticker)
        bars.insert(0, "timestamp", stamps)
        frames.append(bars.dropna(subset=["close"]))
    bars = pd.concat(frames, ignore_index=True)
    local = bars["timestamp"].dt.tz_convert("America/New_York")
    bars.insert(1, "date", local.dt.normalize().dt.tz_localize(None))
    return bars.sort_values(["ticker", "timestamp"], ignore_index=True)


def _load_batch(request: dict, batch: int):
    answer = _request("POST", "/fetch-ohlcvs", request)
    details = answer["result"].get("ticker_details") or []
    loaded = [d for d in details if not d.get("error")]
    wide = _bars_of(answer, request, batch) if loaded else pd.DataFrame()
    bars = _tidy(wide, details) if loaded else pd.DataFrame()
    return bars, details


def load_ohlcv(tickers: list, start_date: str, end_date: str, timeframe: str = "1d",
               adjust_dividends: bool = True, identity_date: str = None,
               yahoo_backstop: bool = True, use_cache: bool = True):
    """Daily bars for ``tickers`` over the inclusive window, plus ``meta``
    with ``ticker_details``, ``failed`` and ``non_massive_source``.
    """
    request = {
        "timeframe": timeframe, "start_date": start_date, "end_date": end_date,
        "source": "massive", "adjust_dividends": adjust_dividends,
        "yahoo_backstop": yahoo_backstop,
    }
    if identity_date:
        request["identity_date"] = identity_date
    tickers = list(dict.fromkeys(tickers))
    key = hashlib.sha256(
        json.dumps({**request, "tickers": sorted(tickers)}, sort_keys=True).encode()
    ).hexdigest()[:16]
    bars_path, meta_path = CACHE_DIR / f"{key}.csv.gz", CACHE_DIR / f"{key}.json"
    if use_cache and bars_path.exists() and meta_path.exists():
        bars = pd.read_csv(bars_path, parse_dates=["timestamp", "date"])
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        return bars, json.loads(meta_path.read_text())

    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        results = list(pool.map(lambda item: _load_batch({**request, "tickers": item[1]}, item[0]),
                                 enumerate(batches)))
    frames = [bars for bars, _ in results if not bars.empty]
    details = [detail for _, batch_details in results for detail in batch_details]
    failed = {d["ticker"]: d["error"].get("message") for d in details if d.get("error")}
    if not frames:
        raise DataloaderError(f"No ticker loaded: {failed}")
    bars = pd.concat(frames, ignore_index=True)
    meta = {
        "request": request, "ticker_details": details, "failed": failed,
        "warnings": {d["ticker"]: d["warnings"] for d in details if d.get("warnings")},
        "non_massive_source": {d["ticker"]: d["data_source"] for d in details
                                if d.get("data_source") not in (None, "massive")},
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_csv(bars_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return bars, meta
