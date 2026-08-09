"""Daily OHLCV via Yahoo Finance's public chart API.

Finnhub's /stock/candle returns 403 ("You don't have access to this
resource") on the free tier this project uses, so historical prices come
from here instead. This is an unofficial/undocumented endpoint (the same
one the `yfinance` package wraps) - no API key, but be polite about
request rate and resilient to transient failures.
"""
import time
import datetime as dt
import requests

from config import YAHOO_CHART_URL, YAHOO_REQUEST_DELAY_SECONDS

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_MAX_RETRIES = 4


def _date_to_epoch(date_str):
    return int(dt.datetime.strptime(date_str, "%Y-%m-%d")
                .replace(tzinfo=dt.timezone.utc).timestamp())


def fetch_ohlcv(ticker, start_date, end_date, session=None):
    """Returns a list of dicts: date, open, high, low, close, volume.
    Raises requests.HTTPError / ValueError after exhausting retries.
    """
    sess = session or requests.Session()
    params = {
        "period1": _date_to_epoch(start_date),
        "period2": _date_to_epoch(end_date) + 86400,
        "interval": "1d",
        "events": "none",
    }
    url = YAHOO_CHART_URL.format(ticker=ticker)

    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = sess.get(url, params=params, headers=_HEADERS, timeout=20)
            if resp.status_code == 404:
                return []  # delisted / unknown ticker - not retryable
            resp.raise_for_status()
            payload = resp.json()
            result = payload.get("chart", {}).get("result")
            if not result:
                err = payload.get("chart", {}).get("error")
                raise ValueError(f"Yahoo chart error for {ticker}: {err}")
            res = result[0]
            timestamps = res.get("timestamp") or []
            quote = res["indicators"]["quote"][0]
            rows = []
            for i, ts in enumerate(timestamps):
                close = quote["close"][i]
                if close is None:
                    continue  # non-trading gap in the series
                rows.append({
                    "date": dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
                              .strftime("%Y-%m-%d"),
                    "open": quote["open"][i],
                    "high": quote["high"][i],
                    "low": quote["low"][i],
                    "close": close,
                    "volume": quote["volume"][i],
                })
            time.sleep(YAHOO_REQUEST_DELAY_SECONDS)
            return rows
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch OHLCV for {ticker}") from last_exc
