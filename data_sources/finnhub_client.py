"""Rate-limited Finnhub client.

Verified against the free-tier key this project uses (2026-08-09):
  works    : /company-news, /stock/financials-reported, /stock/earnings,
             /stock/recommendation, /stock/profile2, /calendar/earnings
  403      : /stock/candle, /stock/price-target, /index/constituents
             (plan-gated - NOT transient, never retried)

Two other free-tier quirks this client works around:
  - /company-news retention is only ~1 year and caps out around ~250
    articles per call regardless of the requested date range, so a call
    covering a wide window on a heavily-covered ticker silently drops
    older articles within that window. fetch_company_news_full() detects
    this (result count near the cap) and recursively bisects the window
    until every sub-window comes back under the cap.
  - rate limit is 60 req/min, enforced via the x-ratelimit-* response
    headers rather than a fixed sleep, so we use the full budget when
    Finnhub says we can.
"""
import time
import datetime as dt
from collections import deque

import requests

from config import FINNHUB_API_KEY, FINNHUB_BASE_URL, FINNHUB_CALLS_PER_MINUTE

_NEWS_RESULT_CAP_THRESHOLD = 235  # observed hard cap ~245-250; bisect before we hit it
_MIN_NEWS_WINDOW_DAYS = 1


class NotAvailableOnPlan(Exception):
    """403 from Finnhub - this endpoint is gated on the current plan.
    Not retryable; callers should skip and log, not retry."""


class FinnhubClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or FINNHUB_API_KEY
        if not self.api_key:
            raise RuntimeError("FINNHUB_API_KEY not set (check .env)")
        self.session = requests.Session()
        self._call_times = deque()  # rolling 60s window, for our own safety margin

    def _throttle(self):
        now = time.time()
        while self._call_times and now - self._call_times[0] > 60:
            self._call_times.popleft()
        if len(self._call_times) >= FINNHUB_CALLS_PER_MINUTE:
            sleep_for = 60 - (now - self._call_times[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._call_times.append(time.time())

    def _get(self, path, params, max_retries=5):
        params = dict(params, token=self.api_key)
        url = f"{FINNHUB_BASE_URL}{path}"
        last_exc = None
        for attempt in range(max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=20)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
                continue

            if resp.status_code == 403:
                raise NotAvailableOnPlan(f"{path}: {resp.text[:200]}")
            if resp.status_code == 429:
                reset = resp.headers.get("x-ratelimit-reset")
                wait = max(float(reset) - time.time(), 1) if reset else 2 ** attempt
                time.sleep(min(wait, 65))
                continue
            if resp.status_code >= 500:
                last_exc = RuntimeError(f"{path} -> HTTP {resp.status_code}")
                time.sleep(2 ** attempt)
                continue

            resp.raise_for_status()

            remaining = resp.headers.get("x-ratelimit-remaining")
            reset = resp.headers.get("x-ratelimit-reset")
            if remaining is not None and int(remaining) <= 1 and reset:
                time.sleep(max(float(reset) - time.time(), 0) + 0.2)

            return resp.json()
        raise RuntimeError(f"Exhausted retries for {path}") from last_exc

    # ---- endpoints ----

    def company_news_raw(self, ticker, from_date, to_date):
        return self._get("/company-news", {
            "symbol": ticker, "from": from_date, "to": to_date,
        })

    def fetch_company_news_full(self, ticker, from_date, to_date,
                                 window_days=14):
        """Adaptively chunk the range so heavily-covered tickers don't
        silently lose articles to the per-call cap. Returns a de-duplicated
        list of raw article dicts (de-dup by Finnhub's article id)."""
        articles = {}

        def pull(start, end):
            if start > end:
                return
            batch = self.company_news_raw(ticker,
                                           start.strftime("%Y-%m-%d"),
                                           end.strftime("%Y-%m-%d"))
            span_days = (end - start).days + 1
            if len(batch) >= _NEWS_RESULT_CAP_THRESHOLD and span_days > _MIN_NEWS_WINDOW_DAYS:
                mid = start + dt.timedelta(days=span_days // 2)
                pull(start, mid)
                pull(mid + dt.timedelta(days=1), end)
            else:
                for art in batch:
                    articles[art["id"]] = art

        start = dt.datetime.strptime(from_date, "%Y-%m-%d").date()
        end = dt.datetime.strptime(to_date, "%Y-%m-%d").date()
        cursor = start
        while cursor <= end:
            window_end = min(cursor + dt.timedelta(days=window_days - 1), end)
            pull(cursor, window_end)
            cursor = window_end + dt.timedelta(days=1)
        return list(articles.values())

    def financials_reported(self, ticker, freq="quarterly"):
        return self._get("/stock/financials-reported",
                          {"symbol": ticker, "freq": freq})

    def earnings_surprises(self, ticker):
        return self._get("/stock/earnings", {"symbol": ticker})

    def recommendation_trends(self, ticker):
        return self._get("/stock/recommendation", {"symbol": ticker})

    def profile2(self, ticker):
        return self._get("/stock/profile2", {"symbol": ticker})
