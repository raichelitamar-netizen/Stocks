"""Shared configuration: paths, API settings, and the stage 0-1 thresholds
from stock_screener_agent.md (kept here as the single source of truth so
the analysis scripts and the ingestion scripts never drift apart).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
DB_PATH = DATA_DIR / "stocks.db"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

load_dotenv(ROOT_DIR / ".env")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
FINNHUB_CALLS_PER_MINUTE = 55  # actual plan limit is 60/min; keep a safety margin

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YAHOO_REQUEST_DELAY_SECONDS = 0.3

# Backfill window. Finnhub news/earnings/recommendation retention on the
# free tier is much shorter than this (see README) - prices and fundamentals
# cover the full range, news/earnings/recommendations will simply come back
# empty for dates outside their retention window.
BACKFILL_START_DATE = "2019-01-01"

# --- Stage 0/1 thresholds, verbatim from stock_screener_agent.md ---
DROP_THRESHOLD_PCT = -7.0          # stage 1 step 2
LIQUIDITY_WINDOW_DAYS = 60         # stage 1 step 3
LIQUIDITY_MIN_AVG_VOLUME = 100_000  # stage 1 step 3
DEDUP_WINDOW_TRADING_DAYS = 20      # stage 1 step 4
MARKET_TREND_WINDOW_DAYS = 60       # stage 0 step 2
