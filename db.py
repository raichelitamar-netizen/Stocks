"""SQLite schema and connection helper for the local market-data store.

Table-to-source mapping (see README.md for the full rationale):
  prices              <- Yahoo Finance chart API (Finnhub's /stock/candle is
                         403 on the free tier we have)
  news                <- Finnhub /company-news (free tier: ~1y retention,
                         ~250 articles/call cap - see data_sources/finnhub_client.py)
  fundamentals_reported <- Finnhub /stock/financials-reported (free tier:
                         full history back to ~2010, no gating observed)
  earnings_surprises  <- Finnhub /stock/earnings (free tier: ~4 quarters only)
  analyst_targets     <- Finnhub /stock/recommendation (free tier: ~4 months
                         only; /stock/price-target is 403, so target_high/
                         low/mean/median stay NULL until a paid plan exposes it)
  sp500_constituents  <- Wikipedia "List of S&P 500 companies" (Finnhub's
                         /index/constituents is 403 on the free tier)
"""
import sqlite3
from contextlib import contextmanager

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    source TEXT NOT NULL DEFAULT 'yahoo',
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date);

CREATE TABLE IF NOT EXISTS news (
    ticker TEXT NOT NULL,
    news_id INTEGER NOT NULL,
    datetime_utc INTEGER NOT NULL,
    date TEXT NOT NULL,
    headline TEXT,
    summary TEXT,
    source TEXT,
    category TEXT,
    url TEXT,
    PRIMARY KEY (ticker, news_id)
);
CREATE INDEX IF NOT EXISTS idx_news_ticker_date ON news(ticker, date);

CREATE TABLE IF NOT EXISTS fundamentals_reported (
    ticker TEXT NOT NULL,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    form TEXT,
    filed_date TEXT,
    accepted_date TEXT,
    start_date TEXT,
    end_date TEXT,
    report_json TEXT NOT NULL,
    PRIMARY KEY (ticker, year, quarter)
);

CREATE TABLE IF NOT EXISTS earnings_surprises (
    ticker TEXT NOT NULL,
    period TEXT NOT NULL,
    year INTEGER,
    quarter INTEGER,
    eps_estimate REAL,
    eps_actual REAL,
    surprise REAL,
    surprise_percent REAL,
    PRIMARY KEY (ticker, period)
);

CREATE TABLE IF NOT EXISTS analyst_targets (
    ticker TEXT NOT NULL,
    period TEXT NOT NULL,
    strong_buy INTEGER,
    buy INTEGER,
    hold INTEGER,
    sell INTEGER,
    strong_sell INTEGER,
    target_high REAL,
    target_low REAL,
    target_mean REAL,
    target_median REAL,
    source TEXT NOT NULL DEFAULT 'recommendation_trend',
    PRIMARY KEY (ticker, period)
);

CREATE TABLE IF NOT EXISTS sp500_constituents (
    ticker TEXT NOT NULL,
    security TEXT,
    gics_sector TEXT,
    gics_sub_industry TEXT,
    date_added TEXT,
    snapshot_date TEXT NOT NULL,
    PRIMARY KEY (ticker, snapshot_date)
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,
    ticker TEXT,
    table_name TEXT NOT NULL,
    status TEXT NOT NULL,
    rows_affected INTEGER DEFAULT 0,
    message TEXT,
    run_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ingestion_log_ticker_table
    ON ingestion_log(ticker, table_name, run_at);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def log_ingestion(conn, run_type, table_name, status, ticker=None,
                   rows_affected=0, message=None):
    conn.execute(
        """INSERT INTO ingestion_log
           (run_type, ticker, table_name, status, rows_affected, message)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run_type, ticker, table_name, status, rows_affected, message),
    )


def last_date_for_ticker(conn, table_name, ticker, date_column="date"):
    """Return the max date/period already stored for a ticker in a table,
    or None if nothing is stored yet. Used by incremental_update.py."""
    row = conn.execute(
        f"SELECT MAX({date_column}) FROM {table_name} WHERE ticker = ?",
        (ticker,),
    ).fetchone()
    return row[0] if row else None


if __name__ == "__main__":
    init_db()
    print(f"Initialized schema at {DB_PATH}")
