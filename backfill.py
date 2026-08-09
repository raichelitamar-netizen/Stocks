"""Initial full-history pull into the local SQLite store.

Usage:
    python3 backfill.py                     # all current S&P 500 tickers, all tables
    python3 backfill.py --tickers AAPL,MSFT  # just these tickers (testing)
    python3 backfill.py --limit 10           # first 10 tickers only (testing)
    python3 backfill.py --only prices,news   # skip fundamentals/earnings/recs
    python3 backfill.py --force              # re-fetch even if a ticker/table already has rows

Safe to interrupt and re-run: by default it skips (ticker, table) pairs that
already have at least one row, so a killed run just picks up where it left off
without burning extra API calls on work that's already done.
"""
import argparse
import datetime as dt
import json
import logging
import sys

from config import BACKFILL_START_DATE, DATA_DIR, LOG_DIR
import db
from data_sources.finnhub_client import FinnhubClient, NotAvailableOnPlan
from data_sources.sp500_universe import fetch_constituents
from data_sources import yahoo_prices

TODAY = dt.date.today().isoformat()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "backfill.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("backfill")


def _has_rows(conn, table, ticker):
    row = conn.execute(
        f"SELECT 1 FROM {table} WHERE ticker = ? LIMIT 1", (ticker,)
    ).fetchone()
    return row is not None


def load_universe(conn):
    df = fetch_constituents()
    rows = list(df.itertuples(index=False))
    conn.executemany(
        """INSERT OR REPLACE INTO sp500_constituents
           (ticker, security, gics_sector, gics_sub_industry, date_added, snapshot_date)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )
    log.info("Loaded %d S&P 500 constituents (snapshot %s)", len(rows), TODAY)
    return df["ticker"].tolist()


def backfill_prices(conn, ticker, force=False):
    if not force and _has_rows(conn, "prices", ticker):
        return 0
    rows = yahoo_prices.fetch_ohlcv(ticker, BACKFILL_START_DATE, TODAY)
    conn.executemany(
        """INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'yahoo')""",
        [(ticker, r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"])
         for r in rows],
    )
    return len(rows)


def backfill_news(conn, client, ticker, force=False):
    if not force and _has_rows(conn, "news", ticker):
        return 0
    articles = client.fetch_company_news_full(ticker, BACKFILL_START_DATE, TODAY)
    conn.executemany(
        """INSERT OR REPLACE INTO news
           (ticker, news_id, datetime_utc, date, headline, summary, source, category, url)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(ticker, a["id"], a["datetime"],
          dt.datetime.utcfromtimestamp(a["datetime"]).strftime("%Y-%m-%d"),
          a.get("headline"), a.get("summary"), a.get("source"),
          a.get("category"), a.get("url"))
         for a in articles],
    )
    return len(articles)


def backfill_fundamentals(conn, client, ticker, force=False):
    if not force and _has_rows(conn, "fundamentals_reported", ticker):
        return 0
    payload = client.financials_reported(ticker, freq="quarterly")
    records = payload.get("data", [])
    conn.executemany(
        """INSERT OR REPLACE INTO fundamentals_reported
           (ticker, year, quarter, form, filed_date, accepted_date, start_date, end_date, report_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(ticker, rec["year"], rec["quarter"], rec.get("form"),
          rec.get("filedDate"), rec.get("acceptedDate"),
          rec.get("startDate"), rec.get("endDate"),
          json.dumps(rec.get("report", {})))
         for rec in records],
    )
    return len(records)


def backfill_earnings(conn, client, ticker, force=False):
    if not force and _has_rows(conn, "earnings_surprises", ticker):
        return 0
    records = client.earnings_surprises(ticker)
    conn.executemany(
        """INSERT OR REPLACE INTO earnings_surprises
           (ticker, period, year, quarter, eps_estimate, eps_actual, surprise, surprise_percent)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [(ticker, rec["period"], rec.get("year"), rec.get("quarter"),
          rec.get("estimate"), rec.get("actual"),
          rec.get("surprise"), rec.get("surprisePercent"))
         for rec in records],
    )
    return len(records)


def backfill_recommendations(conn, client, ticker, force=False):
    if not force and _has_rows(conn, "analyst_targets", ticker):
        return 0
    records = client.recommendation_trends(ticker)
    conn.executemany(
        """INSERT OR REPLACE INTO analyst_targets
           (ticker, period, strong_buy, buy, hold, sell, strong_sell, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'recommendation_trend')""",
        [(ticker, rec["period"], rec.get("strongBuy"), rec.get("buy"),
          rec.get("hold"), rec.get("sell"), rec.get("strongSell"))
         for rec in records],
    )
    return len(records)


TABLE_FUNCS = {
    "prices": lambda conn, client, ticker, force: backfill_prices(conn, ticker, force),
    "news": backfill_news,
    "fundamentals": backfill_fundamentals,
    "earnings": backfill_earnings,
    "recommendations": backfill_recommendations,
}


def run(tickers, only_tables, force):
    client = FinnhubClient()
    db.init_db()
    with db.get_conn() as conn:
        for i, ticker in enumerate(tickers, 1):
            counts = {}
            for table in only_tables:
                try:
                    n = TABLE_FUNCS[table](conn, client, ticker, force)
                    counts[table] = n
                    db.log_ingestion(conn, "backfill", table, "ok", ticker, n)
                except NotAvailableOnPlan as exc:
                    counts[table] = "N/A(plan)"
                    db.log_ingestion(conn, "backfill", table, "not_available_on_plan",
                                      ticker, 0, str(exc))
                except Exception as exc:  # noqa: BLE001 - keep going across 500 tickers
                    counts[table] = f"ERROR:{exc}"
                    db.log_ingestion(conn, "backfill", table, "error", ticker, 0, str(exc))
                    log.exception("Failed %s for %s", table, ticker)
            conn.commit()
            log.info("[%d/%d] %s: %s", i, len(tickers), ticker, counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="Comma-separated ticker list (overrides universe fetch)")
    parser.add_argument("--limit", type=int, help="Only process the first N tickers")
    parser.add_argument("--only", default="prices,news,fundamentals,earnings,recommendations",
                         help="Comma-separated subset of tables to backfill")
    parser.add_argument("--force", action="store_true",
                         help="Re-fetch even if the ticker already has rows in a table")
    args = parser.parse_args()

    db.init_db()
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        with db.get_conn() as conn:
            tickers = load_universe(conn)

    if args.limit:
        tickers = tickers[: args.limit]

    only_tables = [t.strip() for t in args.only.split(",")]
    log.info("Backfilling %d tickers, tables=%s, force=%s", len(tickers), only_tables, args.force)
    run(tickers, only_tables, args.force)
    log.info("Backfill complete.")


if __name__ == "__main__":
    main()
