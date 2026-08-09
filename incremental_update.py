"""Delta-only update, meant to be run daily/weekly on a schedule (cron etc.).

For prices and news (the two large, date-indexed tables) it looks up the
latest date already stored per ticker and only pulls what's newer.
Fundamentals/earnings/recommendations are cheap (a handful of rows per
ticker) so they're simply re-pulled and upserted in full each run - that
also picks up SEC filing restatements and analyst-count revisions for
free, which a delta-only approach would miss.

Usage:
    python3 incremental_update.py                    # all tickers currently in sp500_constituents
    python3 incremental_update.py --tickers AAPL,MSFT
"""
import argparse
import datetime as dt
import logging
import sys

from config import BACKFILL_START_DATE, LOG_DIR
import db
from backfill import (
    backfill_fundamentals, backfill_earnings, backfill_recommendations,
)
from data_sources.finnhub_client import FinnhubClient, NotAvailableOnPlan
from data_sources import yahoo_prices

TODAY = dt.date.today().isoformat()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "incremental_update.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("incremental_update")


def update_prices(conn, ticker):
    last = db.last_date_for_ticker(conn, "prices", ticker)
    start = BACKFILL_START_DATE if last is None else (
        dt.date.fromisoformat(last) + dt.timedelta(days=1)
    ).isoformat()
    if start > TODAY:
        return 0
    rows = yahoo_prices.fetch_ohlcv(ticker, start, TODAY)
    conn.executemany(
        """INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'yahoo')""",
        [(ticker, r["date"], r["open"], r["high"], r["low"], r["close"], r["volume"])
         for r in rows],
    )
    return len(rows)


def update_news(conn, client, ticker):
    last = db.last_date_for_ticker(conn, "news", ticker)
    start = BACKFILL_START_DATE if last is None else last  # re-pull last day too, dedup by news_id
    if start > TODAY:
        return 0
    articles = client.fetch_company_news_full(ticker, start, TODAY)
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


def run(tickers):
    client = FinnhubClient()
    db.init_db()
    with db.get_conn() as conn:
        for i, ticker in enumerate(tickers, 1):
            counts = {}
            steps = {
                "prices": lambda: update_prices(conn, ticker),
                "news": lambda: update_news(conn, client, ticker),
                "fundamentals": lambda: backfill_fundamentals(conn, client, ticker, force=True),
                "earnings": lambda: backfill_earnings(conn, client, ticker, force=True),
                "recommendations": lambda: backfill_recommendations(conn, client, ticker, force=True),
            }
            for table, fn in steps.items():
                try:
                    n = fn()
                    counts[table] = n
                    db.log_ingestion(conn, "incremental", table, "ok", ticker, n)
                except NotAvailableOnPlan as exc:
                    counts[table] = "N/A(plan)"
                    db.log_ingestion(conn, "incremental", table, "not_available_on_plan",
                                      ticker, 0, str(exc))
                except Exception as exc:  # noqa: BLE001
                    counts[table] = f"ERROR:{exc}"
                    db.log_ingestion(conn, "incremental", table, "error", ticker, 0, str(exc))
                    log.exception("Failed %s for %s", table, ticker)
            conn.commit()
            log.info("[%d/%d] %s: %s", i, len(tickers), ticker, counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="Comma-separated ticker list (default: full sp500_constituents table)")
    args = parser.parse_args()

    db.init_db()
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM sp500_constituents "
                "WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM sp500_constituents)"
            ).fetchall()
        tickers = [r[0] for r in rows]
        if not tickers:
            log.error("No tickers found in sp500_constituents - run backfill.py first.")
            sys.exit(1)

    log.info("Incremental update for %d tickers", len(tickers))
    run(tickers)
    log.info("Incremental update complete.")


if __name__ == "__main__":
    main()
