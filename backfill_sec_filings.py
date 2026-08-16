"""Backfills the sec_filings table (8-K filings, 2019-present) for every
current S&P 500 ticker, from SEC EDGAR - free, no key, full history.

This exists specifically to unblock stage 2 (drop-reason classification)
from Finnhub's ~1-year free-tier news retention limit: 8-K item 2.02
("Results of Operations and Financial Condition") is the primary-source
signal for "did this company report earnings on this date", with no
retention window at all.

Usage:
    python3 backfill_sec_filings.py                  # all current S&P 500 tickers
    python3 backfill_sec_filings.py --tickers AAPL,MSFT
    python3 backfill_sec_filings.py --force           # re-fetch even if a ticker already has rows

Resumable like backfill.py: skips tickers that already have rows unless --force.

MANUAL_CIK_OVERRIDES: SEC's company_tickers.json maps a ticker to its
CURRENT registrant CIK, but a handful of large companies did a holding-
company reorganization (successor entity, new CIK) recently enough that
the current CIK's own filing history doesn't reach back to 2019 - the
real operating history sits under the OLD, pre-reorg CIK. Confirmed via
the 8-K12B "successor registrant" marker (XOM, filed 2026-07-01 under
the new CIK) and former-name continuity in SEC's company search (BLK:
current CIK 1364742 was named "BlackRock, Inc." until 2024-09-26). These
are additional CIKs to fetch and merge in, not replacements - both old
and new CIKs' filings get kept (they dedupe naturally by accession
number, which differs between CIKs, so nothing collides or is lost).
"""
import argparse
import logging
import sys

import requests

from config import BACKFILL_START_DATE, LOG_DIR
import db
from data_sources import sec_edgar

MANUAL_CIK_OVERRIDES = {
    "XOM": ["0000034088"],   # Exxon Mobil Corp (pre-2026-07 holdco reorg)
    "BLK": ["0001364742"],   # BlackRock, Inc. (renamed BlackRock Finance, Inc. 2024-09-26)
}
# Tickers absent from company_tickers.json entirely (found via SEC company search).
MANUAL_CIK_PRIMARY = {
    "AEP": "0000004904",   # American Electric Power Co Inc
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "backfill_sec_filings.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("backfill_sec_filings")


def _has_rows(conn, ticker):
    row = conn.execute("SELECT 1 FROM sec_filings WHERE ticker = ? LIMIT 1", (ticker,)).fetchone()
    return row is not None


def run(tickers, force):
    db.init_db()
    session = requests.Session()
    cik_map = sec_edgar.load_ticker_cik_map(session)

    with db.get_conn() as conn:
        for i, ticker in enumerate(tickers, 1):
            if not force and _has_rows(conn, ticker):
                log.info("[%d/%d] %s: skipped (already has rows)", i, len(tickers), ticker)
                continue
            cik10 = cik_map.get(ticker) or MANUAL_CIK_PRIMARY.get(ticker)
            if not cik10:
                log.warning("[%d/%d] %s: no CIK found in SEC company_tickers.json", i, len(tickers), ticker)
                db.log_ingestion(conn, "backfill_sec", "sec_filings", "no_cik", ticker, 0)
                conn.commit()
                continue
            ciks_to_fetch = [cik10] + MANUAL_CIK_OVERRIDES.get(ticker, [])
            try:
                rows = []
                for cik in ciks_to_fetch:
                    filings = sec_edgar.fetch_8k_filings(cik, BACKFILL_START_DATE, session)
                    rows.extend((ticker, cik, f["accession_number"], f["filing_date"],
                                 f["primary_document"], f["items"]) for f in filings)
                conn.executemany(
                    """INSERT OR REPLACE INTO sec_filings
                       (ticker, cik, accession_number, form, filing_date, primary_document, items)
                       VALUES (?, ?, ?, '8-K', ?, ?, ?)""",
                    rows,
                )
                filings = rows  # for the length-logging line below
                db.log_ingestion(conn, "backfill_sec", "sec_filings", "ok", ticker, len(filings))
                conn.commit()
                log.info("[%d/%d] %s: %d 8-K filings since %s", i, len(tickers), ticker,
                          len(filings), BACKFILL_START_DATE)
            except Exception as exc:  # noqa: BLE001 - keep going across 503 tickers
                db.log_ingestion(conn, "backfill_sec", "sec_filings", "error", ticker, 0, str(exc))
                conn.commit()
                log.exception("Failed for %s", ticker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="Comma-separated ticker list (default: current sp500_constituents)")
    parser.add_argument("--force", action="store_true")
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
        tickers = sorted(r[0] for r in rows)

    log.info("Backfilling SEC 8-K filings for %d tickers since %s", len(tickers), BACKFILL_START_DATE)
    run(tickers, args.force)
    log.info("Done.")


if __name__ == "__main__":
    main()
