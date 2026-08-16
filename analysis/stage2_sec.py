"""Stage 2 (drop-reason classification), v2 - SEC EDGAR based, full
2019-2026 coverage (all 4,004 stage-1 events, not just the 585 that fell
inside Finnhub's ~1-year news retention window - see stage2_drop_reason.py
for that earlier, smaller-sample version).

Rule (same as stage2_drop_reason.py):
  1. Was there an earnings-related 8-K (item 2.02 "Results of Operations
     and Financial Condition", or 7.01 "Reg FD Disclosure" for ad hoc
     guidance updates outside the normal cycle) on the event day or the
     day before?
  2. If yes: fetch the filing's primary document + press-release exhibit,
     and classify guidance direction by keyword search on the OFFICIAL
     text - cut (structural) vs maintained/raised (one-off) vs no clear
     signal.
  3. If no matching 8-K: external/market-wide reaction.

Resumable: classification results are appended to data/stage2_sec_events.csv
as they're computed; a rerun skips (ticker, date) pairs already present,
so an interrupted run (this environment has had background jobs killed by
container restarts before) picks up where it left off without re-fetching
SEC documents that were already classified.
"""
import csv
import os
import re
import time
import sqlite3
import numpy as np
import pandas as pd
import requests
from scipy import stats

from config import DB_PATH
from data_sources import sec_edgar

OUTPUT_CSV = "data/stage2_sec_events.csv"
FORWARD_WINDOWS = (20, 60)
EARNINGS_ITEMS = ("2.02", "7.01")

# SEC filing text is guaranteed on-topic (it's the company's own press
# release), unlike news headlines which can mention unrelated macro
# "outlook"/"forecast" - so bare outlook/forecast triggers are safe here
# (see stage2_drop_reason.py's tighter, headline-safe patterns for contrast).
# Official press releases lean heavily on the gerund ("Maintaining full-year
# guidance", "Lowering its outlook") as much as simple present ("cuts",
# "maintains") - a first pass using only simple-present forms silently
# missed real, unambiguous EFX ("Maintaining...guidance") and HII
# ("Affirms...guidance") language, so every verb below covers both forms
# explicitly rather than trying to be clever with a generic suffix.
_CUT_VERBS = (
    r"(?:cuts?|cutting|lowers?|lowering|slashes?|slashing|reduces?|reducing|"
    r"trims?|trimming|withdraws?|withdrawing|pulls?|pulling|"
    r"narrows?\s+down|narrowing\s+down|decreases?|decreasing|"
    r"revises?\s+down(?:ward)?|revising\s+down(?:ward)?)"
)
_RAISE_MAINTAIN_VERBS = (
    r"(?:raises?|raising|boosts?|boosting|hikes?|hiking|lifts?|lifting|"
    r"increases?|increasing|reaffirms?|reaffirming|affirms?|affirming|"
    r"maintains?|maintaining|backs?|backing|sticks?|sticking|"
    r"reiterates?|reiterating|confirms?|confirming|"
    r"revises?\s+up(?:ward)?|revising\s+up(?:ward)?)"
)
GUIDANCE_CUT = re.compile(
    rf"(?i)\b{_CUT_VERBS}\b[^.]{{0,60}}\b(guidance|outlook|forecast)\b"
)
GUIDANCE_RAISE_MAINTAIN = re.compile(
    rf"(?i)\b{_RAISE_MAINTAIN_VERBS}\b[^.]{{0,60}}\b(guidance|outlook|forecast)\b"
)

CATEGORIES = [
    "structural_guidance_cut",
    "one_off_guidance_maintained_or_raised",
    "earnings_no_clear_guidance_signal",
    "external_market_reaction",
]


def load_earnings_filings_by_ticker(conn):
    df = pd.read_sql_query(
        "SELECT ticker, cik, accession_number, filing_date, primary_document, items FROM sec_filings",
        conn,
    )
    df = df[df["items"].apply(lambda s: any(it in (s or "") for it in EARNINGS_ITEMS))]
    return {t: g.sort_values("filing_date") for t, g in df.groupby("ticker")}


def find_matching_filing(ticker, event_date, filings_by_ticker):
    g = filings_by_ticker.get(ticker)
    if g is None:
        return None
    d0 = (event_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    d1 = event_date.strftime("%Y-%m-%d")
    match = g[(g["filing_date"] >= d0) & (g["filing_date"] <= d1)]
    if len(match) == 0:
        return None
    return match.iloc[-1].to_dict()  # if >1 same-window filing, the later one is most relevant


def classify_event(ticker, event_date, filings_by_ticker, session, text_cache):
    filing = find_matching_filing(ticker, event_date, filings_by_ticker)
    if filing is None:
        return "external_market_reaction", None
    accn = filing["accession_number"]
    if accn not in text_cache:
        text_cache[accn] = sec_edgar.fetch_filing_text(
            filing["cik"], accn, filing["primary_document"], session
        )
    text = text_cache[accn]
    if GUIDANCE_CUT.search(text):
        return "structural_guidance_cut", accn
    if GUIDANCE_RAISE_MAINTAIN.search(text):
        return "one_off_guidance_maintained_or_raised", accn
    return "earnings_no_clear_guidance_signal", accn


def load_already_done():
    if not os.path.exists(OUTPUT_CSV):
        return set(), []
    rows = pd.read_csv(OUTPUT_CSV)
    done = set(zip(rows["ticker"], rows["date"]))
    return done, rows.to_dict("records")


def append_row(row):
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def summarize(sub, label):
    row = {"group": label, "n_total": len(sub)}
    for n in FORWARD_WINDOWS:
        col = f"excess_fwd_{n}d"
        vals = sub[col].dropna()
        row[f"n_with_{n}d_data"] = len(vals)
        if len(vals) >= 2:
            t_stat, p_val = stats.ttest_1samp(vals, 0.0)
            row[f"mean_excess_{n}d_pct"] = vals.mean() * 100
            row[f"median_excess_{n}d_pct"] = vals.median() * 100
            row[f"pct_positive_{n}d"] = (vals > 0).mean() * 100
            row[f"t_stat_{n}d"] = t_stat
            row[f"p_value_{n}d"] = p_val
        else:
            for k in ("mean_excess_pct", "median_excess_pct", "pct_positive", "t_stat", "p_value"):
                key = f"{k}_{n}d" if k in ("t_stat", "p_value") else f"{k}_{n}d".replace("_pct_", "_")
                row[key] = np.nan
    return row


def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    events = pd.read_csv("data/stage0_1_events.csv")
    events["date"] = pd.to_datetime(events["date"])
    print(f"Stage-1 events (full 2019-2026 dataset): {len(events)}")

    conn = sqlite3.connect(DB_PATH)
    filings_by_ticker = load_earnings_filings_by_ticker(conn)
    conn.close()

    done_keys, done_rows = load_already_done()
    print(f"Already classified from a prior run (resuming): {len(done_keys)}")

    session = requests.Session()
    text_cache = {}
    results = list(done_rows)
    t0 = time.time()
    n_new = 0
    for i, row in events.iterrows():
        key = (row["ticker"], row["date"].strftime("%Y-%m-%d"))
        if key in done_keys:
            continue
        cat, accn = classify_event(row["ticker"], row["date"], filings_by_ticker, session, text_cache)
        out_row = row.to_dict()
        out_row["date"] = row["date"].strftime("%Y-%m-%d")
        out_row["drop_reason"] = cat
        out_row["sec_accession"] = accn or ""
        append_row(out_row)
        results.append(out_row)
        n_new += 1
        if n_new % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {n_new} new events classified ({i+1}/{len(events)} scanned), "
                  f"{elapsed:.0f}s elapsed, cache size {len(text_cache)}")

    print(f"\nTotal classified this run: {n_new} new + {len(done_keys)} resumed = {len(results)}")

    out = pd.DataFrame(results)
    out["date"] = pd.to_datetime(out["date"])
    for n in FORWARD_WINDOWS:
        out[f"excess_fwd_{n}d"] = pd.to_numeric(out[f"excess_fwd_{n}d"], errors="coerce")

    print("\nCategory counts (full 4,004-event sample):")
    print(out["drop_reason"].value_counts())

    print("\n" + "=" * 70)
    print("Forward excess return by drop-reason category (SEC-based, full sample)")
    print("=" * 70)
    rows = [summarize(g, cat) for cat, g in out.groupby("drop_reason")]
    cat_df = pd.DataFrame(rows).set_index("group").reindex(CATEGORIES)
    cols_20 = ["n_total", "n_with_20d_data", "mean_excess_20d_pct", "median_excess_20d_pct",
               "pct_positive_20d", "t_stat_20d", "p_value_20d"]
    cols_60 = ["n_total", "n_with_60d_data", "mean_excess_60d_pct", "median_excess_60d_pct",
               "pct_positive_60d", "t_stat_60d", "p_value_60d"]
    print("\n-- 20 trading days forward --")
    print(cat_df[cols_20].round(4))
    print("\n-- 60 trading days forward --")
    print(cat_df[cols_60].round(4))
    cat_df.to_csv("data/stage2_sec_by_category.csv")

    print("\n" + "=" * 70)
    print("BY YEAR within each category (the real multi-year consistency test)")
    print("=" * 70)
    out["year"] = out["date"].dt.year
    for cat in CATEGORIES:
        sub_cat = out[out["drop_reason"] == cat]
        print(f"\n--- {cat} (n={len(sub_cat)}) ---")
        year_rows = [summarize(g, str(y)) for y, g in sub_cat.groupby("year")]
        year_df = pd.DataFrame(year_rows).set_index("group")
        print(year_df[["n_total", "mean_excess_20d_pct", "p_value_20d",
                        "mean_excess_60d_pct", "p_value_60d"]].round(4))

    print("\nSaved data/stage2_sec_events.csv and data/stage2_sec_by_category.csv")


if __name__ == "__main__":
    main()
