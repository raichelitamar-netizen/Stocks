"""Stage 2: classify the reason behind each stage-1 drop event, using
Finnhub news headlines/summaries (the only source we have - no transcript
or structured-guidance access on the free tier).

Rule, per the spec conversation:
  1. Was there a quarterly report / management announcement on the event
     day or the day before?
  2. If yes: did management LOWER full-year guidance (structural) or
     maintain/raise it (one-off)?
  3. If no report found: "external/market-wide reaction" category.

IMPORTANT SCOPE LIMITATION, stated up front:
  Finnhub's free-tier news retention is only ~1 year (see backfill.py /
  README). Of the 4,004 stage-1 events found on the full 2019-2026
  dataset, only the ones falling inside each ticker's actual news
  coverage window can be classified at all - everything older is
  excluded outright (not silently mis-labeled as "no report found",
  which would be wrong: we simply don't have the data to say either way).
  That leaves a small, recent, single-year-ish sample. This CANNOT
  answer "does this improve consistency across years" the way the
  stage 0/1 2019-2026 backtest could - there's only about one year of
  usable data, not eight. What it CAN do is show whether the
  classification separates outcomes at all within the window we have,
  and give a same-scale comparison against the unclassified stage-1
  results for the same recent events.

The classifier itself is a headline/summary keyword heuristic, not NLP/
transcript analysis - it will misclassify some events (e.g. ambiguous
wording, guidance news that didn't make an eligible headline). Precision
was checked against a real headline sample (see the "ambiguous" bucket
size reported below) rather than assumed.
"""
import re
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats

from config import DB_PATH

EARNINGS_POSITIVE = re.compile(
    r"(?i)\bearnings call\b"
    r"|\bQ[1-4]\s?20\d\d\s*(earnings|results)\b"
    r"|\breports?\s+(first|second|third|fourth|Q[1-4])[\s-]?(quarter)?\s*(20\d\d)?\s*results?\b"
    r"|\b(beats?|misses?|tops?)\b[^.]{0,30}\bestimates?\b"
    r"|\bEPS\s+(beats?|misses?)\b"
    r"|\bposts?\s+(Q[1-4]|first|second|third|fourth)[^.]{0,20}results?\b"
)
EARNINGS_NEGATIVE = re.compile(
    r"(?i)\bfund\b|\bportfolio\b|\betf\b|\bcommentary\b"
    r"|\bcontributors and detractors\b|\btracking\b.{0,20}\bportfolio\b"
)
GUIDANCE_CUT = re.compile(
    r"(?i)\b(cuts?|lowers?|slashes?|reduces?|trims?|withdraws?|pulls?|narrows?\s+down)\b"
    r"[^.]{0,30}\b(guidance|full[- ]year (outlook|forecast)|annual (outlook|forecast))\b"
)
GUIDANCE_RAISE_MAINTAIN = re.compile(
    r"(?i)\b(raises?|boosts?|hikes?|lifts?|increases?|reaffirms?|affirms?|maintains?|backs?|sticks?)\b"
    r"[^.]{0,30}\b(guidance|full[- ]year (outlook|forecast)|annual (outlook|forecast))\b"
)

CATEGORIES = [
    "structural_guidance_cut",
    "one_off_guidance_maintained_or_raised",
    "earnings_no_clear_guidance_signal",
    "external_market_reaction",
]
FORWARD_WINDOWS = (20, 60)


def load_ticker_news_coverage_start(conn):
    df = pd.read_sql_query(
        "SELECT ticker, MIN(date) AS min_news_date FROM news GROUP BY ticker", conn
    )
    return dict(zip(df["ticker"], df["min_news_date"]))


def load_news_by_ticker(conn):
    df = pd.read_sql_query("SELECT ticker, date, headline, summary FROM news", conn)
    df["text"] = (df["headline"].fillna("") + " " + df["summary"].fillna(""))
    return {t: g for t, g in df.groupby("ticker")}


def classify_event(ticker, event_date, news_by_ticker):
    g = news_by_ticker.get(ticker)
    if g is None:
        return "external_market_reaction", []
    window_start = (event_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    window_end = event_date.strftime("%Y-%m-%d")
    window_news = g[(g["date"] >= window_start) & (g["date"] <= window_end)]
    texts = window_news["text"].tolist()

    has_earnings = any(EARNINGS_POSITIVE.search(t) and not EARNINGS_NEGATIVE.search(t) for t in texts)
    if not has_earnings:
        return "external_market_reaction", texts

    has_cut = any(GUIDANCE_CUT.search(t) for t in texts)
    has_raise_maintain = any(GUIDANCE_RAISE_MAINTAIN.search(t) for t in texts)
    if has_cut:
        return "structural_guidance_cut", texts
    if has_raise_maintain:
        return "one_off_guidance_maintained_or_raised", texts
    return "earnings_no_clear_guidance_signal", texts


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
            for k in ["mean_excess_pct", "median_excess_pct", "pct_positive", "t_stat", "p_value"]:
                row[f"{k}_{n}d" if k in ("t_stat", "p_value") else f"{k}_{n}d".replace("_pct_", "_")] = np.nan
    return row


def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    events = pd.read_csv("data/stage0_1_events.csv")
    events["date"] = pd.to_datetime(events["date"])
    print(f"Stage-1 events (full 2019-2026 dataset): {len(events)}")

    conn = sqlite3.connect(DB_PATH)
    min_news_date = load_ticker_news_coverage_start(conn)
    news_by_ticker = load_news_by_ticker(conn)
    conn.close()

    events["ticker_min_news_date"] = events["ticker"].map(min_news_date)
    usable_mask = (
        events["ticker_min_news_date"].notna() &
        (events["date"] >= pd.to_datetime(events["ticker_min_news_date"]))
    )
    usable = events[usable_mask].copy()
    excluded_no_coverage = len(events) - len(usable)
    print(f"Excluded (event predates that ticker's news coverage - no data, not classified as 'external'): {excluded_no_coverage}")
    print(f"Usable events for stage 2 classification: {len(usable)}")
    print(f"Usable event date range: {usable['date'].min().date()} to {usable['date'].max().date()}")

    categories, sample_texts = [], []
    for _, row in usable.iterrows():
        cat, texts = classify_event(row["ticker"], row["date"], news_by_ticker)
        categories.append(cat)
        sample_texts.append(" || ".join(texts[:3]))
    usable["drop_reason"] = categories
    usable["sample_headlines"] = sample_texts

    print("\nCategory counts:")
    print(usable["drop_reason"].value_counts())

    usable.to_csv("data/stage2_events.csv", index=False)
    print("\nSaved data/stage2_events.csv")

    print("\n" + "=" * 70)
    print("Forward excess return by drop-reason category")
    print("=" * 70)
    rows = [summarize(g, cat) for cat, g in usable.groupby("drop_reason")]
    cat_df = pd.DataFrame(rows).set_index("group").reindex(CATEGORIES)
    cols_20 = ["n_total", "n_with_20d_data", "mean_excess_20d_pct", "median_excess_20d_pct",
               "pct_positive_20d", "t_stat_20d", "p_value_20d"]
    cols_60 = ["n_total", "n_with_60d_data", "mean_excess_60d_pct", "median_excess_60d_pct",
               "pct_positive_60d", "t_stat_60d", "p_value_60d"]
    print("\n-- 20 trading days forward --")
    print(cat_df[cols_20].round(4))
    print("\n-- 60 trading days forward --")
    print(cat_df[cols_60].round(4))
    cat_df.to_csv("data/stage2_by_category.csv")

    # Same-scale baseline: unclassified stage-1 result for the SAME usable
    # (recent) events, so the classified split is compared against a like-
    # for-like baseline rather than the full 2019-2026 numbers.
    print("\n" + "=" * 70)
    print("Baseline: same usable events, UNCLASSIFIED (stage 0/1 only)")
    print("=" * 70)
    baseline = summarize(usable, "all_usable_unclassified")
    print(pd.DataFrame([baseline]).set_index("group")[cols_20].round(4))
    print(pd.DataFrame([baseline]).set_index("group")[cols_60].round(4))

    # Sub-period split within the ~1 year we have - NOT a substitute for
    # multi-year testing, just whatever granularity the window allows.
    print("\n" + "=" * 70)
    print("Sub-period split within the usable window (quarterly) - NOT multi-year, just what we have")
    print("=" * 70)
    usable["period"] = usable["date"].dt.to_period("Q").astype(str)
    for cat in CATEGORIES:
        sub_cat = usable[usable["drop_reason"] == cat]
        if len(sub_cat) == 0:
            continue
        print(f"\n--- {cat} ---")
        period_rows = [summarize(g, p) for p, g in sub_cat.groupby("period")]
        period_df = pd.DataFrame(period_rows).set_index("group")
        print(period_df[["n_total", "mean_excess_20d_pct", "p_value_20d",
                          "mean_excess_60d_pct", "p_value_60d"]].round(4))


if __name__ == "__main__":
    main()
