"""Stage 0-1 screener, implemented verbatim from stock_screener_agent.md,
run on the full local dataset (2019-01-02 through today, current S&P 500
constituents, prices from Yahoo via the local SQLite store).

Stage 1 (drop-event detection):
  1. Daily return per ticker, close-to-close.
  2. Candidate event: daily return <= -7%.
  3. Liquidity filter: 60-trading-day average volume (the 60 days strictly
     before the event) > 100,000 shares.
  4. Dedup: skip if the same ticker already has a kept event within the
     preceding 20 trading days.
  5. Output: (ticker, date, close, pct_drop).

Stage 0 (market regime filter):
  1. Equal-weight market proxy: mean daily return across all tickers, per date.
  2. market_trend_60d(t) = compounded market return over the 60 trading days
     strictly before t.
  3. Split ALL historical dates into terciles of market_trend_60d: weak / neutral / strong.
  4. Label each stage-1 event with the tercile of its event date.

For every surviving event we also compute forward stock return and forward
market return over the next 20 and 60 trading days, and report EXCESS
return (stock forward return minus the market proxy's forward return over
the same window) - this isolates whether the stock recovers better than
the market was doing anyway, rather than conflating the two.

Caveats baked into this run, stated up front rather than buried:
  - Survivorship bias: this uses TODAY's S&P 500 membership applied
    retroactively to 2019-2026. Tickers that left the index (bankruptcy,
    acquisition, demotion) in that window are absent. This was also true
    of the original 2013-2018 Kaggle-based calibration, so the two runs
    are at least consistent with each other on this dimension - but it
    does mean both are biased toward survivors.
  - Events in roughly the last ~90 calendar days (2026-08-10 minus ~60
    trading days) don't have a full 60-trading-day forward window yet;
    they're counted in the raw event totals but excluded from forward-
    return / significance statistics, and this is reported explicitly
    rather than silently dropped.
"""
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats

from config import (
    DB_PATH, DROP_THRESHOLD_PCT, LIQUIDITY_WINDOW_DAYS,
    LIQUIDITY_MIN_AVG_VOLUME, DEDUP_WINDOW_TRADING_DAYS,
    MARKET_TREND_WINDOW_DAYS,
)

FORWARD_WINDOWS = (20, 60)


def load_panel():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT ticker, date, close, volume FROM prices", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    volume = df.pivot(index="date", columns="ticker", values="volume").sort_index()
    return close, volume


def sanity_check_extreme_returns(daily_ret, threshold=0.50):
    """Flag single-day moves bigger than `threshold` in either direction -
    the kind of thing an unhandled stock split or bad tick would produce.
    Printed for manual review, not auto-excluded."""
    flat = daily_ret.stack()
    extreme = flat[flat.abs() > threshold].sort_values()
    print(f"\n=== Sanity check: {len(extreme)} single-day moves > {threshold:.0%} in the whole panel ===")
    if len(extreme):
        print(extreme.head(15))
        print("...")
        print(extreme.tail(15))
    return extreme


def compute_market_proxy(daily_ret):
    market_ret = daily_ret.mean(axis=1, skipna=True)
    log_ret = np.log1p(market_ret)
    trend_log_sum = log_ret.rolling(MARKET_TREND_WINDOW_DAYS).sum().shift(1)
    market_trend_60d = np.expm1(trend_log_sum)
    market_cumret = (1 + market_ret).cumprod()
    return market_ret, market_trend_60d, market_cumret


def market_forward_return(market_cumret, event_date, n_days):
    idx = market_cumret.index
    pos = idx.searchsorted(event_date)
    if pos + n_days >= len(idx):
        return np.nan
    return market_cumret.iloc[pos + n_days] / market_cumret.iloc[pos] - 1


def tercile_labels(market_trend_60d):
    valid = market_trend_60d.dropna()
    labels = pd.qcut(valid, 3, labels=["weak", "neutral", "strong"])
    return labels.reindex(market_trend_60d.index)


def find_events(close, volume, daily_ret):
    avg_vol_60d = volume.rolling(LIQUIDITY_WINDOW_DAYS).mean().shift(1)
    events = []
    dates = close.index
    for ticker in close.columns:
        ret_col = daily_ret[ticker]
        vol_col = avg_vol_60d[ticker]
        close_col = close[ticker]
        candidate_positions = np.where(
            (ret_col.values <= DROP_THRESHOLD_PCT / 100.0) &
            (vol_col.values > LIQUIDITY_MIN_AVG_VOLUME)
        )[0]
        last_kept_pos = -10 ** 9
        for pos in candidate_positions:
            if pos - last_kept_pos <= DEDUP_WINDOW_TRADING_DAYS:
                continue
            last_kept_pos = pos
            events.append({
                "ticker": ticker,
                "date": dates[pos],
                "pos": pos,
                "close": close_col.iloc[pos],
                "pct_drop": ret_col.iloc[pos] * 100,
            })
    return pd.DataFrame(events)


def attach_forward_returns(events_df, close, market_cumret, market_trend_60d, tercile):
    n_positions = len(close.index)
    for n in FORWARD_WINDOWS:
        stock_fwd = []
        mkt_fwd = []
        for _, row in events_df.iterrows():
            pos = row["pos"]
            ticker = row["ticker"]
            if pos + n >= n_positions:
                stock_fwd.append(np.nan)
                mkt_fwd.append(np.nan)
                continue
            c0 = close[ticker].iloc[pos]
            c1 = close[ticker].iloc[pos + n]
            stock_fwd.append(c1 / c0 - 1)
            mkt_fwd.append(market_cumret.iloc[pos + n] / market_cumret.iloc[pos] - 1)
        events_df[f"stock_fwd_{n}d"] = stock_fwd
        events_df[f"market_fwd_{n}d"] = mkt_fwd
        events_df[f"excess_fwd_{n}d"] = events_df[f"stock_fwd_{n}d"] - events_df[f"market_fwd_{n}d"]

    events_df["market_trend_60d"] = events_df["date"].map(market_trend_60d)
    events_df["tercile"] = events_df["date"].map(tercile)
    events_df["year"] = events_df["date"].dt.year
    return events_df


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
            row[f"t_stat_{n}d"] = t_stat
            row[f"p_value_{n}d"] = p_val
            row[f"pct_positive_{n}d"] = (vals > 0).mean() * 100
        else:
            for k in ["mean_excess", "median_excess", "t_stat", "p_value", "pct_positive"]:
                row[f"{k}_{n}d_pct" if k in ("mean_excess", "median_excess", "pct_positive") else f"{k}_{n}d"] = np.nan
    return row


def main():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print("Loading price panel from DB...")
    close, volume = load_panel()
    print(f"Panel shape: {close.shape[0]} trading days x {close.shape[1]} tickers "
          f"({close.index.min().date()} to {close.index.max().date()})")

    daily_ret = close.pct_change()
    sanity_check_extreme_returns(daily_ret)

    print("\nComputing equal-weight market proxy and stage 0 tercile boundaries...")
    market_ret, market_trend_60d, market_cumret = compute_market_proxy(daily_ret)
    tercile = tercile_labels(market_trend_60d)
    valid_trend = market_trend_60d.dropna()
    print("market_trend_60d tercile boundaries (cumulative 60-trading-day market return):")
    print(valid_trend.quantile([0, 1/3, 2/3, 1]))

    print("\nScanning for stage 1 drop events (>=7% single-day, liquidity-filtered, deduped)...")
    events = find_events(close, volume, daily_ret)
    print(f"Total stage-1 events found: {len(events)}")

    events = attach_forward_returns(events, close, market_cumret, market_trend_60d, tercile)

    events.to_csv("data/stage0_1_events.csv", index=False)
    print("Saved full event list to data/stage0_1_events.csv")

    # ---- Q1: total count ----
    print("\n" + "=" * 70)
    print("Q1: TOTAL STAGE-1 EVENTS")
    print("=" * 70)
    print(f"Total events (all years, all terciles): {len(events)}")
    for n in FORWARD_WINDOWS:
        usable = events[f"excess_fwd_{n}d"].notna().sum()
        print(f"  usable for {n}d forward-return stats (full window available): {usable}")

    # ---- Q2: by year ----
    print("\n" + "=" * 70)
    print("Q2: BY YEAR - excess return (stock minus equal-weight market proxy)")
    print("=" * 70)
    year_rows = [summarize(g, str(y)) for y, g in events.groupby("year")]
    year_df = pd.DataFrame(year_rows).set_index("group")
    cols_20 = ["n_total", "n_with_20d_data", "mean_excess_20d_pct", "median_excess_20d_pct",
               "pct_positive_20d", "t_stat_20d", "p_value_20d"]
    cols_60 = ["n_total", "n_with_60d_data", "mean_excess_60d_pct", "median_excess_60d_pct",
               "pct_positive_60d", "t_stat_60d", "p_value_60d"]
    print("\n-- 20 trading days forward --")
    print(year_df[cols_20].round(4))
    print("\n-- 60 trading days forward --")
    print(year_df[cols_60].round(4))

    # ---- Q4: stage 0 tercile split ----
    print("\n" + "=" * 70)
    print("Q4: STAGE 0 - split by market_trend_60d tercile on the event day")
    print("=" * 70)
    tercile_rows = [summarize(g, t) for t, g in events.groupby("tercile", observed=True)]
    tercile_df = pd.DataFrame(tercile_rows).set_index("group").reindex(["weak", "neutral", "strong"])
    print("\n-- 20 trading days forward --")
    print(tercile_df[cols_20].round(4))
    print("\n-- 60 trading days forward --")
    print(tercile_df[cols_60].round(4))

    year_df.to_csv("data/stage0_1_by_year.csv")
    tercile_df.to_csv("data/stage0_1_by_tercile.csv")
    print("\nSaved data/stage0_1_by_year.csv and data/stage0_1_by_tercile.csv")


if __name__ == "__main__":
    main()
