"""Closes the open item from stock_screener_agent.md: is the 2023-2026
"external-reaction drops recover better" pattern a genuine overreaction-
correction effect, or a mechanical artifact of high-beta stocks
outperforming the equal-weight index during a mostly-up market?

The stage 0/1/2 "excess return" so far is stock_fwd - market_fwd, which
implicitly assumes every stock has beta=1 to the equal-weight proxy.
That's fine for market-regime-neutral comparisons, but if (a) stocks that
crash for external/market reasons are systematically higher-beta than
stocks that crash on their own earnings, and (b) 2023-2026 was mostly an
up market for the proxy, then part or all of the "external recovers
better" effect could just be beta exposure, not a real overreaction
correction.

Test: replace excess_fwd (stock - market) with a CAPM-style abnormal
return (stock - beta*market), where beta is estimated from the 252
trading days strictly before the event (no lookahead, and the crash day
itself excluded so it doesn't distort the beta estimate). Re-run the
category comparison and the pre/post-2023 robustness check on the
beta-adjusted numbers and see whether the finding survives.
"""
import numpy as np
import pandas as pd
from scipy import stats

from analysis.stage0_1 import load_panel, compute_market_proxy

MIN_BETA_WINDOW = 120   # minimum trailing trading days required to trust a beta estimate
BETA_WINDOW = 252        # ~1 trading year
FORWARD_WINDOWS = (20, 60)
N_BOOTSTRAP = 5000
RNG_SEED = 20260816


def compute_trailing_betas(events, daily_ret, market_ret):
    """For each event (needs 'ticker' and 'pos' columns), estimate beta
    from the BETA_WINDOW trading days strictly before the event (pos-1
    inclusive back to pos-BETA_WINDOW), excluding the crash day itself."""
    market_vals_full = market_ret.values
    betas = np.full(len(events), np.nan)
    n_obs = np.full(len(events), 0)

    for ticker, group in events.groupby("ticker"):
        stock_vals_full = daily_ret[ticker].values
        for idx, row in group.iterrows():
            pos = row["pos"]
            start = max(0, pos - BETA_WINDOW)
            end = pos  # exclusive of pos itself (the crash day)
            s = stock_vals_full[start:end]
            m = market_vals_full[start:end]
            mask = ~(np.isnan(s) | np.isnan(m))
            n = mask.sum()
            n_obs[events.index.get_loc(idx)] = n
            if n < MIN_BETA_WINDOW:
                continue
            s, m = s[mask], m[mask]
            var_m = np.var(m, ddof=1)
            if var_m == 0:
                continue
            beta = np.cov(s, m, ddof=1)[0, 1] / var_m
            betas[events.index.get_loc(idx)] = beta

    return betas, n_obs


def summarize(vals, label):
    row = {"group": label, "n": len(vals)}
    if len(vals) >= 2:
        t_stat, p_val = stats.ttest_1samp(vals, 0.0)
        row["mean_pct"] = vals.mean() * 100
        row["median_pct"] = vals.median() * 100
        row["pct_positive"] = (vals > 0).mean() * 100
        row["p_value"] = p_val
    return row


def bootstrap_mean_ci(vals, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    v = vals.to_numpy()
    boot_means = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return boot_means.mean(), lo, hi, (boot_means > 0).mean() * 100


def main():
    pd.set_option("display.width", 160)

    print("Loading price panel and market proxy...")
    close, volume = load_panel()
    daily_ret = close.pct_change()
    market_ret, market_trend_60d, market_cumret = compute_market_proxy(daily_ret)

    events = pd.read_csv("data/stage2_sec_events.csv")
    events["date"] = pd.to_datetime(events["date"])

    print(f"Estimating trailing {BETA_WINDOW}d beta for {len(events)} events "
          f"(min {MIN_BETA_WINDOW}d of history required)...")
    betas, n_obs = compute_trailing_betas(events, daily_ret, market_ret)
    events["beta"] = betas
    events["beta_n_obs"] = n_obs
    n_missing = events["beta"].isna().sum()
    print(f"Events with a usable beta: {len(events) - n_missing}/{len(events)} "
          f"({n_missing} excluded - insufficient trailing history, mostly early 2019)")

    for n in FORWARD_WINDOWS:
        events[f"abnormal_fwd_{n}d"] = events[f"stock_fwd_{n}d"] - events["beta"] * events[f"market_fwd_{n}d"]

    print("\n" + "=" * 70)
    print("0. Context: was 2023-2026 actually a mostly-up market for the proxy?")
    print("=" * 70)
    for label, start, end in [("2019-2022", "2019-01-01", "2022-12-31"), ("2023-2026", "2023-01-01", "2026-12-31")]:
        sub = market_ret.loc[start:end]
        cum = (1 + sub).prod() - 1
        ann = (1 + sub.mean()) ** 252 - 1
        print(f"  {label}: cumulative equal-weight proxy return = {cum*100:.1f}%  "
              f"(annualized daily mean = {ann*100:.1f}%)")

    print("\n" + "=" * 70)
    print("1. Beta comparison: external_market_reaction vs earnings-related categories")
    print("=" * 70)
    ext = events[events["drop_reason"] == "external_market_reaction"]
    earn = events[events["drop_reason"] != "external_market_reaction"]
    ext_beta = ext["beta"].dropna()
    earn_beta = earn["beta"].dropna()
    t_stat, p_val = stats.ttest_ind(ext_beta, earn_beta, equal_var=False)
    print(f"  external:  n={len(ext_beta)}  mean beta={ext_beta.mean():.3f}  median={ext_beta.median():.3f}")
    print(f"  earnings-related: n={len(earn_beta)}  mean beta={earn_beta.mean():.3f}  median={earn_beta.median():.3f}")
    print(f"  Welch t={t_stat:.3f}  p={p_val:.5f}")

    print("\n  Same comparison, split pre/post 2023 (did the external category's beta composition shift?):")
    ext_pre = ext[ext["date"] < "2023-01-01"]
    ext_post = ext[ext["date"] >= "2023-01-01"]
    for label, sub in [("pre-2023", ext_pre), ("2023-2026", ext_post)]:
        b = sub["beta"].dropna()
        print(f"    external, {label}: n={len(b)}  mean beta={b.mean():.3f}")

    print("\n" + "=" * 70)
    print("2. Does the pre-2023 vs 2023-2026 gap in external_market_reaction SURVIVE beta-adjustment?")
    print("=" * 70)
    for n in FORWARD_WINDOWS:
        pre = ext_pre[f"abnormal_fwd_{n}d"].dropna()
        post = ext_post[f"abnormal_fwd_{n}d"].dropna()
        pre_raw = ext_pre[f"excess_fwd_{n}d"].dropna()
        post_raw = ext_post[f"excess_fwd_{n}d"].dropna()
        t_beta, p_beta = stats.ttest_ind(post, pre, equal_var=False)
        t_raw, p_raw = stats.ttest_ind(post_raw, pre_raw, equal_var=False)
        print(f"\n  {n}d forward:")
        print(f"    RAW excess (beta=1 assumed):  pre={pre_raw.mean()*100:.3f}%  post={post_raw.mean()*100:.3f}%  "
              f"diff p={p_raw:.6f}")
        print(f"    BETA-ADJUSTED abnormal return: pre={pre.mean()*100:.3f}%  post={post.mean()*100:.3f}%  "
              f"diff p={p_beta:.6f}")

    print("\n" + "=" * 70)
    print("3. Beta-adjusted robustness check on 2023-2026 external events (mirrors the raw-excess version)")
    print("=" * 70)
    post_ext = ext[ext["date"] >= "2023-01-01"]
    median_date = post_ext["date"].median()
    early = post_ext[post_ext["date"] < median_date]
    late = post_ext[post_ext["date"] >= median_date]
    for n in FORWARD_WINDOWS:
        for label, sub in [("EARLY half", early), ("LATE half", late)]:
            row = summarize(sub[f"abnormal_fwd_{n}d"].dropna(), f"{label} {n}d")
            if row["n"] >= 2:
                print(f"  {label} ({n}d, beta-adj): n={row['n']}  mean={row['mean_pct']:.3f}%  p={row['p_value']:.5f}")
        vals = post_ext[f"abnormal_fwd_{n}d"].dropna()
        boot_mean, lo, hi, pct_pos = bootstrap_mean_ci(vals)
        print(f"  Bootstrap ({n}d, beta-adj, n={len(vals)}): mean={boot_mean*100:.3f}%  "
              f"95% CI=[{lo*100:.3f}%, {hi*100:.3f}%]  %resamples>0={pct_pos:.1f}%")

    events.to_csv("data/stage2_beta_adjusted_events.csv", index=False)
    print("\nSaved data/stage2_beta_adjusted_events.csv")


if __name__ == "__main__":
    main()
