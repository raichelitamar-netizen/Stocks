"""Robustness check on the apparent 2023+ regime shift in the
external_market_reaction category (see stage2_sec.py's by-year output).

Mirrors the validation approach used for stage 0 (internal consistency /
half-split / bootstrap): before treating "external-reaction drops recover
better since 2023" as a real, stable pattern rather than one unusually
good stretch (the way 2016 was an outlier within 2013-2018, not a trend),
this:
  1. Splits the 2023-2026 external-reaction events into an early half and
     late half (by date) and checks both halves independently.
  2. Bootstraps the 2023-2026 mean excess return (5,000 resamples) to see
     how stable the estimate is, not just its point value.
  3. Runs a formal two-sample test of pre-2023 vs 2023+ excess returns -
     is the difference between the periods itself statistically
     distinguishable, or within noise given the sample sizes involved.
"""
import numpy as np
import pandas as pd
from scipy import stats

FORWARD_WINDOWS = (20, 60)
N_BOOTSTRAP = 5000
RNG_SEED = 20260816  # fixed for reproducibility


def summarize(sub, label):
    row = {"group": label, "n_total": len(sub)}
    for n in FORWARD_WINDOWS:
        vals = sub[f"excess_fwd_{n}d"].dropna()
        row[f"n_with_{n}d"] = len(vals)
        if len(vals) >= 2:
            t_stat, p_val = stats.ttest_1samp(vals, 0.0)
            row[f"mean_{n}d_pct"] = vals.mean() * 100
            row[f"median_{n}d_pct"] = vals.median() * 100
            row[f"pct_positive_{n}d"] = (vals > 0).mean() * 100
            row[f"p_value_{n}d"] = p_val
    return row


def bootstrap_mean_ci(vals, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    vals = vals.to_numpy()
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    pct_positive = (boot_means > 0).mean() * 100
    return boot_means.mean(), lo, hi, pct_positive


def main():
    pd.set_option("display.width", 160)
    df = pd.read_csv("data/stage2_sec_events.csv")
    df["date"] = pd.to_datetime(df["date"])
    ext = df[df["drop_reason"] == "external_market_reaction"].copy()

    pre = ext[ext["date"] < "2023-01-01"]
    post = ext[ext["date"] >= "2023-01-01"]
    print(f"Pre-2023 external events: {len(pre)}   Post-2023 (2023-2026) external events: {len(post)}")

    print("\n" + "=" * 70)
    print("1. SPLIT-HALF within 2023-2026 (chronological, by median date)")
    print("=" * 70)
    median_date = post["date"].median()
    print(f"Split point (median date): {median_date.date()}")
    early = post[post["date"] < median_date]
    late = post[post["date"] >= median_date]
    for label, sub in [("2023-2026 EARLY half", early), ("2023-2026 LATE half", late)]:
        row = summarize(sub, label)
        print(f"\n{label}: n={row['n_total']}")
        for n in FORWARD_WINDOWS:
            print(f"  {n}d: mean={row.get(f'mean_{n}d_pct', float('nan')):.3f}%  "
                  f"median={row.get(f'median_{n}d_pct', float('nan')):.3f}%  "
                  f"%pos={row.get(f'pct_positive_{n}d', float('nan')):.1f}%  "
                  f"p={row.get(f'p_value_{n}d', float('nan')):.4f}  n_valid={row.get(f'n_with_{n}d')}")

    print("\n" + "=" * 70)
    print("2. BOOTSTRAP (5,000 resamples) of the 2023-2026 mean excess return")
    print("=" * 70)
    for n in FORWARD_WINDOWS:
        vals = post[f"excess_fwd_{n}d"].dropna()
        boot_mean, lo, hi, pct_pos = bootstrap_mean_ci(vals)
        print(f"  {n}d (n={len(vals)}): point mean={vals.mean()*100:.3f}%  "
              f"bootstrap mean={boot_mean*100:.3f}%  95% CI=[{lo*100:.3f}%, {hi*100:.3f}%]  "
              f"% of resamples with mean>0: {pct_pos:.1f}%")

    print("\n" + "=" * 70)
    print("3. Pre-2023 vs 2023-2026: is the DIFFERENCE between periods itself significant?")
    print("=" * 70)
    for n in FORWARD_WINDOWS:
        pre_vals = pre[f"excess_fwd_{n}d"].dropna()
        post_vals = post[f"excess_fwd_{n}d"].dropna()
        t_stat, p_val = stats.ttest_ind(post_vals, pre_vals, equal_var=False)
        print(f"  {n}d: pre-2023 mean={pre_vals.mean()*100:.3f}% (n={len(pre_vals)})  "
              f"vs 2023-2026 mean={post_vals.mean()*100:.3f}% (n={len(post_vals)})  "
              f"Welch t={t_stat:.3f}  p={p_val:.5f}")

    print("\n" + "=" * 70)
    print("4. Quarterly breakdown within 2023-2026 (finer granularity than yearly)")
    print("=" * 70)
    post["quarter"] = post["date"].dt.to_period("Q").astype(str)
    rows = [summarize(g, q) for q, g in post.groupby("quarter")]
    qdf = pd.DataFrame(rows).set_index("group")
    print(qdf[["n_total", "mean_20d_pct", "p_value_20d", "mean_60d_pct", "p_value_60d"]].round(4))


if __name__ == "__main__":
    main()
