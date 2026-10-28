"""Run 15: published strategies, parameters taken AS PUBLISHED (no tuning), tested on Binance 2021-01..2026-10.
Papers' samples end <= Feb 2022, so 2022-07+ is genuinely out-of-sample for each rule.
 S1 Intraday seasonality (Padysak & Vojtko, SSRN 4081000): long BTC 22:00->24:00 UTC daily.
 S2 MAX/MIN (same authors): daily close at 10-day max OR 10-day min -> hold next day, else flat (also 20d/50d shown).
 S3 Funding carry: long spot + short perp (delta-neutral), collect funding; always-on and 'on when 7d avg funding>0'.
 S4 Time-series momentum: long next day if trailing N-day return > 0 (N = 7, 28, 90; all reported).
Costs: spot taker 0.10%/side (+2.5bps slip); perp taker 0.05%/side (+2.5bps); costs charged on every position change."""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(__file__)); os.environ["RT_DATA"] = "binance"
import pipeline as P
b = P.load_15m()
SPOT, PERP = 0.00125, 0.00075
def stats(daily, label, pos=None):
    out = []
    for name, s, e in (("2021-01..2026-04 (dev)", "2021-01-01", "2026-04-26"), ("2022-07..2026-04 (post-paper)", "2022-07-01", "2026-04-26"),
                       ("2026-04-27..now (last 6m)", "2026-04-27", "2026-10-27")):
        d = daily[(daily.index >= s) & (daily.index <= e)]
        eq = d.cumsum(); yrs = len(d) / 365
        out.append(dict(strategy=label, period=name, ann_ret=d.sum() / yrs, ann_vol=d.std() * np.sqrt(365),
                        sharpe=d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan, maxdd=(eq - eq.cummax()).min(), total=d.sum(),
                        exposure=(pos[(pos.index >= s) & (pos.index <= e)] != 0).mean() if pos is not None else 1.0))
    return out
rows = []
close = b.close; lr = np.log(close).diff()
day = close.resample("1D").last(); dlr = np.log(day).diff().dropna()
rows += stats(dlr, "Buy & hold BTC")
# S1: 22:00-24:00 UTC. bars with open hour 22,23 -> return of those 8 bars
h = b.index.hour
s1_gross = lr[(h == 22) | (h == 23)].groupby(lr[(h == 22) | (h == 23)].index.normalize()).sum()
# entry at close of 21:45 bar = log return from 21:45 close..23:45 close; includes 22:00 bar open. fine.
s1_gross = s1_gross.reindex(dlr.index, fill_value=0.0)
for cname, c in (("spot", SPOT), ("perp", PERP)):
    rows += stats(s1_gross - 2 * c, f"S1 seasonality 22-24h UTC ({cname} costs)", pd.Series(1, s1_gross.index))
rows += stats(s1_gross, "S1 seasonality (no costs, reference)", pd.Series(1, s1_gross.index))
# S2: MAX/MIN on daily closes
for n in (10, 20, 50):
    hi, lo = day.rolling(n).max(), day.rolling(n).min()
    sig = ((day >= hi) | (day <= lo)).astype(int).shift(1).reindex(dlr.index).fillna(0)
    chg = sig.diff().abs().fillna(0)
    for cname, c in (("spot", SPOT), ("perp", PERP)):
        rows += stats(sig * dlr - chg * c, f"S2 MAX/MIN {n}d ({cname})", sig)
    # legs separately (10d only)
    if n == 10:
        for leg, cond in (("at-max only", day >= hi), ("at-min only", day <= lo)):
            sg = cond.astype(int).shift(1).reindex(dlr.index).fillna(0); ch = sg.diff().abs().fillna(0)
            rows += stats(sg * dlr - ch * SPOT, f"S2 {leg} 10d (spot)", sg)
# S3: funding carry. funding per 8h, paid to shorts when positive.
f = b.funding.resample("8h").last().dropna()
fd = f.resample("1D").sum()          # daily funding earned by short perp (fraction)
fd = fd.reindex(dlr.index).fillna(0)
always = fd.copy(); always.iloc[0] -= 2 * (SPOT + PERP)        # enter once
rows += stats(always, "S3 funding carry, always on", pd.Series(1, fd.index))
on = (fd.rolling(7).mean() > 0).astype(int).shift(1).fillna(0); ch = on.diff().abs().fillna(0)
rows += stats(on * fd - ch * (SPOT + PERP), "S3 funding carry, on if 7d avg>0", on)
# S4: TSMOM
for n in (7, 28, 90):
    sig = (np.log(day / day.shift(n)) > 0).astype(int).shift(1).reindex(dlr.index).fillna(0); ch = sig.diff().abs().fillna(0)
    rows += stats(sig * dlr - ch * SPOT, f"S4 TSMOM {n}d (spot)", sig)
r = pd.DataFrame(rows)
for c in ("ann_ret", "ann_vol", "maxdd", "total", "exposure"): r[c] = (r[c] * 100).round(1)
r["sharpe"] = r.sharpe.round(2)
os.makedirs("results/v15_literature", exist_ok=True); r.to_csv("results/v15_literature/summary.csv", index=False)
pd.set_option("display.width", 250)
for p in r.period.unique(): print("\n==", p); print(r[r.period == p].drop(columns="period").to_string(index=False))
