"""Run 17: 'buy when the day starts lower' ideas, tested with fees (spot 12.5 bps/side), Binance 2022-07 -> 2026-10.
 A) Intraday profile: average return by UTC hour (is there a reliable low hour?).
 B) Session-open dip: if the first H hours of the UTC day fell by more than k x typical, buy at hour H, sell at day end.
 C) Auction-market version: if price at 00:00 UTC is below yesterday's VWAP (fair value) by more than k x daily vol,
    buy and hold until it touches yesterday's VWAP or the day ends.
All variants reported; nothing selected."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); os.environ["RT_DATA"] = "binance"
import pipeline as P
b = P.load_15m(); b = b[b.index >= "2022-07-01"]; C = 2 * 0.00125
h = b.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
r = np.log(h.close).diff()
prof = r.groupby(r.index.hour).agg(["mean", "std", "count"]); prof["t"] = prof["mean"] / (prof["std"] / np.sqrt(prof["count"]))
prof["mean_bps"] = prof["mean"] * 1e4
print("A) average 1h return by UTC hour (bps), t-stat:"); print(prof[["mean_bps", "t"]].round(2).T.to_string())
days = h.groupby(h.index.normalize())
dvol = np.log(h.close).diff().groupby(h.index.normalize()).std() * np.sqrt(24)
rows = []
for H in (2, 4, 8):
    for k in (0.0, 0.5, 1.0):
        tr = []
        for d, g in days:
            if len(g) < 24 or d not in dvol.index: continue
            v = dvol.shift(1).get(d)
            if not np.isfinite(v): continue
            first = np.log(g.close.iloc[H - 1] / g.open.iloc[0])
            if first < -k * v * np.sqrt(H / 24):
                tr.append(np.log(g.close.iloc[-1] / g.close.iloc[H - 1]) - C)
        tr = np.array(tr)
        rows.append(dict(rule="B open-dip", H=H, k=k, trades=len(tr), net_bps=tr.mean() * 1e4 if len(tr) else np.nan,
                         hit=(tr > 0).mean() if len(tr) else np.nan, total_pct=tr.sum() * 100))
# C: VWAP fair value
vw = (h.close * h.volume).groupby(h.index.normalize()).sum() / h.volume.groupby(h.index.normalize()).sum()
for k in (0.0, 0.25, 0.5):
    tr = []
    for d, g in days:
        prev = d - pd.Timedelta(days=1)
        if prev not in vw.index or len(g) < 24: continue
        v = dvol.shift(1).get(d); fv = vw[prev]; px = g.open.iloc[0]
        if not np.isfinite(v): continue
        if np.log(px / fv) < -k * v:
            hit = g[g.high >= fv]
            exitp = fv if len(hit) else g.close.iloc[-1]
            tr.append(np.log(exitp / px) - C)
    tr = np.array(tr)
    rows.append(dict(rule="C below yday VWAP", H=0, k=k, trades=len(tr), net_bps=tr.mean() * 1e4, hit=(tr > 0).mean(), total_pct=tr.sum() * 100))
res = pd.DataFrame(rows).round(2)
dr = np.log(h.close.resample("1D").last()).diff().dropna()
print("\nbaseline: buy every day at 00:00, sell at 24:00 -> net bps/day", round((dr.mean() - C) * 1e4, 1), "| gross", round(dr.mean() * 1e4, 1))
print(res.to_string(index=False))
os.makedirs("results/v17_open_dip", exist_ok=True); res.to_csv("results/v17_open_dip/rules.csv", index=False); prof.to_csv("results/v17_open_dip/hour_profile.csv")
