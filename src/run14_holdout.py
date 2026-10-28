"""Run 14: one-shot Binance holdout for run-12/13 candidate (4h/24h, full features, perp costs, EV>0).
Caveat: holdout window was viewed in run 8 (Bitstamp) and run 11; the perp-cost design came after. Semi-clean."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pipeline as P, experiment as E, perp_costs as C
b15 = P.load_15m(); rows = []
for bar, hor in (("4h", "24h"), ("1h", "24h"), ("15m", "24h")):      # report neighbours too, to avoid cherry-picking
    df = C.add_execution(P.build(b15, bar, hor), b15); cols = [c for c in P.feat_cols(df) if c != "filled"]
    s, e = P.HOLDOUT_START, df.index[-1] + pd.Timedelta(hours=1)
    tr, te = P.split(df, s, e, embargo_h=24)
    pl = E.fit_predict_multi("lgb", tr, te, cols); ev = C.ev_perp(pl, tr, te, 24); f = te.filled.values
    t = P.sequential(te, (ev > 0) & f); pm = P.perf(t, s, e)
    rng = np.random.default_rng(3); rate = ((ev > 0) & f).sum() / max(f.sum(), 1)
    rp = [P.perf(P.sequential(te, f & (rng.random(len(te)) < rate)), s, e) for _ in range(100)]
    ev_every = P.perf(P.sequential(te, f), s, e)
    rows.append(dict(cfg=f"{bar}_{hor}", n=pm["n"], ev_bps=pm["ev_bps"], hit=pm["hit"], total=pm["total"], sharpe=pm["sharpe"], maxdd=pm["maxdd"],
                     random_total=np.mean([x["total"] for x in rp]), p_vs_random=(sum(x["total"] >= pm["total"] for x in rp) + 1) / 101,
                     every_filled_total=ev_every["total"], spearman=pd.Series(ev).corr(pd.Series(te.net.values), method="spearman")))
bh = b15.close[b15.index >= P.HOLDOUT_START]
r = pd.DataFrame(rows); print(r.round(4).to_string(index=False)); print("buy&hold log return:", round(np.log(bh.iloc[-1] / bh.iloc[0]), 4), "| window", P.HOLDOUT_START.date(), "->", b15.index[-1].date())
r.to_csv("results/v14_holdout_binance.csv", index=False)
