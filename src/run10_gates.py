"""Run 10: gates on top of run-9 perp costs, 24h configs only. Variants (all logged):
 margin m bps on EV; regime gate = rv_168h above its TRAINING-fold median."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pipeline as P, experiment as E, perp_costs as C
b15 = P.load_15m(); rows = []
for bar in ("15m", "1h", "4h"):
    hor = "24h"; df = C.add_execution(P.build(b15, bar, hor), b15); cols = [c for c in P.feat_cols(df) if c != "filled"]
    cache = []
    for s, e in P.folds(df):
        tr, te = P.split(df, s, e, embargo_h=24)
        pl = E.fit_predict_multi("lgb", tr, te, cols); ev = C.ev_perp(pl, tr, te, 24)
        cache.append((s, e, tr, te, ev))
    for regime in (False, True):
        for m in (0, 5, 10, 20):
            tot, tl, pos = 0, [], 0
            for s, e, tr, te, ev in cache:
                mask = (ev > m / 1e4) & te.filled.values
                if regime: mask &= (te.rv_168h > tr.rv_168h.median()).values
                t = P.sequential(te, mask); tl.append(t); pos += t.sum() > 0
            t = pd.concat(tl)
            pm = P.perf(t, pd.Timestamp("2022-07-01", tz="UTC"), P.HOLDOUT_START)
            rows.append(dict(cfg=f"{bar}_24h", regime=regime, margin_bps=m, n=pm["n"], ev_bps=round(pm["ev_bps"], 2),
                             total=round(pm["total"], 3), sharpe=round(pm["sharpe"], 2), maxdd=round(pm["maxdd"], 3), pos_folds=f"{pos}/8"))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/v10_gates.csv", index=False)
