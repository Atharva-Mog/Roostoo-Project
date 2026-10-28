"""Run 12: Binance data. Perp execution costs (run 9). Two feature sets per config:
 base = the 23 OHLCV features (as runs 1-11);  full = base + 16 order-flow/positioning/cross-asset features.
Reports trading results plus ranking/direction diagnostics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pipeline as P, experiment as E, perp_costs as C
from sklearn.metrics import roc_auc_score
b15 = P.load_15m(); rows = []; os.makedirs("results/v12_binance", exist_ok=True)
for bar, hor in P.GRID:
    df = C.add_execution(P.build(b15, bar, hor), b15); allc = [c for c in P.feat_cols(df) if c != "filled"]
    for fs, cols in (("base", [c for c in allc if not c.startswith("x_")]), ("full", allc)):
        tl, evs, nets, labs, pps, pos = [], [], [], [], [], 0
        for s, e in P.folds(df):
            tr, te = P.split(df, s, e, embargo_h=P.HOR_H[hor])
            pl = E.fit_predict_multi("lgb", tr, te, cols); ev = C.ev_perp(pl, tr, te, P.HOR_H[hor])
            t = P.sequential(te, (ev > 0) & te.filled.values); tl.append(t); pos += t.sum() > 0
            evs.append(ev); nets.append(te.net.values); labs.append(te.label.values); pps.append(pl)
        t = pd.concat(tl); ev = np.concatenate(evs); net = np.concatenate(nets); lab = np.concatenate(labs); pp = np.vstack(pps)
        res = lab != 0
        pm = P.perf(t, pd.Timestamp("2022-07-01", tz="UTC"), P.HOLDOUT_START)
        r = dict(cfg=f"{bar}_{hor}", feats=fs, n=pm["n"], ev_bps=pm["ev_bps"], total=pm["total"], sharpe=pm["sharpe"], maxdd=pm["maxdd"],
                 pos_folds=f"{pos}/{len(tl)}", spearman=pd.Series(ev).corr(pd.Series(net), method="spearman"),
                 auc_dir=roc_auc_score(lab[res] == 1, pp[res, 2] / (pp[res, 2] + pp[res, 0])))
        rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
pd.DataFrame(rows).to_csv("results/v12_binance/grid_summary.csv", index=False)
