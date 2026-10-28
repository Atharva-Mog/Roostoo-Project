import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pipeline as P, experiment as E, rfe as R
bar, hor, tag = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(f"results/{tag}", exist_ok=True)
b15 = P.load_15m(); df = P.build(b15, bar, hor); allc = P.feat_cols(df)
surv = {c: 0 for c in allc}; rows = []; preds = []
for s, e in P.folds(df):
    tr, te = P.split(df, s, e, embargo_h=P.HOR_H[hor])
    sel, _ = R.rfecv(tr, allc)
    for c in sel: surv[c] += 1
    pl = E.fit_predict_multi("lgb", tr, te, sel); ev = E.ev_multi(pl, tr, te)
    r = dict(fold=str(s.date()), n_feats=len(sel), feats=";".join(sel))
    for k, v in P.perf(P.sequential(te, ev > 0), s, e).items(): r[f"lgb_{k}"] = v
    rows.append(r); preds.append(pd.DataFrame({"ev_lgb": ev, "net": te.net, "label": te.label, "exit_time": te.exit_time}, index=te.index))
    print(r["fold"], len(sel), {k: round(r[k], 3) for k in ("lgb_n", "lgb_ev_bps", "lgb_sharpe")}, flush=True)
res = pd.DataFrame(rows); res.to_csv(f"results/{tag}/folds.csv", index=False)
pr = pd.concat(preds); pr.to_parquet(f"results/{tag}/preds.parquet")
st = pd.Series(surv).sort_values(ascending=False) / len(res); st.to_csv(f"results/{tag}/feature_survival.csv", header=["survival_rate"])
print(st.round(2).to_string())
print("TOTAL n", res.lgb_n.sum(), "ev_bps(mean folds)", res.lgb_ev_bps.mean().round(1), "sharpe", res.lgb_sharpe.mean().round(2), "pos folds", (res.lgb_total > 0).sum(), "/", len(res), "spearman", round(pr.ev_lgb.corr(pr.net, method="spearman"), 3))
