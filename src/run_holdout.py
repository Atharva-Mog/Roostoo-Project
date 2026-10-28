"""ONE-SHOT holdout evaluation of the dev-selected config (1h bars, 24h horizon, v3 three-class LGBM, EV>0 gate).
Train on all dev data (purged), test on 2026-04-27 -> end. Also cost sensitivity on dev+holdout."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pipeline as P, experiment as E
bar, hor = "1h", "24h"
b15 = P.load_15m(); df = P.build(b15, bar, hor); cols = P.feat_cols(df)
s = P.HOLDOUT_START; e = df.index[-1] + pd.Timedelta(hours=1)
tr, te = P.split(df, s, e, embargo_h=24)
pl = E.fit_predict_multi("lgb", tr, te, cols); ev = E.ev_multi(pl, tr, te)
plr = E.fit_predict_multi("lr", tr, te, cols); evr = E.ev_multi(plr, tr, te)
res = {}
res["model_lgb"] = P.perf(P.sequential(te, ev > 0), s, e)
res["model_lr"] = P.perf(P.sequential(te, evr > 0), s, e)
res["take_every"] = P.perf(P.sequential(te, np.ones(len(te), bool)), s, e)
rng = np.random.default_rng(0); rate = (ev > 0).mean()
rr = [P.perf(P.sequential(te, rng.random(len(te)) < rate), s, e) for _ in range(50)]
res["random_same_rate"] = {k: np.nanmean([x[k] for x in rr]) for k in rr[0]}
bh = b15.close[(b15.index >= s)]; res["buy_hold"] = {"total": np.log(bh.iloc[-1] / bh.iloc[0])}
out = pd.DataFrame(res).T; print(out.round(4).to_string()); print("spearman EV vs net", round(pd.Series(ev).corr(pd.Series(te.net.values), method="spearman"), 3), "| test", te.index[0], "->", te.index[-1], len(te))
out.to_csv("results/holdout_1h_24h.csv")
# cost sensitivity on dev walk-forward preds (v3) : gross of taken trades re-costed
pr = pd.read_parquet("results/v3_multi/preds_1h_24h.parquet"); gross = pr.net + P.COST
rows = []
for rt_bps in (25, 20, 15, 10, 5, 0):
    p2 = pr.copy(); p2["net"] = gross - rt_bps / 1e4
    # gate re-evaluated with the new cost: ev_lgb was computed with 25bps, shift accordingly
    take = (pr.ev_lgb + (P.COST - rt_bps / 1e4)) > 0
    t = P.sequential(p2, take.values)
    pm = P.perf(t, pd.Timestamp("2022-07-01", tz="UTC"), P.HOLDOUT_START)
    rows.append(dict(round_trip_bps=rt_bps, **{k: round(v, 3) for k, v in pm.items()}))
cs = pd.DataFrame(rows); print(cs.to_string(index=False)); cs.to_csv("results/cost_sensitivity_1h_24h.csv", index=False)
