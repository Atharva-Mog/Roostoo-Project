"""Run 11: does the model add value inside the high-vol regime? + bootstrap CI + holdout (NOT clean: holdout seen in run 8)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pipeline as P, experiment as E, perp_costs as C
b15 = P.load_15m(); bar, hor = "1h", "24h"
df = C.add_execution(P.build(b15, bar, hor), b15); cols = [c for c in P.feat_cols(df) if c != "filled"]
def evaluate(s, e, tag):
    tr, te = P.split(df, s, e, embargo_h=24)
    pl = E.fit_predict_multi("lgb", tr, te, cols); ev = C.ev_perp(pl, tr, te, 24)
    reg = (te.rv_168h > tr.rv_168h.median()).values; f = te.filled.values
    out = {"model+regime": P.sequential(te, (ev > 0) & f & reg), "regime_only(every)": P.sequential(te, f & reg),
           "model_only": P.sequential(te, (ev > 0) & f)}
    rng = np.random.default_rng(1); rate = ((ev > 0) & f & reg).sum() / max((f & reg).sum(), 1)
    out["random_in_regime"] = [P.sequential(te, f & reg & (rng.random(len(te)) < rate)) for _ in range(50)]
    return out
dev = {k: [] for k in ("model+regime", "regime_only(every)", "model_only")}; rnd = [[] for _ in range(50)]
for s, e in P.folds(df):
    o = evaluate(s, e, "dev")
    for k in dev: dev[k].append(o[k])
    for i, t in enumerate(o["random_in_regime"]): rnd[i].append(t)
S, Eh = pd.Timestamp("2022-07-01", tz="UTC"), P.HOLDOUT_START
rows = []
for k, v in dev.items():
    t = pd.concat(v); rows.append(dict(period="dev", strategy=k, **P.perf(t, S, Eh)))
    if k == "model+regime": main = t
rp = [P.perf(pd.concat(x), S, Eh) for x in rnd]
rows.append(dict(period="dev", strategy="random_in_regime(mean of 50)", **{m: np.nanmean([x[m] for x in rp]) for m in rp[0]}))
p_rand = (np.sum([x["total"] >= rows[0]["total"] for x in rp]) + 1) / 51
# block bootstrap (monthly blocks) of EV per trade for model+regime
mo = main.groupby(main.index.to_period("M")); blocks = [g.values for _, g in mo]
rng = np.random.default_rng(0); bs = []
for _ in range(5000):
    pick = rng.integers(0, len(blocks), len(blocks)); x = np.concatenate([blocks[i] for i in pick]); bs.append(x.mean() * 1e4)
lo, hi = np.percentile(bs, [2.5, 97.5])
# holdout (contaminated)
ho = evaluate(P.HOLDOUT_START, df.index[-1] + pd.Timedelta(hours=1), "holdout")
for k in ("model+regime", "regime_only(every)", "model_only"):
    rows.append(dict(period="holdout*", strategy=k, **P.perf(ho[k], P.HOLDOUT_START, df.index[-1])))
bh = b15.close[b15.index >= P.HOLDOUT_START]; rows.append(dict(period="holdout*", strategy="buy_hold", total=np.log(bh.iloc[-1] / bh.iloc[0])))
r = pd.DataFrame(rows); print(r.round(4).to_string(index=False))
print(f"\nmodel+regime vs random-in-regime: p={p_rand:.3f} | dev EV/trade 95% block-bootstrap CI: [{lo:.1f}, {hi:.1f}] bps | P(EV>0)={np.mean(np.array(bs) > 0):.3f}")
r.to_csv("results/v11_check.csv", index=False)
