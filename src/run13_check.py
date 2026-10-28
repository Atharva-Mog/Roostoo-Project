"""Run 13: robustness of run-12 best (4h bars / 24h, full features, perp costs, EV>0).
Selected from 16 run-12 variants -> multiple-testing caveat applies."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, lightgbm as lgb, pipeline as P, experiment as E, perp_costs as C
bar, hor = sys.argv[1], sys.argv[2]; NPERM = int(sys.argv[3])
b15 = P.load_15m(); df = C.add_execution(P.build(b15, bar, hor), b15); cols = [c for c in P.feat_cols(df) if c != "filled"]
S, EH = pd.Timestamp("2022-07-01", tz="UTC"), P.HOLDOUT_START
def walk(shuffle=None, keep=False):
    tl, rl, evl, netl, imps = [], [[] for _ in range(50)], [], [], []
    for s, e in P.folds(df):
        tr, te = P.split(df, s, e, embargo_h=24)
        if shuffle is not None:
            idx = np.random.default_rng(shuffle).permutation(len(tr)); tr = tr.copy()
            for c in ("label", "gross"): tr[c] = tr[c].values[idx]
        pl = E.fit_predict_multi("lgb", tr, te, cols); ev = C.ev_perp(pl, tr, te, 24)
        f = te.filled.values; take = (ev > 0) & f; tl.append(P.sequential(te, take)); evl.append(ev); netl.append(te.net.values)
        if keep:
            rate = take.sum() / max(f.sum(), 1); rng = np.random.default_rng(7)
            for i in range(50): rl[i].append(P.sequential(te, f & (rng.random(len(te)) < rate)))
            a = tr.iloc[: int(len(tr) * .8)]
            m = lgb.LGBMClassifier(**dict(E.LGB_PARAMS, objective="multiclass", importance_type="gain")).fit(a[cols], a.label + 1)
            imps.append(pd.Series(m.feature_importances_, cols) / m.feature_importances_.sum())
    t = pd.concat(tl); sp = pd.Series(np.concatenate(evl)).corr(pd.Series(np.concatenate(netl)), method="spearman")
    return t, sp, rl, imps
t, sp, rl, imps = walk(keep=True)
pm = P.perf(t, S, EH)
rp = [P.perf(pd.concat(x), S, EH) for x in rl]
p_rand = (sum(x["total"] >= pm["total"] for x in rp) + 1) / 51
blocks = [g.values for _, g in t.groupby(t.index.to_period("M"))]; rng = np.random.default_rng(0)
bs = [np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]).mean() * 1e4 for _ in range(5000)]
null = [walk(shuffle=i) for i in range(NPERM)]
nsp = np.array([x[1] for x in null]); ntot = np.array([P.perf(x[0], S, EH)["total"] for x in null])
imp = pd.concat(imps, axis=1).mean(1).sort_values(ascending=False)
yr = t.groupby(t.index.year).agg(["count", "sum", "mean"]); yr["mean"] *= 1e4
print(f"MODEL n={pm['n']} ev={pm['ev_bps']:.2f}bps total={pm['total']:.3f} sharpe={pm['sharpe']:.2f} maxdd={pm['maxdd']:.3f} spearman={sp:.3f}")
print(f"random-filter (filled, same rate): mean total={np.mean([x['total'] for x in rp]):.3f}  p={p_rand:.3f}")
print(f"bootstrap EV 95% CI [{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}] bps  P(EV>0)={np.mean(np.array(bs)>0):.3f}")
print(f"shuffled null ({NPERM}): spearman mean={nsp.mean():.3f} p={(np.sum(nsp>=sp)+1)/(NPERM+1):.3f} | total mean={ntot.mean():.3f} p={(np.sum(ntot>=pm['total'])+1)/(NPERM+1):.3f}")
print("by year (n, sum log, mean bps):\n", yr.round(3).to_string())
print("top gain importance:\n", imp.head(12).round(3).to_string())
pd.DataFrame({"bootstrap_ev_bps": bs}).describe().to_csv(f"results/v13_bootstrap_{bar}_{hor}.csv"); imp.to_csv(f"results/v13_importance_{bar}_{hor}.csv")
t.to_csv(f"results/v13_trades_{bar}_{hor}.csv")
