"""Shuffled-label null: permute (label, gross, sigma, exit) jointly across TRAINING rows in each fold, retrain,
score test. Compares spearman(EV, realized net) and pooled net EV of taken trades to the real model."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pipeline as P, experiment as E
bar, hor, nperm = sys.argv[1], sys.argv[2], int(sys.argv[3])
b15 = P.load_15m(); df = P.build(b15, bar, hor); cols = P.feat_cols(df)
def score(shuffle_seed=None):
    evs, nets, trades = [], [], []
    for s, e in P.folds(df):
        tr, te = P.split(df, s, e, embargo_h=P.HOR_H[hor])
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed); idx = rng.permutation(len(tr))
            tr = tr.copy()
            for c in ("label", "gross"): tr[c] = tr[c].values[idx]
        pl = E.fit_predict_multi("lgb", tr, te, cols); ev = E.ev_multi(pl, tr, te)
        evs.append(pd.Series(ev, index=te.index)); nets.append(te.net)
        trades.append(P.sequential(te, ev > 0))
    ev, net, t = pd.concat(evs), pd.concat(nets), pd.concat(trades)
    return ev.corr(net, method="spearman"), (t.mean() * 1e4 if len(t) else np.nan), len(t)
real = score(); print("REAL", real, flush=True)
null = [score(i) for i in range(nperm)]
nd = pd.DataFrame(null, columns=["spearman", "ev_bps", "n"]); nd.to_csv(f"results/null_{bar}_{hor}.csv", index=False)
print(nd.describe().round(3).to_string())
print("p(spearman)", ((nd.spearman >= real[0]).sum() + 1) / (len(nd) + 1), "p(ev)", ((nd.ev_bps >= real[1]).sum() + 1) / (len(nd) + 1))
