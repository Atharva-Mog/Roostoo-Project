"""Fold-internal feature selection: correlation clustering + permutation-importance RFE with a purged
time-ordered inner validation split. Never sees the test fold."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os
from sklearn.metrics import log_loss
sys.path.insert(0, os.path.dirname(__file__))
import pipeline as P, experiment as E


def corr_prune(X, thr=0.8):
    c = X.corr(method="spearman").abs(); keep = []
    for col in X.columns:
        if all(c.loc[col, k] < thr for k in keep):
            keep.append(col)
    return keep


def inner_split(tr):
    cut = int(len(tr) * 0.75); a, v = tr.iloc[:cut], tr.iloc[cut:]
    return a[a.exit_time < v.index[0]], v


def fit(a, cols):
    p = dict(E.LGB_PARAMS, objective="multiclass", random_state=0)
    return lgb.LGBMClassifier(**p).fit(a[cols], a.label + 1, sample_weight=P.uniqueness_weights(a))


def perm_importance(m, v, cols, rng, reps=3):
    base = log_loss(v.label + 1, m.predict_proba(v[cols]), labels=[0, 1, 2]); imp = {}
    for c in cols:
        s = []
        for _ in range(reps):
            vv = v[cols].copy(); vv[c] = rng.permutation(vv[c].values)
            s.append(log_loss(v.label + 1, m.predict_proba(vv), labels=[0, 1, 2]) - base)
        imp[c] = np.mean(s)
    return base, imp


def rfecv(tr, cols, drop_frac=0.2, min_feats=4, seed=0):
    rng = np.random.default_rng(seed)
    a, v = inner_split(tr)
    cur = corr_prune(a[cols]); hist = []
    while True:
        m = fit(a, cur); ll, imp = perm_importance(m, v, cur, rng)
        hist.append((ll, list(cur)))
        if len(cur) <= min_feats:
            break
        k = max(1, int(len(cur) * drop_frac))
        cur = [c for c, _ in sorted(imp.items(), key=lambda x: -x[1])][:len(cur) - k]
    best = min(hist, key=lambda x: x[0])
    return best[1], hist
