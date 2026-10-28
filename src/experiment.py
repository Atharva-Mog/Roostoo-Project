"""Walk-forward experiment for one config with the full benchmark ladder.

Usage: python experiment.py <tag> [--bars 1h --hor 12h] [--all-grid] [--universe setups|all] [--feats file]
"""
import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
sys.path.insert(0, os.path.dirname(__file__))
import pipeline as P
warnings.filterwarnings("ignore")

LGB_PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=8, max_depth=3, min_child_samples=100,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, n_estimators=400,
                  verbose=-1)


def fit_predict(kind, tr, te, cols, params=None, weights=True, seed=0):
    """Fit on first 80% of train (time ordered), isotonic-calibrate on last 20%, predict test."""
    cut = int(len(tr) * 0.8)
    a, c = tr.iloc[:cut], tr.iloc[cut:]
    # purge calibration overlap: a's labels must end before c starts
    a = a[a["exit_time"] < c.index[0]]
    ya, yc = (a.label == 1).astype(int), (c.label == 1).astype(int)
    if kind == "lgb":
        p = dict(LGB_PARAMS, **(params or {}), random_state=seed)
        m = lgb.LGBMClassifier(**p)
        w = P.uniqueness_weights(a) if weights else None
        m.fit(a[cols], ya, sample_weight=w)
    else:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        m.fit(a[cols], ya)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso.fit(m.predict_proba(c[cols])[:, 1], yc)
    return iso.predict(m.predict_proba(te[cols])[:, 1]), m


def ev_net(p, tr, te):
    """Expected net return given P(TP). Non-TP outcome mean (in sigma units) estimated on train."""
    notp = tr[tr.label != 1]
    m0 = (notp.gross / notp.sigma).mean()
    return p * (P.K_TP * te.sigma - P.COST) + (1 - p) * (m0 * te.sigma - P.COST)


def run_config(b15, bar, hor, universe="setups", cols=None, params=None, margin_bps=0.0, weights=True,
               n_random=20, verbose=True):
    df = P.build(b15, bar, hor)
    if universe == "setups":
        df = df[df.setup]
    cols = cols or P.feat_cols(df)
    rows, preds = [], []
    for s, e in P.folds(df):
        tr, te = P.split(df, s, e, embargo_h=P.HOR_H[hor])
        if len(tr) < 500 or len(te) < 30:
            continue
        p_lgb, _ = fit_predict("lgb", tr, te, cols, params, weights)
        p_lr, _ = fit_predict("lr", tr, te, cols)
        ev_l = ev_net(p_lgb, tr, te); ev_r = ev_net(p_lr, tr, te)
        take_l = ev_l > margin_bps / 1e4; take_r = ev_r > margin_bps / 1e4
        y = (te.label == 1).astype(int)
        r = dict(fold=str(s.date()), n_train=len(tr), n_test=len(te), base_rate=y.mean(),
                 auc_lgb=roc_auc_score(y, p_lgb) if y.nunique() > 1 else np.nan,
                 auc_lr=roc_auc_score(y, p_lr) if y.nunique() > 1 else np.nan,
                 brier_lgb=brier_score_loss(y, p_lgb), brier_base=brier_score_loss(y, np.full(len(y), (tr.label == 1).mean())),
                 ll_lgb=log_loss(y, p_lgb, labels=[0, 1]),
                 ll_base=log_loss(y, np.full(len(y), (tr.label == 1).mean()), labels=[0, 1]))
        strat = {"every": np.ones(len(te), bool), "lr": take_r.values, "lgb": take_l.values}
        for k, mask in strat.items():
            for mk, v in P.perf(P.sequential(te, mask), s, e).items():
                r[f"{k}_{mk}"] = v
        # random filter matched to lgb take rate
        rate = take_l.mean(); rnd = []
        rng = np.random.default_rng(0)
        for _ in range(n_random):
            rnd.append(P.perf(P.sequential(te, rng.random(len(te)) < rate), s, e))
        for mk in ("n", "ev_bps", "total", "sharpe"):
            r[f"rand_{mk}"] = np.nanmean([x[mk] for x in rnd])
        # buy & hold over test period (log return)
        bh = b15.close[(b15.index >= s) & (b15.index < e)]
        r["bh_total"] = np.log(bh.iloc[-1] / bh.iloc[0])
        rows.append(r)
        preds.append(pd.DataFrame({"p_lgb": p_lgb, "p_lr": p_lr, "ev_lgb": ev_l, "y": y, "label": te.label,
                                   "net": te.net, "exit_time": te.exit_time}, index=te.index))
    res = pd.DataFrame(rows)
    if res.empty:
        return res, pd.DataFrame(), df
    pr = pd.concat(preds) if preds else pd.DataFrame()
    return res, pr, df


def pooled(res, pr, b15):
    """Pool trades across all dev folds into one sequential track (sequential() re-run on pooled preds)."""
    out = {}
    s, e = pd.Timestamp(res.fold.iloc[0], tz="UTC"), P.HOLDOUT_START
    for k, mask in {"every": np.ones(len(pr), bool), "lgb": (pr.ev_lgb > 0).values}.items():
        out[k] = P.perf(P.sequential(pr, mask), s, e)
    return out


def summarize(res):
    if res.empty:
        return {"n_folds": 0}
    keys = ["n_test", "base_rate", "auc_lgb", "auc_lr", "brier_lgb", "brier_base", "ll_lgb", "ll_base",
            "every_n", "every_ev_bps", "every_sharpe", "lr_n", "lr_ev_bps", "lr_sharpe",
            "lgb_n", "lgb_ev_bps", "lgb_hit", "lgb_sharpe", "lgb_total", "rand_n", "rand_ev_bps", "rand_sharpe",
            "every_total", "lr_total", "rand_total", "bh_total"]
    agg = {k: (res[k].sum() if k.endswith("_n") or k.endswith("total") or k == "n_test" else res[k].mean())
           for k in keys if k in res}
    agg["lgb_folds_positive"] = int((res["lgb_total"] > 0).sum()); agg["n_folds"] = len(res)
    return agg


# ======================= v3: three-class model, EV from all three calibrated probabilities ==============
def fit_predict_multi(kind, tr, te, cols, params=None, weights=True, seed=0):
    """Returns calibrated (n,3) probs for classes [SL, TO, TP]. One-vs-rest isotonic, renormalized."""
    cut = int(len(tr) * 0.8)
    a, c = tr.iloc[:cut], tr.iloc[cut:]
    a = a[a["exit_time"] < c.index[0]]
    ya, yc = a.label.values + 1, c.label.values + 1        # 0=SL,1=TO,2=TP
    if kind == "lgb":
        p = dict(LGB_PARAMS, **(params or {}), random_state=seed); p["objective"] = "multiclass"
        m = lgb.LGBMClassifier(**p)
        m.fit(a[cols], ya, sample_weight=P.uniqueness_weights(a) if weights else None)
    else:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
        m.fit(a[cols], ya)
    pc, pt = m.predict_proba(c[cols]), m.predict_proba(te[cols])
    out = np.zeros_like(pt)
    for k in range(3):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(pc[:, k], yc == k)
        out[:, k] = iso.predict(pt[:, k])
    return out / out.sum(1, keepdims=True)


def ev_multi(pp, tr, te):
    to = tr[tr.label == 0]
    m_to = (to.gross / to.sigma).mean()          # mean timeout return in sigma units (train)
    return (pp[:, 2] * P.K_TP - pp[:, 0] + pp[:, 1] * m_to) * te.sigma.values - P.COST


def run_config_v3(b15, bar, hor, universe="all", cols=None, params=None, margin_bps=0.0, weights=True,
                  n_random=10, df=None):
    if df is None:
        df = P.build(b15, bar, hor)
        if universe == "setups":
            df = df[df.setup]
    cols = cols or P.feat_cols(df)
    rows, preds = [], []
    for s, e in P.folds(df):
        tr, te = P.split(df, s, e, embargo_h=P.HOR_H[hor])
        if len(tr) < 500 or len(te) < 30:
            continue
        pl = fit_predict_multi("lgb", tr, te, cols, params, weights)
        pr_ = fit_predict_multi("lr", tr, te, cols)
        evl, evr = ev_multi(pl, tr, te), ev_multi(pr_, tr, te)
        takel, taker = evl > margin_bps / 1e4, evr > margin_bps / 1e4
        y = (te.label == 1).astype(int)
        r = dict(fold=str(s.date()), n_test=len(te), base_rate=y.mean(),
                 auc_lgb=roc_auc_score(y, pl[:, 2]), auc_lr=roc_auc_score(y, pr_[:, 2]),
                 ev_corr_lgb=np.corrcoef(evl, te.net)[0, 1], ev_corr_lr=np.corrcoef(evr, te.net)[0, 1])
        for k, mask in {"every": np.ones(len(te), bool), "lr": taker, "lgb": takel}.items():
            for mk, v in P.perf(P.sequential(te, mask), s, e).items():
                r[f"{k}_{mk}"] = v
        rng = np.random.default_rng(0); rate = takel.mean(); rnd = []
        for _ in range(n_random):
            rnd.append(P.perf(P.sequential(te, rng.random(len(te)) < rate), s, e))
        for mk in ("n", "ev_bps", "total", "sharpe"):
            r[f"rand_{mk}"] = np.nanmean([x[mk] for x in rnd])
        bh = b15.close[(b15.index >= s) & (b15.index < e)]
        r["bh_total"] = np.log(bh.iloc[-1] / bh.iloc[0])
        rows.append(r)
        preds.append(pd.DataFrame({"p_sl": pl[:, 0], "p_to": pl[:, 1], "p_tp": pl[:, 2], "ev_lgb": evl,
                                   "ev_lr": evr, "label": te.label, "net": te.net, "sigma": te.sigma,
                                   "exit_time": te.exit_time}, index=te.index))
    return pd.DataFrame(rows), (pd.concat(preds) if preds else pd.DataFrame()), df
