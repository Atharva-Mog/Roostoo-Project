"""Run 19: next-24h volatility forecast. Daily forecasts at 00:00 UTC (decision = close of 23:45 15m bar).
Target: log realized variance over the next 24h (sum of squared 15m log returns, 96 bars).
Baselines: (1) random walk = last 24h RV, (2) EWMA of 15m squared returns (span 3 days) x 96,
(3) HAR-RV (log RV on log daily/weekly/monthly RV, OLS fitted on the training fold).
Model: LightGBM regression on the 1h feature set + HAR terms. Same 6-month walk-forward folds as before.
Variance forecasts from log models are bias-corrected with the training-fold smearing factor.
Scores: QLIKE (lower better), MSE of log RV, R^2 of log RV, Diebold-Mariano-style t-test on QLIKE loss differences.
Use: vol-targeted BTC holding (target 40% annualized, max 1x, rebalance cost 12.5 bps per unit) vs buy & hold."""
import os, sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(__file__)); os.environ["RT_DATA"] = "binance"
import pipeline as P
from sklearn.linear_model import LinearRegression
b15 = P.load_15m()
r15 = np.log(b15.close).diff()
rv15 = (r15 ** 2)
# realized variance of the NEXT 96 bars, indexed at decision bar t (bar t is the last known)
fwd = rv15[::-1].rolling(96).sum()[::-1].shift(-1)
past24 = rv15.rolling(96).sum()
ewma = rv15.ewm(span=96 * 3).mean() * 96
f1h = P.features(P.resample(b15, "1h"), 60); f1h.index = f1h.index + pd.Timedelta(minutes=45)   # align to 15m bar close
idx = b15.index[(b15.index.hour == 23) & (b15.index.minute == 45)]
D = pd.DataFrame(index=idx)
D["y"] = np.log(fwd.reindex(idx))
D["rw"] = np.log(past24.reindex(idx)); D["ewma"] = np.log(ewma.reindex(idx))
D["h_d"] = D["rw"]; D["h_w"] = np.log(past24.rolling(96 * 7).mean().reindex(idx)); D["h_m"] = np.log(past24.rolling(96 * 30).mean().reindex(idx))
F = f1h.reindex(idx)
X = pd.concat([D[["h_d", "h_w", "h_m", "ewma"]], F], axis=1)
D = D.join(F); D = D[D.index >= "2021-03-01"].dropna(subset=["y", "rw", "ewma", "h_d", "h_w", "h_m"])
D["exit_time"] = D.index + pd.Timedelta(hours=24)
cols = ["h_d", "h_w", "h_m", "ewma"] + list(F.columns)
LGBP = dict(objective="regression", learning_rate=0.02, num_leaves=8, max_depth=3, min_child_samples=50, subsample=0.7,
            subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, n_estimators=400, verbose=-1, random_state=0)
def smear(res): return np.mean(np.exp(res))
periods = list(P.folds(D)) + [(P.HOLDOUT_START, D.index[-1] + pd.Timedelta(days=1))]
out = []
for s, e in periods:
    tr, te = P.split(D, s, e, 24)
    pred = pd.DataFrame(index=te.index)
    # baselines: rw and ewma are already forecasts of next-24h variance (log); calibrate level with training smearing
    for b in ("rw", "ewma"):
        k = smear(tr.y - tr[b]); pred[b] = np.log(np.exp(te[b]) * k)
    har = LinearRegression().fit(tr[["h_d", "h_w", "h_m"]], tr.y)
    pred["har"] = np.log(np.exp(har.predict(te[["h_d", "h_w", "h_m"]])) * smear(tr.y - har.predict(tr[["h_d", "h_w", "h_m"]])))
    m = lgb.LGBMRegressor(**LGBP).fit(tr[cols], tr.y)
    pred["lgbm"] = np.log(np.exp(m.predict(te[cols])) * smear(tr.y - m.predict(tr[cols])))
    pred["y"] = te.y; pred["period"] = "LAST 6M*" if s == P.HOLDOUT_START else "dev"
    out.append(pred)
R = pd.concat(out)
def qlike(y, f): v, h = np.exp(y), np.exp(f); return v / h - np.log(v / h) - 1
rows = []
for per, g in (("dev 2022-07..2026-04", R[R.period == "dev"]), ("last 6m*", R[R.period != "dev"])):
    for mname in ("rw", "ewma", "har", "lgbm"):
        q = qlike(g.y, g[mname]); d = qlike(g.y, g["har"]) - q          # positive = better than HAR
        rows.append(dict(period=per, model=mname, n=len(g), qlike=q.mean(), mse_log=((g.y - g[mname]) ** 2).mean(),
                         r2_log=1 - ((g.y - g[mname]) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum(),
                         gain_vs_har_pct=(d.mean() / qlike(g.y, g["har"]).mean()) * 100,
                         t_vs_har=d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if mname != "har" else np.nan))
T = pd.DataFrame(rows).round(4); pd.set_option("display.width", 200); print(T.to_string(index=False))
# vol targeting: hold BTC with weight min(1, target / forecast vol); apply weight to next day's return
day = b15.close.resample("1D").last(); dret = np.log(day).diff().shift(-1)            # return of the day after the forecast date
dret.index = dret.index + pd.Timedelta(hours=23, minutes=45)
TGT = 0.40 / np.sqrt(365)
def voltarget(fc):
    w = np.minimum(1.0, TGT / np.sqrt(np.exp(fc))); r = dret.reindex(fc.index)
    return w * r - np.abs(pd.Series(w, fc.index).diff().fillna(0)) * 0.00125, w
def st(x, w=None):
    eq = x.cumsum(); return dict(ann_ret=x.mean() * 365 * 100, ann_vol=x.std() * np.sqrt(365) * 100, sharpe=x.mean() / x.std() * np.sqrt(365),
                                 maxdd=(np.exp(eq) / np.exp(eq).cummax() - 1).min() * 100, avg_weight=np.mean(w) if w is not None else 1.0)
rows = []
for per, g in (("dev", R[R.period == "dev"]), ("last 6m*", R[R.period != "dev"])):
    rows.append(dict(period=per, strategy="buy & hold", **st(dret.reindex(g.index).dropna())))
    for mname in ("ewma", "har", "lgbm"):
        x, w = voltarget(g[mname]); rows.append(dict(period=per, strategy=f"vol-target ({mname})", **st(x.dropna(), w)))
V = pd.DataFrame(rows).round(3); print(); print(V.to_string(index=False))
os.makedirs("results/v19_vol", exist_ok=True); T.to_csv("results/v19_vol/forecast_scores.csv", index=False); V.to_csv("results/v19_vol/vol_target.csv", index=False); R.to_csv("results/v19_vol/forecasts.csv")
imp = pd.Series(m.feature_importances_, cols).sort_values(ascending=False).head(10); print("\ntop features (last model):", imp.to_dict())
