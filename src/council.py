"""Run 16: adaptive council of experts (multiplicative weights / Hedge).

Experts (all daily, net of costs, fully out-of-sample):
  ml     - ML trade filter: 1h bars / 24h horizon, full Binance features, perp execution costs (runs 9/12),
           walk-forward with 6-month folds running all the way to today. P&L is booked on the EXIT day.
  trend  - 4-week time-series momentum on spot (run 15): long next day if trailing 28d return > 0, else cash.
  carry  - funding carry (long spot + short perp); return on capital = half the funding (two legs need capital).
  cash   - 0.

Combiner: w_i(d) proportional to exp(eta * S_i(d-1)), where S_i is the discounted sum of expert i's daily net returns
          up to the END of day d-1 (so weights only use resolved P&L), S_i(d) = gamma * S_i(d-1) + r_i(d).
          Portfolio return(d) = sum_i w_i(d) r_i(d) - turnover cost of weight changes.
Selection: (eta, gamma) chosen on 2022-07-01..2024-12-31 ONLY; reported for every grid point. Test: 2025-01-01..now.
Benchmarks: equal weights, best single expert chosen on the selection period, buy & hold."""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); os.environ["RT_DATA"] = "binance"
import pipeline as P, experiment as E, perp_costs as C

OUT = os.path.join(P.ROOT, "results", "v16_council"); os.makedirs(OUT, exist_ok=True)
SEL = ("2022-07-01", "2024-12-31"); TEST = ("2025-01-01", "2026-10-26")
SPOT = 0.00125; REBAL = 0.00075            # per unit of weight moved into/out of a risky expert
ETAS = (0, 2, 5, 10, 20, 50); GAMMAS = (1.0, 0.99, 0.97)


def ml_expert(b15):
    cache = os.path.join(OUT, "ml_daily.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    df = C.add_execution(P.build(b15, "1h", "24h"), b15); cols = [c for c in P.feat_cols(df) if c != "filled"]
    trades = []
    for s, e in P.folds(df, end=df.index[-1] + pd.Timedelta(hours=1)):
        tr, te = P.split(df, s, e, embargo_h=24)
        pl = E.fit_predict_multi("lgb", tr, te, cols); ev = C.ev_perp(pl, tr, te, 24)
        sel = te[(ev > 0) & te.filled.values]; busy = None
        for t, r in sel.iterrows():
            if busy is not None and t < busy: continue
            trades.append((r.exit_time, r.net)); busy = r.exit_time
        print("ml fold", s.date(), len(trades), flush=True)
    s = pd.Series([x[1] for x in trades], index=pd.DatetimeIndex([x[0] for x in trades]))
    d = s.groupby(s.index.normalize()).sum()
    d.to_csv(cache); return d


def experts():
    b15 = P.load_15m()
    day = b15.close.resample("1D").last(); dlr = np.log(day).diff().dropna()
    idx = dlr.index
    ml = ml_expert(b15).reindex(idx, fill_value=0.0)
    sig = (np.log(day / day.shift(28)) > 0).astype(int).shift(1).reindex(idx).fillna(0)
    trend = sig * dlr - sig.diff().abs().fillna(0) * SPOT
    fd = b15.funding.resample("8h").last().dropna().resample("1D").sum().reindex(idx).fillna(0)
    carry = fd / 2
    if DIRECTIONAL:   # run 16c: comparable directional views; idle capital (the flat weight) earns funding carry
        X = pd.DataFrame({"ml": ml, "trend": trend, "hold": dlr, "flat": 0.0}, index=idx)
        globals()["CARRY"] = carry[carry.index >= "2022-07-01"]
    else:
        X = pd.DataFrame({"ml": ml, "trend": trend, "carry": carry, "cash": 0.0}, index=idx)
    return X[X.index >= "2022-07-01"], dlr[dlr.index >= "2022-07-01"]


def hedge(X, eta, gamma, risk_adj=False):
    """risk_adj=True (run 16b): reward = r_i / trailing 60d vol of expert i (lagged one day), so a high-volatility
    expert cannot dominate the weights just by having bigger swings."""
    Z = X / X.rolling(60, min_periods=20).std().shift(1).replace(0, np.nan) if risk_adj else X
    Z = Z.fillna(0.0)
    n = X.shape[1]; S = np.zeros(n); W, R = [], []
    w_prev = np.ones(n) / n
    risky = np.array([c not in ("cash", "flat") for c in X.columns])
    carry = CARRY.reindex(X.index).fillna(0).values if DIRECTIONAL else np.zeros(len(X))
    flat_i = list(X.columns).index("flat") if "flat" in X.columns else None
    for (d, r), z_r in zip(X.iterrows(), Z.values):
        z = eta * S; w = np.exp(z - z.max()); w /= w.sum()          # uses only P&L resolved before day d
        cost = np.abs(w - w_prev)[risky].sum() * REBAL
        idle = w[flat_i] * carry[len(R)] if flat_i is not None else 0.0
        R.append(float(w @ r.values) + idle - cost); W.append(w); w_prev = w
        S = gamma * S + z_r
    return pd.Series(R, X.index), pd.DataFrame(W, X.index, X.columns)


def stats(r, s, e):
    x = r[(r.index >= s) & (r.index <= e)]; eq = x.cumsum()
    return dict(ann_ret=x.sum() / (len(x) / 365) * 100, sharpe=x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan,
                maxdd=(eq - eq.cummax()).min() * 100, total=x.sum() * 100)


DIRECTIONAL = "--directional" in sys.argv
CARRY = None

if __name__ == "__main__":
    RISK = "--risk-adj" in sys.argv
    if RISK:
        ETAS = (0, 0.02, 0.05, 0.1, 0.2, 0.5); OUT = os.path.join(P.ROOT, "results", "v16b_council_riskadj"); os.makedirs(OUT, exist_ok=True)
        import shutil; src = os.path.join(P.ROOT, "results", "v16_council", "ml_daily.csv")
        if DIRECTIONAL: OUT = os.path.join(P.ROOT, "results", "v16c_council_directional"); os.makedirs(OUT, exist_ok=True)
        if os.path.exists(src): shutil.copy(src, os.path.join(OUT, "ml_daily.csv"))
    X, bh = experts()
    X.to_csv(os.path.join(OUT, "expert_daily_returns.csv"))
    rows = []
    for eta in ETAS:
        for g in GAMMAS:
            if eta == 0 and g != 1.0: continue
            r, W = hedge(X, eta, g, RISK)
            rows.append(dict(eta=eta, gamma=g, **{f"sel_{k}": v for k, v in stats(r, *SEL).items()},
                             **{f"test_{k}": v for k, v in stats(r, *TEST).items()}))
    grid = pd.DataFrame(rows).round(3); grid.to_csv(os.path.join(OUT, "hedge_grid.csv"), index=False)
    best = grid.sort_values("sel_sharpe", ascending=False).iloc[0]
    eta, g = float(best.eta), float(best.gamma)
    r, W = hedge(X, eta, g, RISK); W.to_csv(os.path.join(OUT, "weights.csv")); r.to_csv(os.path.join(OUT, "council_daily.csv"))
    sel_best_expert = max(X.columns, key=lambda c: stats(X[c], *SEL)["sharpe"] if X[c].std() > 0 else -9)
    eq_w, _ = hedge(X, 0, 1.0, RISK)
    comp = {f"Council (Hedge eta={eta:g}, gamma={g:g})": r, "Equal weights (eta=0)": eq_w,
            f"Best expert on selection period ({sel_best_expert})": X[sel_best_expert], "Buy & hold BTC": bh}
    comp.update({f"Expert alone: {c}": X[c] for c in X.columns if c not in ("cash", "flat")})
    if DIRECTIONAL: comp["Flat earning carry only"] = CARRY
    rows = []
    for k, v in comp.items():
        for pn, (s, e) in (("selection 2022-07..2024-12", SEL), ("TEST 2025-01..now", TEST)):
            rows.append(dict(strategy=k, period=pn, **{a: round(b, 2) for a, b in stats(v, s, e).items()}))
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
    pd.set_option("display.width", 200)
    print(grid.to_string(index=False)); print()
    for pn in res.period.unique(): print("==", pn); print(res[res.period == pn].drop(columns="period").to_string(index=False))
    print("\nmean weights in TEST:", W[W.index >= TEST[0]].mean().round(3).to_dict())
    print("weights by quarter:\n", W.resample("QE").mean().round(2).to_string())
