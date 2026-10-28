"""Run 9: execution-aware cost model for BTC USDT-M perpetuals (Binance VIP0: maker 0.02%, taker 0.05%).
- Entry: post-only limit at decision close minus 2 bps; FILLED only if a later 15m low within the next hour
  trades through it (strictly below), else the trade is skipped (models fill risk + adverse selection).
- TP exit: resting limit -> maker. SL and time-out exits: taker + 2.5 bps slippage.
- Funding: longs pay ~1 bp per 8h held (2021-26 average BTC funding ~0.01%/8h).
Gate uses the outcome-dependent expected cost."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pipeline as P, experiment as E

MAKER, TAKER, SLIP, OFFSET, FUND_8H = 0.0002, 0.0005, 0.00025, 0.0002, 0.0001

def add_execution(df, b15):
    lo = np.log(b15.low).reindex(df.index.union(b15.index)).sort_index()
    low = b15.low.values; idx = b15.index.get_indexer(df.index)
    nxt = np.full(len(df), np.inf)
    for k in range(1, 5):                                  # next 4 x 15m bars = 1 hour
        j = np.clip(idx + k, 0, len(low) - 1); nxt = np.minimum(nxt, low[j])
    closep = b15.close.values[idx]
    df = df.copy()
    df["filled"] = nxt < closep * (1 - OFFSET)
    hold_h = df.exit_off * 0.25
    exit_cost = np.where(df.label == 1, MAKER, TAKER + SLIP)
    # entry 2 bps better than close (log approx) but label measured from close: conservative is to ignore the gain
    df["net"] = df.gross - MAKER - exit_cost - FUND_8H * hold_h / 8
    return df

def ev_perp(pp, tr, te, hours):
    to = tr[tr.label == 0]; m_to = (to.gross / to.sigma).mean()
    gross = (pp[:, 2] * P.K_TP - pp[:, 0] + pp[:, 1] * m_to) * te.sigma.values
    cost = MAKER + pp[:, 2] * MAKER + (1 - pp[:, 2]) * (TAKER + SLIP) + FUND_8H * hours / 2 / 8
    return gross - cost

def run(b15, bar, hor, margin_bps=0.0, s0=None, e0=None):
    df = add_execution(P.build(b15, bar, hor), b15); cols = P.feat_cols(df); cols = [c for c in cols if c != "filled"]
    fl = P.folds(df) if s0 is None else [(s0, e0)]
    rows, trades_all = [], []
    for s, e in fl:
        tr, te = P.split(df, s, e, embargo_h=P.HOR_H[hor])
        if len(tr) < 500 or len(te) < 30: continue
        pl = E.fit_predict_multi("lgb", tr, te, cols); ev = ev_perp(pl, tr, te, P.HOR_H[hor])
        f = te.filled.values
        strategies = {"lgb": (ev > margin_bps / 1e4) & f, "every": f}
        rng = np.random.default_rng(0); rate = ((ev > margin_bps / 1e4) & f).sum() / max(f.sum(), 1)
        r = dict(fold=str(s.date()))
        for k, m in strategies.items():
            t = P.sequential(te, m)
            if k == "lgb": trades_all.append(t)
            for mk, v in P.perf(t, s, e).items(): r[f"{k}_{mk}"] = v
        rnd = [P.perf(P.sequential(te, f & (rng.random(len(te)) < rate)), s, e) for _ in range(20)]
        for mk in ("n", "ev_bps", "total", "sharpe"): r[f"rand_{mk}"] = np.nanmean([x[mk] for x in rnd])
        rows.append(r)
    return pd.DataFrame(rows), (pd.concat(trades_all) if trades_all else pd.Series(dtype=float))

if __name__ == "__main__":
    b15 = P.load_15m(); out = []; os.makedirs("results/v9_perp", exist_ok=True)
    for bar, hor in P.GRID:
        res, t = run(b15, bar, hor)
        res.to_csv(f"results/v9_perp/folds_{bar}_{hor}.csv", index=False); t.to_csv(f"results/v9_perp/trades_{bar}_{hor}.csv")
        s = dict(cfg=f"{bar}_{hor}", lgb_n=res.lgb_n.sum(), lgb_ev_bps=t.mean() * 1e4, lgb_total=res.lgb_total.sum(),
                 lgb_sharpe=res.lgb_sharpe.mean(), pos_folds=f"{(res.lgb_total > 0).sum()}/{len(res)}",
                 every_ev=res.every_ev_bps.mean(), every_total=res.every_total.sum(), rand_ev=res.rand_ev_bps.mean(), rand_total=res.rand_total.sum())
        out.append(s); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    pd.DataFrame(out).to_csv("results/v9_perp/grid_summary.csv", index=False)
