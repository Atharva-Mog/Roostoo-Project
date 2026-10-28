"""Run 18: add a SHORT side (perpetual futures, 1x, same 1.5:1 barriers, same 1h/24h config and model).
Short label: TP = price falls 1.5 sigma before rising 1 sigma (15m path; same-bar ties -> stop first).
Execution: sell-limit 2 bps above the decision close, filled only if a later 15m high within 1h trades through it;
TP exit maker, stop/time-out taker + slippage. Funding: shorts RECEIVE the actual funding rate while holding
(pay when negative), using the funding rate known at entry x hold time.
Strategies (one position at a time): long-only, short-only, both (take the side with higher positive EV), and
random filters at matched rates. Dev = 8 walk-forward folds 2022-07..2026-04; last 6m reported separately (seen before)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); os.environ["RT_DATA"] = "binance"
from numpy.lib.stride_tricks import sliding_window_view
import pipeline as P, experiment as E, perp_costs as C

def short_labels(b15, hours, k=P.K_TP):
    n = int(hours * 4); lc = np.log(b15.close.values); hi = np.log(b15.high.values); lo = np.log(b15.low.values)
    sig = P.sigma_horizon(b15, hours).values
    H = sliding_window_view(hi[1:], n); L = sliding_window_view(lo[1:], n); Cc = sliding_window_view(lc[1:], n)
    m = H.shape[0]; base = lc[:m, None]; s = sig[:m, None]
    up = (base - L) >= k * s; dn = (H - base) >= s            # 'up' = profit for a short
    BIG = n + 1
    tu = np.where(up.any(1), up.argmax(1), BIG); td = np.where(dn.any(1), dn.argmax(1), BIG)
    lab = np.zeros(m, int); lab[(td <= tu) & (td < BIG)] = -1; lab[tu < td] = 1
    gross = np.where(lab == 1, k * sig[:m], np.where(lab == -1, -sig[:m], -(Cc[:, -1] - lc[:m])))
    off = np.where(lab == 1, tu + 1, np.where(lab == -1, td + 1, n))
    return pd.DataFrame({"label": lab, "gross": gross, "exit_off": off, "sigma": sig[:m]}, index=b15.index[:m]).dropna()

def build_short(b15, long_df):
    s = short_labels(b15, 24).reindex(long_df.index)
    df = long_df.copy()
    for c in ("label", "gross", "exit_off", "sigma"): df[c] = s[c].values
    df = df.dropna(subset=["label"]); df["label"] = df.label.astype(int)
    idx = b15.index.get_indexer(df.index); high = b15.high.values; nxt = np.full(len(df), -np.inf)
    for k in range(1, 5): nxt = np.maximum(nxt, high[np.clip(idx + k, 0, len(high) - 1)])
    df["filled"] = nxt > b15.close.values[idx] * (1 + C.OFFSET)
    hold_h = df.exit_off * 0.25
    exit_cost = np.where(df.label == 1, C.MAKER, C.TAKER + C.SLIP)
    df["net"] = df.gross - C.MAKER - exit_cost + df.funding_entry * hold_h / 8      # shorts receive funding
    df["exit_time"] = df.index + pd.to_timedelta(df.exit_off * 15, unit="m")
    return df

def ev_short(pp, tr, te):
    to = tr[tr.label == 0]; m_to = (to.gross / to.sigma).mean()
    gross = (pp[:, 2] * P.K_TP - pp[:, 0] + pp[:, 1] * m_to) * te.sigma.values
    cost = C.MAKER + pp[:, 2] * C.MAKER + (1 - pp[:, 2]) * (C.TAKER + C.SLIP)
    return gross - cost + te.funding_entry.values * 12 / 8          # expected funding over ~12h average hold

def seq(events):
    """events: DataFrame index=time with net, exit_time; one position at a time."""
    out, busy = [], None
    for t, r in events.sort_index().iterrows():
        if busy is not None and t < busy: continue
        out.append((t, r.net)); busy = r.exit_time
    return pd.Series([x[1] for x in out], index=[x[0] for x in out], dtype=float)

b15 = P.load_15m()
L = C.add_execution(P.build(b15, "1h", "24h"), b15)
fund = b15.funding.reindex(L.index).ffill().values
L["funding_entry"] = fund
S = build_short(b15, L.drop(columns=["filled"]))
cols = [c for c in P.feat_cols(L) if c not in ("filled", "funding_entry")]
print("short label mix", S.label.value_counts(normalize=True).round(3).to_dict(), "| long", L.label.value_counts(normalize=True).round(3).to_dict())
periods = [(s, e) for s, e in P.folds(L)] + [(P.HOLDOUT_START, L.index[-1] + pd.Timedelta(hours=1))]
res = {k: [] for k in ("long", "short", "both", "rand_both")}; rows = []
for s, e in periods:
    trL, teL = P.split(L, s, e, 24); trS, teS = P.split(S, s, e, 24)
    pL = E.fit_predict_multi("lgb", trL, teL, cols); pS = E.fit_predict_multi("lgb", trS, teS, cols)
    evL = pd.Series(C.ev_perp(pL, trL, teL, 24), teL.index); evS = pd.Series(ev_short(pS, trS, teS), teS.index)
    lt = teL[(evL > 0) & teL.filled]; st = teS[(evS > 0) & teS.filled]
    common = evL.index.intersection(evS.index)
    pick_long = (evL.reindex(common) >= evS.reindex(common))
    bl = teL.loc[common][pick_long & (evL.reindex(common) > 0) & teL.filled.reindex(common)]
    bs = teS.loc[common][~pick_long & (evS.reindex(common) > 0) & teS.filled.reindex(common)]
    both = pd.concat([bl[["net", "exit_time"]], bs[["net", "exit_time"]]])
    tl, ts, tb = seq(lt), seq(st), seq(both)
    # random: same number of candidate events per side, drawn from filled events
    rng = np.random.default_rng(0); rr = []
    for _ in range(30):
        a = teL[teL.filled].sample(n=min(len(bl), teL.filled.sum()), random_state=rng.integers(1e9))
        b = teS[teS.filled].sample(n=min(len(bs), teS.filled.sum()), random_state=rng.integers(1e9))
        rr.append(seq(pd.concat([a[["net", "exit_time"]], b[["net", "exit_time"]]])).sum())
    hold = np.log(b15.close[(b15.index >= s) & (b15.index < e)]).diff().sum()
    tag = "LAST 6M*" if s == P.HOLDOUT_START else str(s.date())
    rows.append(dict(period=tag, long_n=len(tl), long_tot=tl.sum() * 100, short_n=len(ts), short_tot=ts.sum() * 100,
                     both_n=len(tb), both_short_share=(tb.index.isin(bs.index)).mean() if len(tb) else 0, both_tot=tb.sum() * 100,
                     rand_both_tot=np.mean(rr) * 100, p_vs_rand=(np.sum(np.array(rr) >= tb.sum()) + 1) / 31, btc_hold=hold * 100))
    for k, v in (("long", tl), ("short", ts), ("both", tb)): res[k].append((tag, v))
r = pd.DataFrame(rows).round(3); pd.set_option("display.width", 220); print(r.to_string(index=False))
dev = r[r.period != "LAST 6M*"]
print("\nDEV totals (2022-07..2026-04): long %.1f%% | short %.1f%% | both %.1f%% (random both %.1f%%) | BTC hold %.1f%% | both folds>0: %d/%d"
      % (dev.long_tot.sum(), dev.short_tot.sum(), dev.both_tot.sum(), dev.rand_both_tot.sum(), dev.btc_hold.sum(), (dev.both_tot > 0).sum(), len(dev)))
os.makedirs("results/v18_shorts", exist_ok=True); r.to_csv("results/v18_shorts/folds.csv", index=False)
