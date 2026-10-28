"""Decision agent: a trade-decision coach for a beginner BTC trader.

Loop (every DECISION_EVERY hours, bar by bar, no look-ahead):
  1. observe   - latest closed 1h bar + features (only data <= t)
  2. read market - volatility regime (percentile vs trailing year) + 4-week trend (robust base layer, run 15)
  3. predict   - calibrated P(TP first), P(SL first), P(time-out) over 24h  (3-class LightGBM, retrained monthly
                 on labels that had fully resolved before the retrain date)
  4. decide    - expected value AFTER THE TRADER'S OWN FEES -> TAKE / SKIP (fees eat it) / STAND ASIDE
  5. plan      - entry, take-profit (+1.5 sigma), stop (-1 sigma), 24h time limit, size so the stop risks 1% of account
  6. explain   - top-3 reasons from per-prediction SHAP contributions, in plain words
  7. confidence - low / medium / high from where the EV sits in the training distribution, with that bucket's
                 historical after-fee hit rate
  8. learn     - log every card; resolve outcomes only once their exit time has passed; track calibration of
                 P(TP) over the last 60 resolved predictions (logged even while paused, so the agent can recover); PAUSE when realized TP rate drifts below predicted (and resume when it recovers)

Usage:  RT_DATA=binance python src/decision_agent.py --fees 25 --tag beginner_spot
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("RT_DATA", "binance")
import pipeline as P, experiment as E

BAR, HOR, HOURS = "1h", "24h", 24
DECISION_EVERY = 4                      # hours between cards
RISK_PER_TRADE = 0.01                   # fraction of account lost if the stop is hit
CAL_WINDOW, CAL_MIN = 60, 30            # calibration monitor: last N resolved predictions (~10 days of cards)
PAUSE_Z = -1.64                         # pause if realized TP rate is 1.64 SE below predicted (one-sided 5%)

PLAIN = {   # feature -> (description when the value is HIGH vs training median, description when LOW)
    "rv_168h": ("weekly volatility is high", "weekly volatility is low"),
    "rv_24h": ("volatility over the last day is high", "the last day has been quiet"),
    "rv_4h": ("the last few hours have been volatile", "the last few hours have been quiet"),
    "dist_ma_168h": ("price is above its 1-week average", "price is below its 1-week average"),
    "dist_ma_24h": ("price is above its 24h average", "price has pulled back below its 24h average"),
    "vol_rel_168h": ("trading volume is building versus last week", "trading volume is fading versus last week"),
    "vol_rel_24h": ("volume right now is strong", "volume right now is thin"),
    "vol_of_vol": ("volatility itself is unstable", "volatility is steady"),
    "range_24h": ("the last day's range is wide", "the last day's range is narrow"),
    "weekend": ("it is the weekend (thinner liquidity)", "it is a weekday (deeper liquidity)"),
    "hour_sin": ("time of day", "time of day"), "hour_cos": ("time of day", "time of day"),
    "hrs_to_funding": ("the next funding payment is hours away", "a funding payment is imminent"),
    "x_btc_eth_corr_7d": ("BTC and ETH are moving together", "BTC and ETH are out of sync"),
    "x_eth_rel_24h": ("ETH is outperforming BTC today", "ETH is underperforming BTC today"),
    "x_eth_ret_4h_z": ("ETH has risen in the last few hours", "ETH has fallen in the last few hours"),
    "x_taker_buy_share_24h": ("buyers have been more aggressive today", "sellers have been more aggressive today"),
    "x_taker_buy_share_4h": ("aggressive buying in the last few hours", "aggressive selling in the last few hours"),
    "x_trades_rel_24h": ("trade activity is high", "trade activity is low"),
    "x_funding": ("futures funding is high (longs paying up)", "futures funding is low"),
    "x_funding_z_30d": ("funding is high for this month (crowded longs)", "funding is low for this month (little crowding)"),
    "x_funding_chg_24h": ("funding has risen over the day", "funding has cooled over the day"),
    "x_oi_chg_4h_z": ("open interest rose in the last few hours", "open interest fell in the last few hours"),
    "x_oi_chg_24h_z": ("open interest rose over the day (more leverage)", "open interest fell over the day (deleveraging)"),
    "x_oi_vs_7d": ("leverage is high versus last week", "leverage is low versus last week"),
    "x_crowding": ("leverage and funding are rising together (squeeze risk)", "no sign of crowded leverage"),
    "x_top_ls": ("top traders are heavily long", "top traders are not heavily long"),
    "x_global_ls_chg_24h": ("retail traders have added longs", "retail traders have cut longs"),
    "x_taker_ls_24h": ("futures buyers are aggressive", "futures sellers are aggressive"),
    "ret_720h_z": ("the 1-month trend is up", "the 1-month trend is down"),
    "ret_168h_z": ("the 1-week trend is up", "the 1-week trend is down"),
    "ret_24h_z": ("the last day was positive", "the last day was negative"),
    "ret_4h_z": ("the last few hours were positive", "the last few hours were negative"),
    "last_ret_z": ("the last hour jumped up", "the last hour dropped"),
    "rsi_24h": ("buying pressure dominated the day", "selling pressure dominated the day"),
    "upvol_share_24h": ("more volume traded on up-moves", "more volume traded on down-moves"),
    "rv_ratio_24_168": ("volatility is expanding", "volatility is contracting"),
}


def plain(f, value, median, contrib):
    hi, lo = PLAIN.get(f, (f"{f} is high", f"{f} is low"))
    what = hi if value > median else lo
    return f"{what[0].upper() + what[1:]}: {'supports' if contrib > 0 else 'works against'} the trade"


class Model:
    """3-class LightGBM + one-vs-rest isotonic calibration, trained on resolved labels only."""
    def fit(self, tr, cols):
        self.cols = cols
        cut = int(len(tr) * 0.8); a, c = tr.iloc[:cut], tr.iloc[cut:]
        a = a[a.exit_time < c.index[0]]
        p = dict(E.LGB_PARAMS, objective="multiclass", random_state=0)
        self.m = lgb.LGBMClassifier(**p).fit(a[cols], a.label + 1, sample_weight=P.uniqueness_weights(a))
        pc = self.m.predict_proba(c[cols])
        self.iso = [IsotonicRegression(out_of_bounds="clip", y_min=.001, y_max=.999).fit(pc[:, k], c.label.values + 1 == k)
                    for k in range(3)]
        to = tr[tr.label == 0]; self.m_to = (to.gross / to.sigma).mean()
        # reference distribution of gross edge (in sigma units) on the calibration slice -> confidence buckets
        g = self._edge(self.proba(c))
        self.q70, self.q90 = np.quantile(g, [.7, .9])
        cal = c.assign(edge=g)
        self.bucket_stats = {}
        for name, lo, hi in (("low", -np.inf, self.q70), ("medium", self.q70, self.q90), ("high", self.q90, np.inf)):
            s = cal[(cal.edge > lo) & (cal.edge <= hi)]
            self.bucket_stats[name] = dict(tp_rate=float((s.label == 1).mean()), n=int(len(s)),
                                           gross_bps=float(s.gross.mean() * 1e4))
        self.med = tr[cols].median()
        return self

    def proba(self, X):
        pr = self.m.predict_proba(X[self.cols])
        out = np.column_stack([self.iso[k].predict(pr[:, k]) for k in range(3)])
        return out / out.sum(1, keepdims=True)

    def _edge(self, pp):                       # expected gross move in sigma units
        return pp[:, 2] * P.K_TP - pp[:, 0] + pp[:, 1] * self.m_to

    def shap(self, X):                          # contribution to log-odds of TP minus SL
        c = self.m.predict(X[self.cols], pred_contrib=True)   # (n, 3*(F+1))
        F = len(self.cols) + 1
        return c[:, 2 * F:3 * F - 1] - c[:, 0:F - 1]


def run(fees_bps, tag, start=P.HOLDOUT_START):
    b15 = P.load_15m()
    df = P.build(b15, BAR, HOR)
    cols = P.feat_cols(df)
    day = b15.close.resample("1D").last()
    rv_hist = df.rv_168h
    cost = fees_bps / 1e4
    cards, log = [], []          # log = TAKE signals awaiting/after resolution
    paused, model, model_month = False, None, None
    times = df.index[(df.index >= start) & (df.index.hour % DECISION_EVERY == DECISION_EVERY - 1)]
    for t in times:
        # ---- retrain at the start of each month on labels resolved before t (no look-ahead)
        if model_month != (t.year, t.month):
            tr = df[(df.index < t) & (df.exit_time < t)]
            model, model_month = Model().fit(tr, cols), (t.year, t.month)
        row = df.loc[[t]]
        price = float(b15.close.loc[t]); sigma = float(row.sigma.iloc[0])
        # ---- 2. market read
        past = rv_hist[(rv_hist.index <= t) & (rv_hist.index > t - pd.Timedelta(days=365))]
        vol_pct = float((past < row.rv_168h.iloc[0]).mean() * 100)
        d = day[day.index <= t.normalize() - pd.Timedelta(days=1)]
        trend_up = bool(np.log(d.iloc[-1] / d.iloc[-29]) > 0) if len(d) > 29 else None
        # ---- 3. predict
        pp = model.proba(row)[0]
        edge_sigma = float(model._edge(pp[None])[0])
        gross = edge_sigma * sigma
        ev = gross - cost
        # ---- 8a. resolve past TAKE signals whose exit time has passed; calibration monitor
        for s in log:
            if s["outcome"] is None and pd.Timestamp(s["exit_time"]) <= t:
                s["outcome"] = s["_label"]; s["net"] = s["_gross"] - cost
        res = [s for s in log if s["outcome"] is not None][-CAL_WINDOW:]
        cal = None
        if len(res) >= CAL_MIN:
            pred = np.mean([s["p_tp"] for s in res]); real = np.mean([s["outcome"] == 1 for s in res])
            se = np.sqrt(max(pred * (1 - pred), 1e-4) / len(res)); z = (real - pred) / se
            cal = dict(n=len(res), predicted_tp=round(pred, 3), realized_tp=round(real, 3), z=round(z, 2),
                       net_bps=round(np.mean([s["net"] for s in res]) * 1e4, 1))
            paused = z < PAUSE_Z
        # ---- 4. decide
        conf = "high" if edge_sigma > model.q90 else "medium" if edge_sigma > model.q70 else "low"
        if paused:
            state = "PAUSED"
        elif ev > 0:
            state = "TAKE"
        elif gross > 0:
            state = "SKIP"           # positive edge, but smaller than this trader's fees
        else:
            state = "STAND ASIDE"
        # ---- 5. plan
        tp, sl = price * np.exp(P.K_TP * sigma), price * np.exp(-sigma)
        size_frac = min(RISK_PER_TRADE / (1 - np.exp(-sigma)), 1.0)    # fraction of account in BTC
        # ---- 6. explain
        sh = model.shap(row)[0]; order = np.argsort(-np.abs(sh))[:3]
        reasons = [plain(cols[i], float(row[cols[i]].iloc[0]), float(model.med[cols[i]]), sh[i]) for i in order]
        base = None if trend_up is None else ("4-week trend is UP: if you already hold BTC, holding has historically beaten trading in this state"
                                              if trend_up else "4-week trend is DOWN: staying in cash has historically avoided most of the drawdown")
        card = dict(time=str(t), price=round(price, 2), state=state, confidence=conf,
                    odds=dict(tp_first=round(pp[2], 3), stop_first=round(pp[0], 3), nothing=round(pp[1], 3)),
                    expected_gross_bps=round(gross * 1e4, 1), your_fees_bps=fees_bps, expected_net_bps=round(ev * 1e4, 1),
                    plan=dict(entry=round(price, 2), take_profit=round(tp, 2), stop=round(sl, 2), time_limit_h=HOURS,
                              size_pct_of_account=round(size_frac * 100, 1), risk_pct=RISK_PER_TRADE * 100),
                    market=dict(vol_percentile=round(vol_pct), trend_4w=("up" if trend_up else "down") if trend_up is not None else None),
                    reasons=reasons, base_layer=base, confidence_history=model.bucket_stats[conf], calibration=cal,
                    _label=int(row.label.iloc[0]), _gross=float(row.gross.iloc[0]), exit_time=str(row.exit_time.iloc[0]))
        cards.append(card)
        # log EVERY prediction (incl. while paused): the monitor checks recent calibration of P(TP), so it can resume
        if True:
            log.append(dict(time=str(t), p_tp=float(pp[2]), exit_time=card["exit_time"], outcome=None, net=None,
                            _label=card["_label"], _gross=card["_gross"]))
    return cards


def summarize(cards, fees_bps):
    c = pd.DataFrame(cards); c["time"] = pd.to_datetime(c.time)
    c["net"] = c._gross - fees_bps / 1e4
    # sequential execution of TAKE cards (one position at a time) -> realized track record
    took, busy = [], None
    for _, r in c[c.state == "TAKE"].iterrows():
        if busy is not None and r.time < busy: continue
        took.append(r); busy = pd.Timestamp(r.exit_time)
    tk = pd.DataFrame(took)
    s = dict(cards=len(c), states=c.state.value_counts().to_dict(),
             trades_taken=len(tk), take_hit_rate=float((tk._label == 1).mean()) if len(tk) else None,
             take_net_bps=float(tk.net.mean() * 1e4) if len(tk) else None, take_total_pct=float(tk.net.sum() * 100) if len(tk) else 0.0,
             skipped_setups_if_traded_net_bps=float(c[c.state.isin(["SKIP", "STAND ASIDE"])].net.mean() * 1e4),
             every_card_if_traded_net_bps=float(c.net.mean() * 1e4),
             by_confidence={k: dict(n=int(len(g)), tp_rate=round(float((g._label == 1).mean()), 3), net_bps=round(float(g.net.mean() * 1e4), 1))
                            for k, g in c.groupby("confidence")},
             paused_pct=float((c.state == "PAUSED").mean() * 100))
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fees", type=float, default=25); ap.add_argument("--tag", default="beginner_spot")
    a = ap.parse_args()
    out = os.path.join(P.ROOT, "results", "agent"); os.makedirs(out, exist_ok=True)
    cards = run(a.fees, a.tag)
    pub = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cards]
    with open(os.path.join(out, f"cards_{a.tag}.jsonl"), "w") as f:
        for c in pub: f.write(json.dumps(c) + "\n")
    s = summarize(cards, a.fees)
    json.dump(s, open(os.path.join(out, f"summary_{a.tag}.json"), "w"), indent=2)
    print(json.dumps(s, indent=2))
