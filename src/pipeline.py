"""Core pipeline: bars, triple-barrier labels, features, primary rule, walk-forward model, backtest.

Conventions
- Base data: 15m bars. A config = (bar size for features & decisions, horizon).
- Decision at the CLOSE of a bar; entry at that close; barriers checked on subsequent 15m highs/lows.
- Costs: 0.10% taker per side + 0.025% slippage per side  -> 0.25% round trip.
- Long only.
"""
import numpy as np, pandas as pd, os
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.join(os.path.dirname(__file__), "..")
COST = 2 * (0.0010 + 0.00025)          # round-trip cost, fraction
K_TP = 1.5                              # TP = 1.5 sigma, SL = 1 sigma
HOLDOUT_START = pd.Timestamp("2026-04-27", tz="UTC")   # pre-registered: last ~6 months untouched
DEV_START = pd.Timestamp("2021-01-01", tz="UTC")

BAR_MIN = {"15m": 15, "1h": 60, "4h": 240}
HOR_H = {"4h": 4, "12h": 12, "24h": 24}
GRID = [(b, h) for b in BAR_MIN for h in HOR_H if HOR_H[h] * 60 / BAR_MIN[b] >= 3]


DATA = os.environ.get("RT_DATA", "bitstamp")   # "bitstamp" (runs 1-11) or "binance" (runs 12+)


def load_15m():
    f = "binance_15m.parquet" if DATA == "binance" else "btc_15m.parquet"
    return pd.read_parquet(os.path.join(ROOT, "data", f))


def resample(b15, bar):
    if bar == "15m":
        return b15.copy()
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    extra = {c: ("sum" if c in ("trades", "taker_buy_base") else "last") for c in b15.columns if c not in agg}
    return b15.resample(bar.replace("m", "min")).agg({**agg, **extra}).dropna(subset=list(agg))


# ----------------------------------------------------------------- volatility
def sigma_horizon(b15, hours):
    """Trailing realized vol of 15m log returns (EWMA, ~3-day half-life window), scaled to horizon.
    Known at each 15m close (uses only past returns)."""
    r = np.log(b15.close).diff()
    v = (r ** 2).ewm(span=4 * 24 * 3, min_periods=4 * 24).mean()
    return np.sqrt(v * hours * 4)


# ----------------------------------------------------------------- labels
def triple_barrier(b15, hours, k_tp=K_TP):
    """For every 15m close t: enter at close, TP=+k*sig, SL=-sig (log space), time-out after `hours`.
    Uses 15m high/low path; if TP and SL both touched in the same 15m bar we assume SL first (conservative).
    Returns DataFrame: label (1=TP, -1=SL, 0=timeout), gross log return, exit time index offset, sigma."""
    n = int(hours * 4)
    c = b15.close.values; hi = np.log(b15.high.values); lo = np.log(b15.low.values); lc = np.log(c)
    sig = sigma_horizon(b15, hours).values
    N = len(c)
    # future windows: bars t+1 .. t+n
    H = sliding_window_view(hi[1:], n)          # row t -> hi[t+1 : t+1+n]
    L = sliding_window_view(lo[1:], n)
    C = sliding_window_view(lc[1:], n)
    m = H.shape[0]                              # = N - n
    base = lc[:m, None]; s = sig[:m, None]
    up = (H - base) >= k_tp * s
    dn = (L - base) <= -s
    BIG = n + 1
    t_up = np.where(up.any(1), up.argmax(1), BIG)
    t_dn = np.where(dn.any(1), dn.argmax(1), BIG)
    label = np.zeros(m, int)
    label[(t_dn <= t_up) & (t_dn < BIG)] = -1
    label[(t_up < t_dn)] = 1
    gross = np.where(label == 1, k_tp * sig[:m], np.where(label == -1, -sig[:m], C[:, -1] - lc[:m]))
    exit_off = np.where(label == 1, t_up + 1, np.where(label == -1, t_dn + 1, n))
    out = pd.DataFrame({"label": label, "gross": gross, "exit_off": exit_off, "sigma": sig[:m]},
                       index=b15.index[:m])
    return out.dropna()


# ----------------------------------------------------------------- features
def features(bars, bar_min):
    """Features on the decision bar series. Windows specified in hours, converted to bars."""
    f = pd.DataFrame(index=bars.index)
    nb = lambda hrs: max(int(round(hrs * 60 / bar_min)), 2)
    lc = np.log(bars.close); r = lc.diff()
    vol24 = r.rolling(nb(24)).std()
    # volatility family
    for hrs in (4, 24, 168):
        if nb(hrs) >= 3:
            f[f"rv_{hrs}h"] = r.rolling(nb(hrs)).std() * np.sqrt(60 / bar_min)  # per-hour scale
    f["rv_ratio_24_168"] = r.rolling(nb(24)).std() / r.rolling(nb(168)).std()
    f["range_24h"] = (np.log(bars.high.rolling(nb(24)).max()) - np.log(bars.low.rolling(nb(24)).min())) / (vol24 * np.sqrt(nb(24)))
    f["vol_of_vol"] = vol24.rolling(nb(168)).std() / vol24.rolling(nb(168)).mean()
    # overextension
    for hrs in (24, 168):
        ma = lc.rolling(nb(hrs)).mean()
        f[f"dist_ma_{hrs}h"] = (lc - ma) / (vol24 * np.sqrt(nb(hrs)))
    f["last_ret_z"] = r / vol24
    d = r.clip(lower=0).rolling(nb(24)).mean(); u = (-r.clip(upper=0)).rolling(nb(24)).mean()
    f["rsi_24h"] = d / (d + u)
    # trend
    for hrs in (4, 24, 168):
        f[f"ret_{hrs}h_z"] = (lc - lc.shift(nb(hrs))) / (vol24 * np.sqrt(nb(hrs)))
    f["ret_720h_z"] = (lc - lc.shift(nb(720))) / (vol24 * np.sqrt(nb(720)))
    # volume
    lv = np.log1p(bars.volume)
    f["vol_rel_24h"] = lv - lv.rolling(nb(24)).mean()
    f["vol_rel_168h"] = lv.rolling(nb(24)).mean() - lv.rolling(nb(168)).mean()
    upv = (bars.volume * (r > 0)).rolling(nb(24)).sum(); allv = bars.volume.rolling(nb(24)).sum()
    f["upvol_share_24h"] = upv / allv
    # time
    hr = bars.index.hour + bars.index.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hr / 24); f["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    f["weekend"] = (bars.index.dayofweek >= 5).astype(int)
    f["hrs_to_funding"] = (8 - (hr % 8)) % 8          # Binance perp funding at 00/08/16 UTC
    # ---- Binance-only families (prefix x_ : allowed to be NaN, LightGBM handles missing) ----
    if "taker_buy_base" in bars:                        # order flow
        tb = bars.taker_buy_base / bars.volume
        f["x_taker_buy_share_4h"] = tb.rolling(nb(4)).mean() - 0.5
        f["x_taker_buy_share_24h"] = tb.rolling(nb(24)).mean() - 0.5
        f["x_trades_rel_24h"] = np.log1p(bars.trades) - np.log1p(bars.trades).rolling(nb(168)).mean()
    if "funding" in bars:                               # positioning: funding
        fr = bars.funding
        f["x_funding"] = fr * 1e4
        f["x_funding_chg_24h"] = (fr - fr.shift(nb(24))) * 1e4
        f["x_funding_z_30d"] = (fr - fr.rolling(nb(720)).mean()) / fr.rolling(nb(720)).std()
    if "oi_value" in bars:                              # positioning: open interest & long/short
        oi = np.log(bars.oi_value)
        f["x_oi_chg_4h_z"] = (oi - oi.shift(nb(4))) / (vol24 * np.sqrt(nb(4)))
        f["x_oi_chg_24h_z"] = (oi - oi.shift(nb(24))) / (vol24 * np.sqrt(nb(24)))
        f["x_oi_vs_7d"] = oi - oi.rolling(nb(168)).mean()
        f["x_top_ls"] = np.log(bars.top_ls_ratio)
        f["x_global_ls_chg_24h"] = np.log(bars.global_ls_ratio) - np.log(bars.global_ls_ratio.shift(nb(24)))
        f["x_taker_ls_24h"] = np.log(bars.taker_ls_vol_ratio).rolling(nb(24)).mean()
        # crowding interaction (plan hypothesis): rising OI with rich funding
        f["x_crowding"] = f["x_oi_chg_24h_z"] * f["x_funding_z_30d"]
    if "eth_close" in bars:                             # cross-asset
        le = np.log(bars.eth_close); re = le.diff()
        f["x_eth_rel_24h"] = ((le - le.shift(nb(24))) - (lc - lc.shift(nb(24)))) / (vol24 * np.sqrt(nb(24)))
        f["x_eth_ret_4h_z"] = (le - le.shift(nb(4))) / (re.rolling(nb(24)).std() * np.sqrt(nb(4)))
        f["x_btc_eth_corr_7d"] = r.rolling(nb(168)).corr(re)
    return f.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------- primary rule
def primary_rule(f):
    """Trend pullback: medium-term uptrend (above 7d MA) but stretched down vs 24h MA."""
    return (f["dist_ma_168h"] > 0) & (f["dist_ma_24h"] < -0.5)


# ----------------------------------------------------------------- dataset for a config
def build(b15, bar, hor):
    bars = resample(b15, bar)
    hours = HOR_H[hor]
    lab = triple_barrier(b15, hours)
    # decision at bar close: resample index = bar open time; close time = open + bar length - 15m (last 15m bar open)
    close_15m_idx = bars.index + pd.Timedelta(minutes=BAR_MIN[bar] - 15)
    f = features(bars, BAR_MIN[bar])
    f.index = close_15m_idx            # align features to the 15m bar whose close == decision time
    df = f.join(lab, how="inner")
    df["net"] = df["gross"] - COST
    df["exit_time"] = df.index + pd.to_timedelta(df["exit_off"] * 15, unit="m")
    df["setup"] = primary_rule(df)
    core = [c for c in df.columns if not c.startswith("x_")]
    df = df[df.index >= DEV_START].dropna(subset=core)
    return df


FEATS_EXCLUDE = {"label", "gross", "exit_off", "sigma", "net", "exit_time", "setup"}


def feat_cols(df):
    return [c for c in df.columns if c not in FEATS_EXCLUDE]


# ----------------------------------------------------------------- walk-forward folds
def folds(df, first_test="2022-07-01", step_months=6, end=HOLDOUT_START):
    """Expanding-window walk-forward. Test blocks of 6 months until the holdout start."""
    starts = pd.date_range(pd.Timestamp(first_test, tz="UTC"), end, freq=f"{step_months}MS")
    out = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else end
        if s >= e:
            continue
        out.append((s, e))
    return out


def split(df, s, e, embargo_h):
    """Purge: training events whose label ends after test start are dropped. Embargo after test not needed
    (expanding window only trains on the past)."""
    tr = df[(df.index < s) & (df["exit_time"] < s - pd.Timedelta(hours=embargo_h))]
    te = df[(df.index >= s) & (df.index < e)]
    return tr, te


def uniqueness_weights(tr):
    """Average uniqueness of each label over its lifespan (Lopez de Prado), approximated on 15m grid."""
    st = tr.index.values.astype("datetime64[m]").astype(np.int64) // 15
    en = tr["exit_time"].values.astype("datetime64[m]").astype(np.int64) // 15
    lo = st.min(); T = en.max() - lo + 2
    cnt = np.zeros(T)
    np.add.at(cnt, st - lo, 1); np.add.at(cnt, en - lo + 1, -1)
    conc = np.cumsum(cnt); conc[conc == 0] = 1
    inv = np.concatenate([[0], np.cumsum(1 / conc)])
    w = (inv[en - lo + 1] - inv[st - lo]) / (en - st + 1)
    return w / w.mean()


# ----------------------------------------------------------------- backtest
def sequential(te, take):
    """Trade selected events one at a time (no overlapping positions). Returns list of trade net log returns
    with entry times."""
    sel = te[take]
    trades = []; busy_until = None
    for t, row in sel.iterrows():
        if busy_until is not None and t < busy_until:
            continue
        trades.append((t, row["net"])); busy_until = row["exit_time"]
    return pd.Series([x[1] for x in trades], index=[x[0] for x in trades], dtype=float)


def perf(tr_series, start, end):
    """Trade-level and daily-equity metrics (full notional per trade)."""
    if len(tr_series) == 0:
        return dict(n=0, ev_bps=np.nan, hit=np.nan, total=0.0, sharpe=np.nan, maxdd=0.0)
    days = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
    daily = tr_series.groupby(tr_series.index.normalize()).sum().reindex(days, fill_value=0.0)
    eq = daily.cumsum()
    sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else np.nan
    return dict(n=len(tr_series), ev_bps=tr_series.mean() * 1e4, hit=(tr_series > 0).mean(),
                total=tr_series.sum(), sharpe=sharpe, maxdd=(eq - eq.cummax()).min())
