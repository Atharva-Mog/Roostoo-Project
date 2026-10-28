"""Merge Binance downloads (from get_binance.py) into one 15m parquet with leak-free alignment.
- Spot klines: open time in ms (pre-2025) or us (2025+) -> normalized.
- Funding: rate known at calc_time; forward-filled onto 15m bars whose CLOSE is after calc_time.
- Metrics (5-min OI / long-short ratios): create_time stamps; we lag by one 5-min period before use, then take the
  last value known at each 15m bar close."""
import pandas as pd, numpy as np, os, sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Roostoo/data/binance"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "binance_15m.parquet")
K = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
def kl(path):
    d = pd.read_csv(path, header=None, names=K)
    t = d.open_time.astype("int64"); t = np.where(t > 1e14, t // 1000, t)
    d.index = pd.to_datetime(t, unit="ms", utc=True); return d[~d.index.duplicated()].sort_index()
b = kl(f"{SRC}/btc_spot_15m.csv"); e = kl(f"{SRC}/eth_spot_15m.csv")
df = b[["open", "high", "low", "close", "volume", "trades", "taker_buy_base"]].copy()
df["eth_close"] = e.close.reindex(df.index)
bar_close = (df.index + pd.Timedelta(minutes=15)).as_unit("ns")
f = pd.read_csv(f"{SRC}/btc_funding.csv"); f.index = pd.DatetimeIndex(pd.to_datetime(f.calc_time, unit="ms", utc=True)).as_unit("ns"); f = f.sort_index()
df["funding"] = pd.merge_asof(pd.DataFrame({"t": bar_close}), f[["last_funding_rate"]].rename_axis("t").reset_index(), on="t").last_funding_rate.values
m = pd.read_csv(f"{SRC}/btc_metrics.csv"); m.index = pd.DatetimeIndex(pd.to_datetime(m.create_time, utc=True) + pd.Timedelta(minutes=5)).as_unit("ns")
m = m[~m.index.duplicated()].sort_index()
mc = ["sum_open_interest_value", "count_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]
mm = pd.merge_asof(pd.DataFrame({"t": bar_close}), m[mc].rename_axis("t").reset_index(), on="t", tolerance=pd.Timedelta("1h"))
for c, n in zip(mc, ["oi_value", "top_ls_ratio", "global_ls_ratio", "taker_ls_vol_ratio"]): df[n] = mm[c].values
df.to_parquet(OUT); print(df.shape, df.index[0], df.index[-1]); print(df.isna().mean().round(3).to_dict()); print(df.tail(2).T)
