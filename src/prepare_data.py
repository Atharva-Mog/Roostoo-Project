"""Build 15m OHLCV bars (2021-01-01 onward) from Bitstamp BTC/USD 1-minute data.
Source: github.com/ff137/bitstamp-btcusd-minute-data (Binance endpoints were unreachable from our environment)."""
import pandas as pd, sys, os
D = os.path.join(os.path.dirname(__file__), "..", "data")
h = pd.read_csv(os.path.join(D, "hist.csv.gz"))
l = pd.read_csv(os.path.join(D, "latest.csv"))
m = pd.concat([h, l]).drop_duplicates("timestamp").sort_values("timestamp")
m = m[m.timestamp >= pd.Timestamp("2020-12-01", tz="UTC").timestamp()]
m.index = pd.to_datetime(m.timestamp, unit="s", utc=True)
b = m.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
b = b.dropna()
b = b.iloc[:-1]  # drop last (possibly incomplete) bar
b.to_parquet(os.path.join(D, "btc_15m.parquet"))
print(b.shape, b.index[0], b.index[-1]); print(b.tail(2))
