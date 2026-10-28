"""Download Binance public historical data into data/binance/ (standard library only, no pip needed).

Run from this folder:   python get_binance.py
Safe to re-run: files already downloaded are skipped. Takes a few minutes.
"""
import datetime as dt
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://data.binance.vision/data"
OUT = Path(__file__).resolve().parent / "data" / "binance"
START = dt.date(2021, 1, 1)
TODAY = dt.date.today()


def months(start, end):
    y, m = start.year, start.month
    while (y, m) < (end.year, end.month):          # complete months only
        yield f"{y}-{m:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def days(start, end):
    d = start
    while d < end:
        yield d.isoformat(); d += dt.timedelta(days=1)


jobs = []  # (url, local path)
first_of_month = TODAY.replace(day=1)
for sym in ("BTCUSDT", "ETHUSDT"):
    for ym in months(START, TODAY):
        jobs.append((f"{BASE}/spot/monthly/klines/{sym}/15m/{sym}-15m-{ym}.zip", f"spot_15m/{sym}/{sym}-15m-{ym}.zip"))
    for d in days(first_of_month, TODAY):
        jobs.append((f"{BASE}/spot/daily/klines/{sym}/15m/{sym}-15m-{d}.zip", f"spot_15m/{sym}/{sym}-15m-{d}.zip"))
for ym in months(START, TODAY):
    jobs.append((f"{BASE}/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{ym}.zip", f"funding/BTCUSDT-fundingRate-{ym}.zip"))
for d in days(dt.date(2021, 12, 1), TODAY):
    jobs.append((f"{BASE}/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{d}.zip", f"metrics/BTCUSDT-metrics-{d}.zip"))


def fetch(job):
    url, rel = job
    p = OUT / rel
    if p.exists() and p.stat().st_size > 0:
        return "skip"
    p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                p.write_bytes(r.read())
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "missing"
        except Exception:
            pass
    return "fail"


if __name__ == "__main__":
    print(f"Downloading {len(jobs)} files into {OUT} ...")
    counts = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, res in enumerate(ex.map(fetch, jobs), 1):
            counts[res] = counts.get(res, 0) + 1
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {counts}")
    print("Done.", counts)
    if counts.get("fail"):
        print("Some downloads failed (network hiccup). Just run the script again; it resumes.")
