# BTC Trade Coach: Predictive ML Trading Assistant

Submission for the Roostoo ML intern assessment by **Atharva Moghe** (IIT Madras).

A machine-learning agent that coaches a new BTC/USDT trader on **whether a trade is worth taking after their own fees**, how big it should be, where to exit, and when to sit out. It does not try to call the next price move.

- **Report (3 pages):** [`report/Atharva_report_pdf.pdf`](report/Atharva_report_pdf.pdf) (LaTeX source in `report/`)
- **Interactive mock-up of the trader screen:** [`mockup/btc_trade_coach.html`](mockup/btc_trade_coach.html). Download it and open it in a browser; it runs on the agent's real 6-month replay.
- **Every experiment, including the failures:** [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) (19 runs)
- **How the decision agent works:** [`AGENT.md`](AGENT.md)

## Key results

| Finding | Number |
| --- | --- |
| Direction of the next 24h (take-profit vs stop) | AUC about 0.55 (weak) |
| Ranking of trades vs shuffled-label retrains | Beat 97% of them; held out of sample |
| Average edge per trade vs beginner fees | 0.09% vs 0.25% (breakeven about 0.10%) |
| Next-24h volatility forecast vs HAR-RV benchmark | 7% lower QLIKE over 8 periods, 19% lower on the final 6 months |
| Final 6 months: coach vs acting on every signal (beginner fees) | -9.1% vs -61% (buy and hold: +2%) |

The coach mostly says "don't trade", and that is where its value is. None of the configurations beat buy-and-hold on the final 6 months; the report and log say so plainly.

## Repository layout

```
get_binance.py        download Binance spot, funding and open-interest data (standard library only)
src/
  prepare_binance.py  merge the downloads into one leak-free 15-minute dataset
  pipeline.py         labels (triple barrier), features, walk-forward folds, backtest helpers
  experiment.py       calibrated LightGBM / logistic models, expected-value gate, benchmarks
  perp_costs.py       futures execution model: limit-order fills, maker/taker fees, funding
  decision_agent.py   the 4-hour decision loop that produces the trader cards
  vol_forecast.py     next-24h volatility forecast vs yesterday / EWMA / HAR-RV
  council.py          self-adjusting council of strategies (multiplicative weights)
  shorts.py, lit_strategies.py, open_dip.py, run*.py   individual experiments from the log
results/              CSV/JSON outputs of every run (v1 ... v19), figures
report/               LaTeX report, figures, compiled PDF
mockup/               interactive trader screen (single HTML file)
```

## Reproduce

```bash
pip install -r requirements.txt
python get_binance.py                                         # ~2,000 small files into data/binance/
python src/prepare_binance.py data/binance                    # builds data/binance_15m.parquet
RT_DATA=binance python src/run12_binance.py                   # main grid with futures costs
RT_DATA=binance python src/vol_forecast.py                    # volatility forecast vs baselines
RT_DATA=binance python src/decision_agent.py --fees 25 --tag beginner_spot   # agent replay
```

On Windows, set the variable first with `set RT_DATA=binance` and drop it from the command. `prepare_binance.py` expects the merged CSVs (`btc_spot_15m.csv`, `eth_spot_15m.csv`, `btc_funding.csv`, `btc_metrics.csv`) in the folder you pass it. Runs 1 to 11 in the log used Bitstamp data (`src/prepare_data.py`) because Binance was unreachable at the time; all later runs use Binance.

## Method in brief

- **Target:** probability that a long trade hits its take-profit (+1.5x the typical 24h move) before its stop (-1x), or neither, within 24 hours.
- **Model:** shallow three-class LightGBM with label-uniqueness weights and isotonic calibration; logistic regression as the simple reference.
- **Validation:** expanding walk-forward over 8 six-month periods with purged labels, a held-out final 6 months, real fees and slippage, a random-filter benchmark with the same number of trades, and shuffled-label retrains.

## Note on AI assistance

Claude (Anthropic) was used to help write parts of the codebase and to draft and edit the report. The research direction, design decisions and final judgement on every result are mine, and I take full responsibility for the content.

Contact: cs23b096@smail.iitm.ac.in · mogheatharva16@gmail.com
