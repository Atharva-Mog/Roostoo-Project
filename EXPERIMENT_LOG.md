# Experiment log: BTC triple-barrier trade filter

Every run is listed here in order, including failed ones. All numbers come from walk-forward test folds (8 folds of 6 months, from 2022-07 to 2026-04) unless marked HOLDOUT. Per-fold CSVs are in `results/<tag>/`.

## Fixed setup (pre-registered, unchanged across runs)

| Item | Value |
| --- | --- |
| Data | Bitstamp BTC/USD, 1-minute bars aggregated to 15m; 2021-01-01 to 2026-10-27 (207k bars). Binance endpoints were unreachable from our environment, so we used [ff137/bitstamp-btcusd-minute-data](https://github.com/ff137/bitstamp-btcusd-minute-data). No funding or open-interest data: the only free source found had 578 sparse rows, so we dropped it. |
| Label | Long triple barrier: TP = +1.5σ, SL = −1σ, time-out at horizon. σ = EWMA realized vol of 15m returns scaled to the horizon. Barriers are checked on 15m highs and lows. If both are hit in the same 15m bar, the stop counts as first (conservative). |
| Costs | 0.10% taker + 0.025% slippage per side = **25 bps round trip** |
| Grid | Bars {15m, 1h, 4h} × horizons {4h, 12h, 24h}, keeping configs with at least 3 bars per horizon → 8 configs |
| Validation | Expanding walk-forward; training labels purged if they end within one horizon of the test start; inner 80/20 time split for isotonic calibration (also purged) |
| Holdout | 2026-04-27 to 2026-10-26, **used once** (run 8) |
| Trading | One position at a time, full notional, net log returns; Sharpe from daily P&L × √365 |
| Benchmarks | Buy and hold · take every bar · logistic regression filter · random filter at the same take rate · LightGBM filter |

## Runs

### Run 1 (`v1_setups`): meta-labeling on a trend-pullback rule
Primary rule: price above 7-day MA and more than 0.5 normalized units below the 24h MA. LightGBM binary P(TP) → EV gate.
- Setups are rare: 1,401 at 1h bars and 422 at 4h bars over 5.8 years. At 4h bars no fold had 500 training rows, so those configs couldn't run.
- AUC 0.47–0.56; the model took 8–36 trades in total per config.
- Setup EV before ML was −20 to −37 bps, **worse than taking every bar** (−26). The rule itself has no edge on this data.
- **Decision:** drop the primary rule and filter every bar (run 2). This departs from the plan; the reason is sample size.

### Run 2 (`v2_all`): filter every bar, binary P(TP)
- AUC rose to 0.55–0.67, but net EV of taken trades stayed −21 to −33 bps in 6 of 8 configs, about the same as the random filter.
- **Diagnosis:** AUC for "any barrier hit" is the same as AUC for TP (0.658 vs 0.661 at 1h/4h). AUC for TP vs SL among resolved trades is only **0.55**. So the model mostly predicts *volatility* (whether something happens), not *direction*.
- **Bug found:** the EV formula used a fixed average outcome for non-TP trades, so in high-volatility periods it raised P(TP) without raising P(SL), which inflated EV exactly when the stop is also more likely. Fixed in run 3.

### Run 3 (`v3_multi`): three-class model (SL / time-out / TP), EV from all three calibrated probabilities  **← main model**
EV = (1.5·P(TP) − P(SL) + P(TO)·E[timeout return in σ]) · σ − cost; trade if EV > 0.

| Config | AUC | Trades | Net EV/trade (bps) | Net Sharpe | Random-filter Sharpe | Folds > 0 |
| --- | --- | --- | --- | --- | --- | --- |
| 15m / 4h | 0.68 | 485 | −28.5 | −2.69 | −5.05 | 1/8 |
| 15m / 12h | 0.63 | 832 | −9.7 | −1.59 | −4.18 | 2/8 |
| 15m / 24h | 0.59 | 763 | +13.2 | −0.79 | −1.97 | 2/8 |
| 1h / 4h | 0.67 | 402 | −20.1 | −1.87 | −3.35 | 0/8 |
| 1h / 12h | 0.63 | 700 | −23.7 | −1.93 | −2.97 | 0/8 |
| 1h / 24h | 0.58 | 703 | −10.8 | −0.85 | −1.78 | 2/8 |
| 4h / 12h | 0.63 | 558 | −15.1 | −1.55 | −2.30 | 2/8 |
| 4h / 24h | 0.59 | 610 | −19.6 | −1.10 | −1.47 | 1/8 |

- The LightGBM filter beats the random filter on Sharpe in **all 8 configs**, so selection skill is real. It still loses money net of fees.
- Spearman(predicted EV, realized return) is only 0.01–0.04.
- Horizon pattern: 24h configs are the least bad at every bar size; 4h configs are the worst. This matches the fee argument in the plan, since short horizons pay the same 25 bps on smaller moves. See `results/figures/grid_heatmap.png`.

### Run 4 (`v4_margin25`): stricter gate, trade only if EV > +25 bps
- Trades fell 3–25×, and EV per trade **did not improve** consistently (1h/24h: −10.8 → −21.7 bps).
- **Conclusion:** the model's high-EV tail is not reliable. Calibration is reasonable on average but noisy at the extremes, so a tighter threshold doesn't buy precision. Reverted to EV > 0.

### Diagnostic (`v3_multi/ev_deciles_gross_bps.csv`): gross return by predicted-EV decile
- Gross (pre-fee) returns rise with predicted EV in most configs. At 1h/24h: decile 1 is −2.9 bps, deciles 8–10 are +13 to +24 bps.
- **The best deciles still earn less than the 25 bps round-trip cost.** That's the core finding: a weak, real edge that is smaller than retail fees.
- **Config selected on dev data only: 1h bars / 24h horizon.** It has the highest rank correlation (0.044), the 24h horizon is best at every bar size (a smooth surface, not one lucky cell), and its fold sample sizes are adequate.

### Run 5 (`v5_rfe_1h_24h`): RFECV inside each fold
Correlation clustering (|ρ| > 0.8), then backward elimination by permutation importance on a purged inner validation split, dropping 20% per round and keeping the set with the best inner log-loss.
- Kept 6–18 of 23 features per fold. Net EV −18.1 bps vs −10.8 with all features; 2/8 folds positive. **RFE did not help**, so we kept all features.
- Feature survival across folds (`feature_survival.csv`):
  - **Always or almost always kept:** weekend (100%), weekly volume trend (100%), 24h and 7d realized vol, 24h volume, 24h MA distance, hour of day (88%).
  - **Never kept:** 24h and 7d return momentum, RSI (0%). Partly because correlation pruning folds them into the MA-distance features, but either way **the momentum hypothesis found no independent support**. Volatility, volume and time effects carry the signal.
- Per-fold results show **alpha decay**: positive in 2022H2 and 2023H1, negative in every fold since 2023H2.

### Run 6 (`null_1h_24h.csv`): shuffled-label null, 30 full retrains
| Statistic | Real model | Null mean (sd) | p-value |
| --- | --- | --- | --- |
| Spearman(EV, realized net) | 0.044 | −0.015 (0.010) | **0.03** |
| Net EV of taken trades (bps) | −10.8 | −19.9 (14.6) | 0.29 |

- The pipeline does **not** leak: the null centres near zero.
- Ranking skill is statistically significant. Trading profit is not distinguishable from chance.

### Run 7 (`cost_sensitivity_1h_24h.csv`): what fee level would make it work? (dev folds)
| Round trip (bps) | Trades | Net EV/trade (bps) | Net Sharpe |
| --- | --- | --- | --- |
| 25 (retail taker) | 703 | −10.8 | −0.65 |
| 15 | 992 | −3.2 | −0.25 |
| 10 | 1,152 | +1.2 | +0.10 |
| 5 | 1,243 | +4.7 | +0.41 |
| 0 | 1,348 | +8.8 | +0.83 |

Breakeven is about **10 bps round trip**. That's maker-fee or VIP territory, not a beginner paying taker fees.

### Run 8 (`holdout_1h_24h.csv`): HOLDOUT, run once, 2026-04-27 to 2026-10-26
| Strategy | Trades | Net EV/trade (bps) | Hit rate | Total net (log) | Sharpe | Max DD |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM filter | 74 | −21.4 | 38% | −15.8% | −1.52 | −19.3% |
| Logistic filter | 58 | −30.9 | 43% | −17.9% | −1.87 | −26.4% |
| Random, same rate | 163 | −30.0 | 42% | −48.9% | −3.30 | −50.2% |
| Take every bar | 241 | −27.3 | 45% | −65.7% | −3.56 | −65.2% |
| Buy and hold | – | – | – | +1.7% | – | – |

Spearman(EV, realized) on the holdout = **0.11**, so ranking skill holds out of sample and is even stronger. But the filter still loses after fees, and buy-and-hold beat every active strategy. Reported as-is; no re-tuning after seeing the holdout.


## Runs 9–11: chasing positive P&L (added after the first holdout; holdout below is NOT clean)

### Run 9 (`v9_perp/`): realistic perp execution costs
BTC USDT-M perpetual, Binance VIP0 fees: post-only limit entry 2 bps below the decision close (maker 0.02%), **filled only if price trades through within 1h**. TP is a resting limit (maker); SL and time-out exits are taker 0.05% + 2.5 bps slippage. Funding is 1 bp per 8h held. Round trip ≈ 4–10 bps instead of 25. The EV gate uses outcome-dependent expected cost.

| Config | Trades | Net EV/trade (bps) | Total net (log) | Folds > 0 |
| --- | --- | --- | --- | --- |
| 15m / 24h | 1,148 | **+0.2** | +2.8% | 4/8 |
| 1h / 24h | 1,140 | −2.9 | −32.8% | 4/8 |
| 15m / 12h | 1,961 | −4.5 | −87.4% | 2/8 |
| 4h / 24h | 966 | −7.3 | −70.5% | 2/8 |
| 4h-horizon configs | 2,700–3,300 | −7.8 to −9.5 | large losses | 0/8 |

- Breakeven is reached at 15m/24h, but not a profit.
- **Adverse selection is the new cost:** limit orders fill mostly when price keeps falling. Even "take every filled order" loses about 9–11 bps despite cheaper fees.

### Run 10 (`v10_gates.csv`): EV margin {0, 5, 10, 20 bps} × high-vol regime gate (7d realized vol above the training-fold median), 24h configs only. 24 variants.
- Without the regime gate, nothing is meaningfully positive (best +2.0 bps).
- **1h/24h + regime gate is positive at every margin** (+11.8 to +14.5 bps/trade; total +21% to +36% over 3.8 years; Sharpe 0.28–0.42). Other bar sizes are mixed.

### Run 11 (`v11_check.csv`): is run 10's best result real?
| Check | Result |
| --- | --- |
| Model + regime, dev | 258 trades, +13.0 bps/trade, total +33.6%, Sharpe 0.39, max DD −29.9% |
| Regime gate alone (every filled trade), dev | 380 trades, +3.2 bps, total +12.1% |
| Random filter inside regime (same rate, 50 draws) | +6.6 bps, total +21.7% |
| Model vs random-in-regime | **p = 0.28 (not significant)** |
| Block-bootstrap 95% CI, EV/trade | **[−16.9, +43.4] bps**; P(EV > 0) = 0.78 |
| Holdout* model + regime | 19 trades, −22.7 bps, total −4.3% |
| Holdout* regime gate alone | 26 trades, +10.4 bps, total +2.7% (buy and hold +1.7%) |

\* The holdout was already seen in run 8, and runs 9–10 were designed after it, so these holdout numbers are informative but not a clean test.

**Conclusion of runs 9–11:** with realistic perp execution, the high-volatility regime is the only place P&L is positive (on dev, and for the regime gate alone on the holdout). The ML filter's extra gain inside that regime is **not statistically distinguishable from luck**, and 24 variants were tried, so a deflated Sharpe would be lower still. No positive-P&L claim survives our own benchmarking standard yet.

## Runs 12–14: Binance data (downloaded with `get_binance.py`), adding positioning and order-flow features

Data: Binance spot BTCUSDT and ETHUSDT 15m (2021-01 to 2026-10-26), USDT-M funding (8h, aligned at publish time), and 5-min open-interest and long/short metrics from 2021-12 (lagged one period, ~16% missing before Dec 2021, left as NaN for LightGBM). Binance and Bitstamp closes differ by a median 4 bps, and 15m returns correlate at 0.99, so runs 1–11 were not a data artifact.

16 new features (prefix `x_`): taker-buy share 4h and 24h, trade count vs 7d; funding level, 24h change, 30d z-score; OI change 4h and 24h (σ-normalized), OI vs 7d; top-trader L/S, global L/S 24h change, taker L/S vol ratio; crowding = OI change × funding z; ETH vs BTC 24h, ETH 4h return, BTC–ETH 7d correlation. All use run-9 perp execution costs.

### Run 12 (`v12_binance/grid_summary.csv`): base (23 OHLCV features) vs full (39 features), all 8 configs, dev folds
| Config | Base EV (bps) | Full EV (bps) | Full total | Full folds > 0 | Spearman base → full |
| --- | --- | --- | --- | --- | --- |
| 4h / 24h | +0.9 | **+3.8** | **+35.0%** | **6/8** | 0.040 → 0.061 |
| 1h / 24h | −3.8 | −0.2 | −2.0% | 4/8 | 0.040 → 0.051 |
| 15m / 24h | −1.0 | −0.1 | −1.3% | 5/8 | 0.030 → 0.059 |
| 4h / 12h | −8.8 | −4.7 | −58.9% | 2/8 | 0.031 → 0.034 |
| 1h / 12h | −6.2 | −4.1 | −63.9% | 4/8 | 0.027 → 0.031 |
| 15m / 12h | −3.2 | −5.5 | −97.6% | 2/8 | 0.034 → 0.034 |
| 1h / 4h | −9.7 | −7.9 | large loss | 1/8 | 0.022 → 0.033 |
| 15m / 4h | −7.5 | −7.2 | large loss | 1/8 | 0.027 → 0.028 |

- The new features improve EV in 7 of 8 configs, and ranking at 24h rises by about 25–100%. Positioning, order-flow and cross-asset data carry real information.
- Direction AUC (TP vs SL among resolved trades) is still only 0.53–0.57.

### Run 13 (`v13_*`): robustness of the best variant, 4h / 24h full (picked from 16)
| Check | Result |
| --- | --- |
| Dev P&L | 910 trades, +3.85 bps/trade, total +35.0%, Sharpe 0.29, max DD −38.7% |
| vs random filter (filled orders, same rate, 50 draws) | random −79.7% → **p = 0.02** |
| Shuffled-label null (30 retrains) | Spearman p = 0.03; total P&L p = 0.03 |
| Block-bootstrap 95% CI, EV/trade | [−6.4, +14.0] bps, P(EV > 0) = 0.77 |
| By year (bps/trade) | 2022 −12.7 · 2023 +12.6 · 2024 +5.4 · 2025 −2.7 · 2026 +1.4 |
| Top features (gain) | rv_168h, rv_24h, dist_ma_168h, vol_rel_168h, vol_of_vol, **x_btc_eth_corr_7d, x_taker_buy_share_24h, x_funding_z_30d**, x_eth_rel_24h |

This was the first result to beat its random benchmark and the null, but the bootstrap CI includes 0 and it was selected from 16 variants (Bonferroni would not pass).

### Run 14 (`v14_holdout_binance.csv`): holdout 2026-04-27 → 2026-10-26, run once (semi-clean*)
| Config | Trades | EV (bps) | Total | Sharpe | Random (same rate) | p vs random | Every filled order |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **4h / 24h (candidate)** | 102 | −28.0 | **−28.6%** | −2.54 | −24.4% | 0.64 | −39.7% |
| 1h / 24h | 122 | −20.8 | −25.4% | −2.21 | −40.4% | 0.05 | −45.5% |
| 15m / 24h | 159 | −18.0 | −28.6% | −1.99 | −38.0% | 0.05 | −36.3% |
| Buy and hold | – | – | +2.0% | – | – | – | – |

\* The window was seen in runs 8 and 11 (Bitstamp), and the perp-cost model was designed afterwards, but the Binance features and run-12/13 selection never touched it.

**The candidate failed out of sample.** The neighbouring configs beat their random filter (p ≈ 0.05), so the selection skill is still there, but every variant lost money because "go long" setups had a poor six months (−36% to −46% for every filled order). The dev edge was concentrated in 2023–2024 and has decayed since 2025, consistent with the alpha-decay pattern in run 5.

## Run 15 (`v15_literature/`): published strategies, parameters as published, Binance 2021-01 → 2026-10
The source papers' samples end by Feb 2022, so **2022-07 onward is genuinely out-of-sample** for every rule. Costs: spot 12.5 bps/side, perp 7.5 bps/side, charged on each position change. Returns are summed daily log returns.

| Strategy (source) | Post-paper 2022-07→2026-04: ann. return / Sharpe / max DD | Last 6 months: total / Sharpe | Verdict |
| --- | --- | --- | --- |
| Buy & hold BTC | 35.9% / 0.74 / −68% | +1.9% / 0.10 | benchmark |
| **S4 time-series momentum, 28d** (long if 28-day return > 0, else flat) | 28.7% / 0.82 / **−44%** | **+10.6% / 0.79** | **Works**: lookbacks 14–60d are all positive post-paper (15–34%/yr); 5 of 6 are positive in the last 6m (`tsmom_sensitivity.csv`). About B&H Sharpe with ~55% exposure and much smaller drawdowns. |
| **S3 funding carry** (long spot + short perp, collect funding) | 7.1% / very high* / −0.4% | +2.6% (5.2% ann.) | **Works, steadily**: positive in every period. *Sharpe is overstated (see caveats). |
| S2 MAX/MIN 10/20/50d ([Padyšák & Vojtko](https://quantpedia.com/?p=17895)) | −0.1% to 10.5% / ≤ 0.45 | −4% to −7% | Weak; the reported 98%/yr did not persist. |
| S1 long 22:00–24:00 UTC ([Quantpedia](https://quantpedia.com/strategies/intraday-seasonality-in-bitcoin)) | 7.6%/yr **before costs**; −47% to −84%/yr after | −33% to −51% | **Fee trap**: ~2 bps/day gross vs 15–25 bps/day costs. |

Caveats:
- Funding carry ignores perp–spot basis mark-to-market, margin needs (capital is split across two legs, so return on capital is roughly half: ~3.5–5.5%/yr post-paper) and exchange/counterparty risk. It is a carry yield, not alpha.
- TSMOM's edge over buy-and-hold is mainly drawdown control (−44% vs −68%) and side-stepping down-trends; in strong bull years it lags.
- Three lookbacks were reported for S4 and S2 to avoid picking. The 14–60d sweep shows a smooth, all-positive surface.

**Takeaway:** the strategies that made money out of sample are slow and structural (4-week trend; funding carry), not the fast intraday ML signals of runs 1–14. That matches the fee argument: the profitable rules trade 20–50 times a year, not daily.

## Run 16 (`v16*_council/`, `src/council.py`): adaptive council of experts with self-adjusting weights (Hedge)
Experts' daily net P&L is fully out-of-sample: the ML filter uses walk-forward through today (P&L booked on the exit day); trend = 4-week momentum; carry = funding carry at half yield (two-leg capital). Weights follow w ∝ exp(η · discounted cumulative reward) and use only P&L resolved before each day; rebalancing costs 7.5 bps per unit of weight moved. **η and γ were chosen on 2022-07 → 2024-12 only (best Sharpe); the test is 2025-01 → 2026-10.** All grid points are in `hedge_grid.csv`.

| Variant | Selected | What the weights did | Test 2025-01 → now (ann. / Sharpe / max DD / total) |
| --- | --- | --- | --- |
| 16a: reward = raw daily return; experts ml/trend/carry/cash | η = 50, γ = 1 | Collapsed onto **trend** by mid-2023 | −2.6% / −0.10 / −28% / −4.8% |
| 16b: reward = return ÷ trailing 60d vol (lagged) | η = 0.5, γ = 1 | Collapsed onto **carry** within weeks | **+2.2% / very high / −0.2% / +4.0%** |
| 16c: directional experts ml/trend/hold/flat, idle weight earns carry | **η = 0** (no adaptation won on selection) | Stayed at equal weights | −13.7% / −0.67 / −39% / −25.0% |
| Benchmarks: buy & hold | – | – | −8.5% / −0.20 / −75% / −15.5% |
| Benchmarks: equal weights (16a experts) | – | – | −11.6% / −1.00 / −26% / −21.1% |
| Benchmarks: ML expert alone | – | – | −46% / −1.55 / −90% / −83.8% |

Findings:
- **Hedge behaves as theory says.** It converges to the single best expert, and fast. With experts this different (a near-riskless carry yield vs ~35%-vol directional bets), "self-adjusting weights" turns into "pick one expert", not a smooth reallocation. Which expert it picks depends on the reward definition (raw → trend; risk-adjusted → carry), so that choice is effectively the real decision.
- **16b is the only positive test result** (+4.0% while buy-and-hold lost 15.5%), and it was selected without seeing the test. But it is simply the carry trade. The adaptive layer added no skill beyond "notice early that carry is the most reliable thing".
- **16c, comparing like with like, found no benefit from adapting** on the selection period, and equal weights lost 25% in the test.
- **The ML expert was dropped by every variant** within months of starting. That matches its weak, decaying edge (runs 12–14).
- Three designs were tried. The test period was not used to choose among η values, but choosing among the three designs is itself a mild multiple-comparison risk. Figure: `results/figures/council.png`.

## Run 17 (`v17_open_dip/`, `src/open_dip.py`): "buy when the day starts lower" (session-open dip / auction idea)
Binance 2022-07 → 2026-10, spot fees 25 bps round trip. All 12 variants are reported; none were selected.
- **Hour-of-day profile:** no reliable "low hour". Hourly means are ±3 bps, and only 22:00 UTC reaches |t| > 2. That is 1 of 24 hours, which is what chance alone would produce, and the 22–24h rule already failed after fees in run 15.
- **Open-dip** (first 2/4/8 hours of the UTC day down more than k × typical, buy then and sell at day end): **all 9 variants lose 10–38 bps per trade** after fees. Baseline "buy every day" loses 16 bps/day net (gross +9).
- **Auction / fair-value** (day opens below yesterday's VWAP by more than k × daily vol, sell when price returns to VWAP or at day end): **wins 54–65% of the time but still loses 10–21 bps per trade**. Many small wins are wiped out by a few large losses when price keeps falling. Best variant (k = 0.5): −9.8 bps, 213 trades.
- **Conclusion:** BTC has no opening auction (it trades 24/7), and "lower at the start of the day" carries no edge after fees. Buying below fair value does revert more often than not, but the average payoff is negative. This is the same pattern as the limit-order adverse selection in run 9.

## Run 18 (`v18_shorts/`, `src/shorts.py`): adding a short side (perpetual futures, 1×)
This mirrors the long setup: the short target is a 1.5σ fall before a 1σ rise. Sell-limit entries count only if price trades through within 1h; take-profits are maker, stops and time-outs are taker + slippage. Shorts receive funding while holding. Same model, features and folds as the long side (1h bars / 24h horizon). "Both" takes whichever side has the higher positive EV, one position at a time.

| Period | Long | Short | Both | Random (both sides, same counts) | BTC hold |
| --- | --- | --- | --- | --- | --- |
| Dev total 2022-07 → 2026-04 (8 folds) | −2.0% | **−93.1%** | −20.9% (3/8 folds > 0) | −155.1% | +137.9% |
| 2022 H2 (falling market) | −15.1% | **+12.4%** | −5.3% | +5.1% | −17.6% |
| 2025 H1 | −25.1% | **+6.4%** | −20.5% | −23.1% | +13.5% |
| 2026 H1 (falling market) | −15.5% | **+12.3%** | **+51.8%** (p = 0.03 vs random) | −16.2% | −10.9% |
| Last 6 months* | −25.4% | −9.5% | −33.9% | −18.0% | +2.0% |

- **Shorts lose heavily overall,** as expected over a period when BTC rose 138%. But they were **positive in the three weak periods** (2022 H2, 2025 H1, 2026 H1), which is exactly where long-only had nothing to offer.
- **Choosing between sides adds selection skill:** "both" lost 21% vs 155% for random trades in the same counts. It still loses money, and in the last 6 months it did worse than long-only.
- Natural follow-up (not run): let the 4-week trend decide the side, long in uptrends and short in downtrends. This should be pre-registered before it is run, because it is a direct reaction to the table above.

## Run 19 (`v19_vol/`, `src/vol_forecast.py`): next-24h volatility forecast (the model's real strength)
Daily forecasts at 00:00 UTC of log realized variance over the next 24h (squared 15m returns). Same 8 walk-forward folds; the last 6 months are reported separately (seen before, but this model was never tuned on them). Log forecasts are bias-corrected with a training-fold smearing factor.

| Model | QLIKE (dev) | R² log (dev) | vs HAR (dev) | QLIKE (last 6m) | R² (last 6m) | vs HAR (last 6m) |
| --- | --- | --- | --- | --- | --- | --- |
| Random walk (yesterday's RV) | 0.493 | −0.08 | −31% | 0.374 | −0.37 | −36% |
| EWMA (3-day span) | 0.437 | 0.06 | −16% | 0.304 | −0.05 | −11% |
| HAR-RV (OLS, refit per fold) | 0.376 | 0.17 | – | 0.274 | 0.01 | – |
| **LightGBM (39 features + HAR terms)** | **0.349** | **0.28** | **+7.1% (t = 1.57)** | **0.221** | **0.35** | **+19.3% (t = 2.34)** |

- LightGBM beats every baseline, including HAR-RV, the standard academic benchmark. The gain is significant on the last 6 months and borderline on the dev folds.
- Top features: trade-count surge vs the week, 7-day and 4-hour realized vol, the HAR daily and weekly terms, vol-of-vol, the 4-hour change in open interest, and ETH vs BTC. Positioning and activity data add information that pure-price vol models cannot see.
- **Using it:** hold BTC at weight min(1, 40% / forecast vol), rebalanced daily at 12.5 bps per unit traded. Annualized vol drops from 48.5% to 38.5% and max drawdown from −49.5% to −42.9% (dev). Sharpe is lower than buy-and-hold in this strong bull market (0.61 vs 0.75), but highest among the vol-targeting variants (EWMA 0.57, HAR 0.58). In the last 6 months the LightGBM vol-target had the best Sharpe of all (0.22 vs 0.20 buy-and-hold, max DD −25.9% vs −28.7%).
- **Takeaway:** direction did not survive fees, but volatility forecasting gives a real, out-of-sample improvement over standard baselines. That supports the coach's design of predicting move size for stops, sizing and the "expect a wild day" warning.

## Bottom line (runs 1–8)
1. The model has **real but small** predictive skill: it beats a random filter in all 8 configs, passes a shuffled-label test (p = 0.03), and keeps its ranking out of sample.
2. Most of that skill is **volatility prediction**; directional skill is weak (TP vs SL AUC about 0.55).
3. The gross edge (about 9 bps on average, up to about 24 bps in the top decile) is **below retail round-trip costs (25 bps)**. Breakeven is about 10 bps.
4. For the beginner-facing product, the honest output most of the time is **"no setup worth taking"**, and the model's best use is as a **risk and volatility context tool** and a filter that avoids the worst trades. It cuts losses about 3–4× compared with trading indiscriminately.

## Known limitations of these runs
- Bitstamp data instead of Binance, and no funding or open-interest features (the positioning hypothesis is untested).
- The random-filter benchmark matches the candidate *take rate*, not the final sequential trade count (in the holdout it took 163 trades vs 74).
- Daily Sharpe over sparse trades is noisy. Block-bootstrap confidence intervals and a deflated Sharpe ratio are not computed yet.
- LightGBM hyperparameters were fixed a priori and not tuned. That's deliberate, to limit the search, but it means the result may understate achievable skill.

## Next tweaks worth trying
1. Get Binance funding and open-interest data once network access allows, to test the positioning hypothesis.
2. Train a direction-only model on resolved trades (TP vs SL) and a separate volatility model, then combine them.
3. Use maker-only entry (limit orders): this could bring cost near the about 10 bps breakeven, but fill risk needs modelling.
4. Add a regime gate (e.g. trade only when 7d realized vol is above its median), since the edge looks larger in high-volatility periods.

## How to reproduce
```
pip install pandas numpy scikit-learn lightgbm pyarrow matplotlib
python src/prepare_data.py          # needs data/hist.csv.gz + data/latest.csv from the Bitstamp repo
python src/run_grid.py v1_setups setups
python src/run_grid.py v2_all all
python src/run_grid_v3.py v3_multi 0
python src/run_grid_v3.py v4_margin25 25
python src/run_rfe.py 1h 24h v5_rfe_1h_24h
python src/run_null.py 1h 24h 30
python src/run_holdout.py           # runs the one-shot holdout + cost sensitivity
```
