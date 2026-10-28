# Decision agent: a trade coach for a beginner BTC trader

`src/decision_agent.py` runs the full decision loop bar by bar with no look-ahead, producing one card every 4 hours. It's a deterministic agent: every number comes from the model or from arithmetic. No LLM is involved.

The design follows from the research in `EXPERIMENT_LOG.md`:
- The model is good at predicting **volatility** and at **ranking** setups, but weak at predicting direction (AUC about 0.55).
- The edge is smaller than retail fees for most setups.

So the agent is built to say whether a trade is worth taking *for this trader*, how big it should be, and when to sit out. It is **not** built to say "BTC will go up".

## The loop

| Step | What the agent does | Why |
| --- | --- | --- |
| 1. Observe | Takes the latest closed 1h bar and builds 39 features from data up to *t* only | No look-ahead |
| 2. Read the market | Volatility percentile vs the trailing year; 4-week trend up or down | Volatility is the model's strongest skill; the 4-week trend was the most robust published rule out of sample (run 15) |
| 3. Predict | Calibrated P(take-profit first), P(stop first), P(nothing happens) over 24h. Retrained monthly, only on labels fully resolved before the retrain date | Calibration matters because the decision compares a probability with a fee |
| 4. Decide | Expected value after **this trader's own fees** → **TAKE** (EV > 0) · **SKIP** (positive edge, but fees eat it) · **STAND ASIDE** (no edge) · **PAUSED** (model drifting) | Breakeven is about 10 bps round trip, so the same setup can be a TAKE for a maker-fee trader and a SKIP for a beginner paying taker fees |
| 5. Plan | Entry, take-profit at +1.5σ, stop at −1σ, 24h time limit, position size so the stop costs 1% of the account | Turns the volatility forecast into risk control |
| 6. Explain | Top 3 per-prediction SHAP drivers, as "what the market is doing" + "supports / works against the trade" | Plain language, as the brief asks |
| 7. Confidence | Low / medium / high = where the expected edge sits in the training distribution (below 70th percentile, 70–90th, top 10%), with that bucket's historical hit rate | Confidence is tied to evidence, not raw probability |
| 8. Learn | Logs every prediction; resolves outcomes only after their exit time; compares realized vs predicted take-profit rate over the last 60 resolved predictions (~10 days); **pauses** when realized is 1.64 SE below predicted and **resumes** when it recovers | Alpha decay showed up in our data (edge in 2023–24, gone by 2025), so the agent must notice when it stops working |

## Replay: last 6 months (2026-04-27 → 2026-10-26), 1,092 cards
Holdout caveat: this window was used in runs 8, 11 and 14, so it's a demonstration of the agent's behaviour, not a clean test.

| | Beginner (spot taker, 25 bps round trip) | Perp maker (8 bps round trip) |
| --- | --- | --- |
| Cards: TAKE / SKIP / STAND ASIDE / PAUSED | 71 / 254 / 418 / 349 | 217 / 108 / 418 / 349 |
| Trades actually taken (one at a time) | 25 | 69 |
| **Agent's result** | **−10.7%** | **−0.1%** |
| If the trader had acted on every card | −60.9% (215 trades) | −24.4% |
| Same agent without the pause rule | −10.3% (31 trades) | −7.6% (89 trades) |
| Buy and hold over the window | +2.0% | +2.0% |

What the replay shows:
- **The skip/stand-aside logic does its job.** It said "no trade" on 93% (beginner) and 80% (perp) of cards, and cut losses by 5× (beginner) and 240× (perp) compared with trading every card.
- **The pause rule helped the perp-maker trader** (−7.6% → −0.1%) and made no real difference for the beginner (−10.3% → −10.7%). It triggered for 32% of the window, because the model's realized take-profit rate (about 10%) fell well below its predictions (about 20%).
- **The agent did not make money.** The beginner still lost 10.7% on 25 trades, and buy-and-hold beat both versions. The model's ranking edge didn't carry into this period. That's the same conclusion as run 14, now seen from the trader's side.
- **The confidence buckets were not informative** in this window (high-confidence cards hit TP 11.8% of the time vs 11.4% for low). A real deployment should show this on the track-record panel, not hide it.

## Example cards (beginner fee tier, from the replay)

**TAKE, 2026-05-02 23:45 UTC.** BTC $78,687 · confidence: high
- Odds: take-profit first 22% · stop first 12% · nothing happens 67%
- Expected: +42.7 bps gross − 25 bps your fees = **+17.7 bps**
- Plan: buy $78,687 · take-profit $80,281 · stop $77,642 · exit after 24h if neither is hit · size 75% of account (risks 1%)
- Market: volatility 18th percentile · 4-week trend up
- Track record of high-confidence setups: TP hit 21% of the time (917 past cases)

**SKIP, 2026-05-03 07:45 UTC.** BTC $78,363
- Expected +20.8 bps gross, but your fees are 25 bps, so **−4.2 bps for you**. A trader on maker fees would take this.

**PAUSED, 2026-05-03 19:45 UTC.**
- Over the last 35 resolved predictions, the model expected take-profit 19.9% of the time and got 8.6% (z = −1.67). Signals are paused until calibration recovers.
- Base layer: "4-week trend is UP: if you already hold BTC, holding has historically beaten trading in this state."

Full cards: `results/agent/cards_<tier>.jsonl`. Summaries: `summary_<tier>.json`, `replay_comparison.json`.

## Design fixes made while building (documented as part of the work)
1. **The pause could get stuck.** Once paused, the agent logged no new signals, so its calibration never updated and it stayed paused for 93% of the replay. Fix: log every prediction (including while paused) and check calibration over the last 60 resolved predictions.
2. **Reasons could contradict the facts.** One card said "weekly volatility is elevated" while volatility sat at the 18th percentile, because the text was chosen from the SHAP sign rather than the feature's value. Fix: describe the actual value vs the training median, then say whether it supports or works against the trade.

## Not built (next steps)
- A trader-facing screen: the card, signals and barriers drawn on the price chart, and a fee-tier toggle.
- A pre-trade check ("I want to buy now"): grade the trader's own idea with the same card.
- Live mode: pull the latest Binance bar on a schedule instead of replaying history.
- An optional LLM layer that only rephrases the card; all numbers stay from the model.
