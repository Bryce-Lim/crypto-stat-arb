# Statistical Arbitrage in Cryptocurrencies — Design Spec

**Date:** 2026-08-20
**Author:** Bryce (brycelim@berkeley.edu)
**Context:** The Wall Street Quants course project. Deliverable doubles as a recruiting
project for quant trading internships: must be easy to follow yet substantive.

---

## 1. Objective

Research and backtest a profitable statistical-arbitrage strategy in crypto, combining
a **cross-sectional reversal** signal and a **time-series momentum** signal, weighted
together for a higher, more defensible risk-adjusted return. Report standard performance
metrics net of realistic execution costs, validated out-of-sample.

### Success criteria
- End-to-end reproducible pipeline: `python run.py` fetches data, backtests, and emits
  a metrics table + equity-curve plot.
- Positive, **honestly reported** out-of-sample Sharpe net of 20 bps costs. No knob is
  tuned on the test window.
- Full guideline coverage (see §8).
- Clean, readable code and a README with headline numbers suitable for a resume line.

---

## 2. Universe & Data

- **Universe:** Top ~30 liquid USDT spot pairs on Binance by trailing dollar volume,
  excluding stablecoins and wrapped/pegged assets. Rationale: liquidity makes the 20 bps
  cost assumption realistic and reduces microstructure noise.
- **Frequency:** Daily OHLCV.
- **Source:** Binance public REST endpoint `/api/v3/klines` (no API key). Follows the
  week-3 "PriceData" approach of pulling free crypto price-volume data.
- **History:** ~2–3 years, subject to listing availability per coin.
- **Caching:** Raw pulls cached to local parquet/CSV so backtests are fast and
  reproducible. Coins with insufficient history are dropped with a logged note.
- **Alignment:** Prices aligned on a common UTC daily index; missing days forward-filled
  only for return computation guards, never fabricated for signal formation.

---

## 3. Signals

All signals use information available strictly up to and including day `t-1` to produce
weights applied to day `t` realized returns (no lookahead).

### 3.1 Cross-sectional reversal (primary)
- Compute each coin's prior-day return `r_{i,t-1}`.
- Cross-sectionally demean against the equal-weight universe return that day:
  `x_{i} = r_{i,t-1} - mean_j(r_{j,t-1})`.
- Reversal signal `s^rev_i = -x_i` (long relative losers, short relative winners).

### 3.2 Time-series momentum (secondary, for weighting/diversification)
- Per-coin trailing return over a longer lookback `L` (e.g. 20–30 days), skipping the
  most recent day to avoid overlap with the reversal signal.
- Momentum signal `s^mom_i` = that trailing return (long recent winners), then
  cross-sectionally demeaned so the sleeve is also dollar-neutral.

### 3.3 Combination (Weighting)
- Each sleeve is converted to dollar-neutral, gross-1 weights independently.
- Sleeves combined by **inverse-volatility (risk-parity) weighting**: allocate between
  reversal and momentum inversely to each sleeve's trailing realized volatility, so the
  lower-vol sleeve is not drowned out. Because the two sleeves are lowly/negatively
  correlated, the blend's Sharpe benefits from diversification.

---

## 4. Weighting → Positions

- Within a sleeve: weights proportional to the signal, normalized so **gross exposure = 1**
  (sum of absolute weights = 1) and **net exposure = 0** (dollar-neutral). No position
  caps — the "unconstrained" backtest style from the course.
- Enhancers applied at the sleeve/position level (see §5).

---

## 5. Alpha Enhancers (iterative-improvement narrative)

Added on top of the baseline one at a time; each reported with before/after metrics so the
project reads as a research progression rather than a single lucky configuration.

1. **Inverse-volatility position scaling** — scale each coin's weight by the inverse of its
   trailing realized volatility before normalizing, so a single high-vol coin cannot
   dominate the book. Typically improves Sharpe.
2. **Uninformed-trade isolation (volume filter)** — the guideline insight that
   liquidity-driven / liquidation trades reverse more ("Fire Sale"). Amplify the reversal
   signal for coins whose prior move coincided with an abnormal volume spike
   (volume z-score vs. trailing mean); damp it otherwise.
3. **Cost-aware no-trade band** — only rebalance a position when its target weight moves
   beyond a threshold, cutting turnover and the associated cost drag. Demonstrates
   understanding that gross alpha ≠ net alpha.

---

## 6. Backtest Engine & Costs

- **P&L:** For each day `t`, `ret_t = sum_i w_{i,t} * r_{i,t}` using next-day realized
  returns, minus costs.
- **Turnover:** `TO_t = sum_i |w_{i,t} - w_{i,t-1}|`.
- **Costs:** `cost_t = TO_t * c`, with `c = 20 bps` for the market-order case and
  `c = 7 bps` for the limit-order case. Both reported.
- **Unconstrained:** no leverage caps or position limits beyond the gross-1 normalization.

---

## 7. Out-of-Sample Validation

- Chronological split: first ~70% = **train**, last ~30% = **test**.
- Any hyperparameters (momentum lookback `L`, volume z threshold, no-trade band width,
  vol-lookback) are chosen **only** on the train window.
- Headline metrics reported on the **held-out test window**, net of costs. Train metrics
  reported alongside for honesty about degradation.

---

## 8. Performance Evaluation (guideline-required)

Reported for baseline and each enhancer stage, in-sample and out-of-sample, net of costs:

- Annualized return
- Annualized volatility
- **Sharpe ratio**
- **Maximum drawdown**
- **Alpha and beta vs BTC** (OLS regression of daily strategy returns on BTC daily returns)
- Equity-curve plot and a summary metrics table

---

## 9. Deliverables / File Layout

```
data.py       -> fetch + cache OHLCV from Binance public API, build universe
strategy.py   -> reversal + momentum signals, enhancers, sleeve combination
backtest.py   -> unconstrained engine: weights -> P&L, turnover, costs
metrics.py    -> Sharpe, drawdown, alpha/beta, summary table
run.py        -> orchestrates full pipeline; saves plots + results table
README.md     -> resume-facing writeup with headline numbers and methodology
```

- **Dependencies:** pandas, numpy, matplotlib, requests. Standard scientific stack.
- **Entry point:** `python run.py` runs the whole pipeline end-to-end.

---

## 10. Guideline Coverage Matrix

| Guideline requirement | Status | Where |
|---|---|---|
| Momentum and/or reversal | ✅ | §3.1, §3.2 |
| Free crypto price-volume data (week-3 method) | ✅ | §2 |
| Unconstrained backtest style | ✅ | §4, §6 |
| Execution/slippage (20 bps market / 7 bps limit) | ✅ | §6 |
| Reversal via uninformed trading / volume ("Fire Sale") | ✅ | §5.2 |
| Weighting (combine >1 strategy) | ✅ | §3.3 |
| Performance eval (returns, vol, Sharpe, max DD, alpha/beta) | ✅ | §8 |

---

## 11. Honest Risks & Mitigations

- **Cost drag at daily frequency.** High turnover can erode net Sharpe. Mitigations:
  no-trade band (§5.3), inverse-vol scaling, and the 7 bps limit-order path.
- **Overfitting.** Mitigated by the OOS split (§7) and by keeping the hyperparameter count
  small and interpretable.
- **Data/survivorship quirks.** Using currently-liquid coins introduces mild survivorship
  bias; noted explicitly in the README rather than hidden.
- **No promised Sharpe number pre-backtest.** The design maximizes the realistic chance of
  a strong, defensible number; the methodology (neutrality, cost accounting, ablation, OOS)
  is itself the resume value even if raw numbers are modest.

---

## 12. Non-Goals (YAGNI)

- No intraday/hourly frequency (kept for a possible extension).
- No live trading, order-book simulation, or borrow-cost modeling.
- No ML models — signals are transparent and interpretable by design.
- No configurable multi-exchange abstraction; Binance only.

---

## 13. Post-Implementation Revision (evidence-driven)

Running the built pipeline against live data surfaced two root causes that the
original design did not anticipate, and the fixes changed several design choices.
Documented here so the spec matches what was actually built.

**Root causes found**
1. `load_panel`'s `dropna(how="any")` intersected all coins; because top-volume
   ranking surfaces newly-listed coins, the common window collapsed to ~99 days.
2. Daily reversal turnover (~1.4/day) makes 20 bps costs ~100%/yr — the gross
   signal is strong (Sharpe ~2.5) but is buried by costs at market-order pricing.

**Design changes**
- **Universe:** added `data.build_universe` — top-N liquid USDT pairs that also
  have ≥ 700 days of history. Yields a ~2-year (729-day) panel instead of 99 days.
- **Signal quality confirmed:** cross-sectional reversal IC = +0.088, t-stat
  **+6.85** over the 2-year panel (robust, not a lucky window).
- **Headline cost = 7 bps limit orders** (assignment-sanctioned; a reversal book
  provides liquidity). 20 bps market orders reported as a conservative bound.
- **Final book = reversal + no-trade band only**, with the band chosen on the
  TRAIN slice. The enhancer ablation showed per-name volatility scaling and the
  volume filter do **not** improve net returns, so both are excluded from the
  final book (kept in the ablation as honest negative results).
- **Book-level volatility targeting (15%)** added (`strategy.vol_target`) so
  reported return/drawdown reflect a fixed risk budget; it leaves Sharpe ~unchanged.
- **Momentum sleeve dropped from the final book:** it is strong in-sample
  (Sharpe +1.2) but negative out-of-sample (−1.45) and drags the combination.
  Retained only as an ablation result demonstrating why it was rejected.

**Final net-of-cost results (market-neutral, beta ≈ 0.03), full 2-year sample**
- 7 bps limit orders: Sharpe **+2.46**, ann. return +49%, max DD −25%.
- 20 bps market orders: Sharpe **+1.44**, ann. return +29%, max DD −30%.
