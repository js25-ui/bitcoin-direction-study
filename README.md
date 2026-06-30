# Can deep learning predict next-day Bitcoin direction?

A rigorous test, and an honest answer: no.

This project benchmarks LSTM and transformer sequence models against linear baselines on the task of predicting the direction of the next Bitcoin price move. Across five experimental settings, spanning hourly and daily data and three different data sources, the simple linear model matched or beat both deep models every time, and nothing produced a usable edge above the noise floor. The value of the project is the rigor of the test and the honesty of the result, not a winning prediction.

## The question

Can a sequence model (LSTM or transformer) extract a tradeable directional signal from Bitcoin price history, and does it beat a plain linear model? Predicting next-step direction is the noisiest target in markets, so this is also a test of whether added model complexity or added data helps at all.

## Method

The emphasis throughout is on not fooling myself. Financial machine learning is famous for impressive backtests that secretly leak the future into the past.

- Data: roughly six and a half years of hourly BTCUSDT bars from Binance public dumps, plus on-chain network metrics from Coin Metrics and exchange flow data pulled from on-chain queries via Dune.
- Target: the next-period log return (direction is its sign).
- Features: causal only. Momentum and volatility at multiple horizons, volume, candle shape, network activity, and exchange netflow, each computed from the present bar and earlier, never the future.
- Normalization: rolling z-scores using only a trailing window, so no future statistic leaks into any row.
- Validation: purged walk-forward cross validation with embargo gaps, so no training row's label can reach into its test block. Every model is judged only on data that comes strictly after its training.
- Leakage check: a shuffle test confirming that when the target is scrambled, the pipeline finds nothing, which is the smoke alarm for leakage.
- Metrics: directional accuracy and information coefficient, reported per fold with the spread, not a single lucky split.

## Models

- Linear and ridge regression baselines.
- A small LSTM (16 hidden units) reading a lookback window, trained with early stopping.
- A small transformer (single layer, 2 heads) with learned positional encodings, trained with early stopping.

Both deep models are kept deliberately small. On low signal-to-noise data, large models overfit instantly, so restraint is the correct choice rather than a limitation.

## Results

| Setting | Best model | Accuracy | IC |
|---|---|---|---|
| Hourly, 4 price features | linear | ~0.51 | ~0.01 |
| Hourly, 24 price features | ridge | ~0.51 | ~0.02 |
| Daily, price + on-chain | tie at chance | ~0.49 | negative |
| Daily, price + on-chain + exchange flows | tie at chance | ~0.50 | negative |

Across every setting the simple model matched or beat the deep models, and all results hovered at the coin-flip noise floor. Adding more features improved the linear model and did nothing for the deep ones. Adding genuinely new data (on-chain metrics, then real exchange inflow and outflow) did not move next-day direction above chance either.

## Takeaway

At this horizon and resolution, model complexity is not the bottleneck, information is. The predictable edge that real trading desks harvest lives in finer-grained, faster, richer data captured at scale, not in a bigger neural network applied to daily bars. Properly validated, the honest result here is a faint or null edge, and reporting that straight is the entire point.

Getting the exchange flow data meant beating a query-timeout wall to assemble six years of history from on-chain transactions, which surfaced a second lesson: the genuinely predictive data is gated behind compute and access limits, which is part of why this edge is not freely available.

## Repository

- The notebook contains the full pipeline, from data ingestion through feature construction, the purged walk-forward splitter, the shared training loop, and the five-regime evaluation.

## Notes

This is a research study, not a trading signal. The conclusion is that next-day Bitcoin direction, from freely available data, is at the noise floor, and that neither model complexity nor the data sources tested here change that.
