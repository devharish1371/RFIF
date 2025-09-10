# RFIF Train vs Test Evaluation Report

Date generated: auto

## Setup
- Train window end: 2022-01-21 (weekly, W-FRI)
- Test week: 2022-01-28
- Start: 2007-09-17
- Pre-tuned d_vec: d_vec_opt.npy used (K-1 = 379)
- COS settings: Ncos=512, L=6
- RFIF settings: m=2, basis_iters=80, basis_grid=600

## Metrics
- Train Open fit
  - MAE: 88.46
  - RMSE: 151.55
  - MAPE: 1.20%

- Test Open (2022-01-28)
  - Actual: 17,575.15
  - Predicted: 18,235.65
  - Abs error: 660.50
  - Rel error: 3.76%

- Test Close (2022-01-28)
  - Actual: 17,110.15
  - Predicted median: 18,247.33
  - Abs error: 1,137.18
  - Rel error: 6.65%
  - 95% band: [17,272.42, 19,222.24]
  - Covered: False (actual fell below band)

## Interpretation
- In-sample Open fit is strong; the model captures training geometry well.
- The one-week-ahead Open extrapolation shows moderate deviation (~3.8%), typical boundary drift.
- The Close prediction underestimates downside risk; actual Close lies below the 2.5% quantile. This suggests:
  - Heavier left tails than assumed by the heuristic CTS parameters, and/or
  - Under-dispersed week volatility from omegas near the boundary, and/or
  - Regime shift post training window.

## Recommendations
- Fit CTS parameters on training returns (MLE or moments) to better calibrate tails.
- Increase COS resolution for tails (e.g., Ncos=1024–2048, L=8–10) and compare.
- Re-tune RFIF parameters over a recent rolling window to capture local structure.
- Run a rolling OOS over early 2022 to assess stability and recalibration needs.

## Figure
- See `compare_plot.png` for training fit and the test week band/median vs actual.
