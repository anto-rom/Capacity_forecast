# 📘 Forecasting Model Evaluation and Ensemble Approach
## Overview
This project evaluates daily ticket‑volume forecasting models across multiple departments and verticals (Hospitality, Partners, and Payments). The goal is to identify the most accurate forecasting method for each department and build a weighted ensemble that improves robustness and prediction stability for capacity‑planning purposes.
The models evaluated include:

ARIMA
Prophet
TBATS / ETS (including ETS Damped Trend)
Weighted Ensemble based on cross‑validated performance

Performance is measured using sMAPE (Symmetric Mean Absolute Percentage Error) under cross‑validation.

## Dataset
Each row in the results corresponds to a department and includes:

department_id
department_name
vertical (Hospitality / Partners / Payments)
Cross‑validated sMAPE for:

Prophet
ARIMA
TBATS_ETS


Best_Model chosen based on lowest CV error
Ensemble weights:

Weight_Prophet
Weight_ARIMA
Weight_TBATS_ETS




## Key Findings
1. ARIMA performs best in most cases
ARIMA is the most frequently selected “best model,” especially in Hospitality and several Payment departments.
It performs particularly well when the time series is stable with moderate trend and limited structural breaks.
2. TBATS/ETS excels in seasonal or irregular patterns
TBATS/ETS (including ETS Damped Trend) performs best when:

Seasonality is more complex
The series has smoother long‑term patterns
This happens frequently in the Partners vertical.

3. Prophet rarely dominates
Prophet only outperforms other models in specific cases, usually when:

There is a clear long‑term trend
Weekly seasonality is stable
No strong noise or spikes are present
Its performance tends to worsen in highly irregular daily data.

4. Partners is the most predictable vertical
Most departments in Partners achieve sMAPE between 15–40%, indicating strong forecastability.
5. Hospitality contains the noisiest series
Some Hospitality sub‑departments show extremely high variability, with sMAPE exceeding 100%.
These may require:

Weekly aggregation
Outlier smoothing
Alternative modelling techniques


## Ensemble Behavior
The ensemble weights reflect each model’s relative performance in cross‑validation:

ARIMA typically receives the highest weight (0.30–0.40).
TBATS/ETS often receives strong weight (0.30–0.46) when it closely matches ARIMA performance.
Prophet contributes less frequently and usually with weights < 0.20.

The ensemble improves prediction stability by avoiding reliance on a single model, especially in noisy departments.

## Interpretation of sMAPE Values
General guidance:

< 30% → Excellent
30–50% → Good and operationally useful
50–80% → Acceptable for planning but indicates instability
> 100% → Very noisy series; consider transformation or redesign


Recommendations

Weekly aggregation for high‑noise departments.
Outlier detection and correction before model training.
Hybrid modelling (e.g., LightGBM with temporal features) for extremely irregular patterns.
Introduce a naive baseline for clearer performance comparison.
Use ensemble predictions for all departments to maximize stability.


## Conclusion
This evaluation demonstrates that no single model dominates across all domains.
However, the weighted ensemble approach provides consistent and resilient forecasts, which is ideal for capacity planning at multi‑vertical scale.
The results highlight:

ARIMA → strong baseline for stable patterns
TBATS/ETS → best for seasonal or smoothed trends
Ensemble → overall best choice for operational planning
