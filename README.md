# **Capacity & Ticket Forecasting Model — v17.3**

This repository contains the **corporate_hybrid_forecast_v17_3** model, an end‑to‑end framework that forecasts ticket volumes for **Payments, Partners, and Hospitality** verticals, and generates department‑level operational capacity boards for **Jan‑2026 → Feb‑2027**.

The pipeline integrates forecasting engines (STL/SARIMAX), bias calibration, Einstein deduction, agent‑based capacity modeling, inventory enrichment, and a complete export system.

---

# 📐 **Pipeline Diagram**

```mermaid
flowchart TD

A[1. Load Inputs\nIncoming_new, Dept Map,\nAgents, Einstein, Inventory] --> B

B[2. Preprocessing & Cleaning\n- Normalize headers\n- Filter verticals\n- Remove excluded departments] --> C

C[3. Daily Forecast Engines\nSTL Baseline / SARIMAX-7\n+ fallback logic] --> D

D[4. Daily → Monthly Aggregation\nSum p50/p05/p95\nValidate quantiles] --> E

E[5. Einstein Deduction\n3‑month solved-rate smoothing\nApply deduction & clipping] --> F

F[6. Bias-Based Calibration\nApply bias table\nClip calib_factor 0.70–1.30] --> G

G[7. Agents KPIs Extraction\ncapacity_agents\nproductivity_agents] --> H

H[8. Historical Capacity Merge\nFill future months\nUse fallback logic] --> I

I[9. Build long_dept Table\nForecast + Actuals + Capacity + Productivity + Inventory] --> J

J[10. Build Board_[dept]_2627\nFull KPI matrix for 2026–2027] --> K

K[11. Build consolidated\ncapacity_forecast Long Table] --> L

L[12. Export to Excel\nSanitized sheet names\nReplace or create sheets]
``
























FilePurposeeinstein.xlsxEinstein-solved ticketsinventory_month.xlsxDaily open ticket inventoryagent_language_n_target.xlsxHistorical capacity (fallback)productivity_agents.xlsxAgent productivity & target data

2. Architecture
Full pipeline sections:

Configuration & paths
Helper utilities
Forecast engines
Input loading
Recommended model mapping
Daily forecasting
Monthly aggregation
Einstein deduction
Bias calibration
Agents metrics
Capacity consolidation
KPI Board creation
Consolidated outputs
Excel export


3. Input Requirements
Mandatory Inputs

















FilePurposeIncoming_new.xlsxDaily raw ticket activitydepartment.xlsxMapping of departments (ID, Name, Vertical)
Optional Inputs

























FilePurposeeinstein.xlsxEinstein-solved ticketsinventory_month.xlsxDaily open ticket inventoryagent_language_n_target.xlsxHistorical capacity (fallback)productivity_agents.xlsxAgent productivity & target data
All files are normalized and validated automatically.

4. Forecasting Logic
4.1 Daily Engines
STL Baseline (default)

Weekly seasonality
Robust trend extraction
Log smoothing
Uses last 7 days pattern as fallback

SARIMAX‑7

SARIMA (1,0,1) × (1,0,1)_7
Log space modeling
Uses empirical residual variance

Selection rule:
Department → recommended engine (if exists) → else STL.

4.2 Monthly Aggregation





















StepDescriptionSummationDaily p50/p05/p95 → monthly totalsValidationEnsures: p05 ≤ p50 ≤ p95SmoothingReduces noise from irregular months

5. Einstein Deduction





























ConceptDescriptionSolved ticketsTaken from einstein.xlsxRate formulaeinstein_rate = solved / actuals3‑month rollingSmoothes noiseDeductionforecast *= (1 - rate_recent)SafetyClip 0 → 0.9 max deduction

6. Bias-Based Calibration





















FieldMeaningmape_pctMean Absolute Percentage Errorwape_pctWeighted Absolute Percentage Errorbias_pctSystematic over/under‑forecast
Calibration formula:
calib_factor = 1 - bias_pct / 100
Range: 0.70 → 1.30

Applied to:

Forecast
P05
P95


7. Agents-Based Capacity & Productivity (v17.3 Key Upgrade)





















KPIDefinitionagents_gt1_dayAgents with meaningful targets (>1)target_mean_gt1_dayMean target of productive agentsprod_sum_dayTicket-equivalent productivity
Monthly metrics:
capacity_agents = mean(agents_gt1_day) * mean(target_mean_gt1_day)
productivity_agents = sum(prod_sum_day)


8. Capacity Consolidation
Priority:
capacity = agents_capacity if exists else historical_capacity
productivity = productivity_agents if exists else capacity

Future months:
Mean of last 3 months of historical capacity.

9. long_dept Construction
Contains:

Forecast (calibrated)
Actuals
Einstein-adjusted values
Capacity & productivity
Inventory
Cleaned numeric fields

This is the central dataset that powers the boards.

10. Boards (Board_[Department]_2627)
Columns:
J‑26, F‑26, M‑26, A‑26, M‑26, J‑26 … F‑27
Rows (KPIs):

















































KPIDescriptionForecastMonthly calibrated forecastActual VolumeActual ticket volumeForecast Accuracy`100 −CapacityMonthly capacityProductivityMonthly productivityDifference Capacity vs ProductivityPercentage differenceExpected Forecast vs CapacityGap between FC and capacityActual Volume vs ProductivityPerformance vs workloadInventoryMean daily open casesCommentsEmpty row for user notes
All ratios include protection against divide-by-zero.

11. Consolidated Output: capacity_forecast

























ColumnDescriptionMonthJ-26 → F-27department_nameDepartmentKPIMetric nameTotalValue
Used directly in BI dashboards.

12. Excel Export
Features:

Full sanitization of sheet names
Replace or create sheets safely
Index shown for department boards
Handles empty datasets gracefully
Writes all KPI pages + consolidated outputs


13. Improvements vs v17.2

Areav17.2v17.3Agents-based capacity❌✔ Full integrationEinstein deductionSimple✔ 3‑month rollingCalibrationManual✔ Automated + boundedSheet exportFragile✔ Sanitized + safeKPI BoardPartial✔ Complete matrixQuantile guardWeak✔ Strict monotonic check

14. How to Run
Shellpython model_v17_3.pyMostrar más líneas
Outputs generated to:
outputs/capacity_forecast_v17_3.xlsx


15. Author & Context
Built for Continuous Improvement & Operations teams across:

Payments
Partners
Hospitality

Priorities:

Reliable, stable forecasting
Capacity-planning alignment
Operational usability
Robustness to noisy inputs









































Areav17.2v17.3Agents‑based capacity❌ No✅ Yes — new and robust methodEinstein deductionBasicImproved 3‑month rolling avgBias calibrationManualAutomatic, standardized tableDaily model failuresWeak fallbackFull fault‑tolerant pipelineSheet sanitizationLimitedStrong sanitization & indexingBoard generationPartialComplete, validated, saferQuantile guardinconsistentstrict monotonicity check

13. Limitations & Future Work

Backtesting not automated yet (manual Model_Used_and_Error)
No hierarchical reconciliation across departments
Ensemble method could be added
Agent productivity may require anomaly detection
Extended language‑based forecasting may be added (we intentionally ignore language per your specifications)


14. How to Run
Shellpython model_v17_3.pyMostrar más líneas
Outputs will be created under:
/outputs/capacity_forecast_v17_3.xlsx


15. Author & Context
The model was co‑designed for Continuous Improvement operations in Payments, Partners, Hospitality verticals, with focus on:

Accurate demand forecasting
Reliable capacity estimation
Simplified operational communication
Robustness against noisy ticket patterns
Easy integration into planning discussions


If you'd like, I can also generate:
✅ A shorter README version
✅ A diagram of the full pipeline
✅ A CONTRIBUTING.md or architecture.md
✅ A GitHub Wiki structure
Just tell me what format you prefer.
Orígenes
