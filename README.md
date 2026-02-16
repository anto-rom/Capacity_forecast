
# **Capacity & Ticket Forecasting Model — v17.3**

This repository contains the **corporate_hybrid_forecast_v17_3** model, an end‑to‑end framework that forecasts ticket volumes for **Payments, Partners, and Hospitality** verticals, and generates department‑level operational capacity boards for **Jan‑2026 → Feb‑2027**.

The pipeline integrates forecasting engines (STL/SARIMAX), bias calibration, Einstein deduction, agent‑based capacity modeling, inventory enrichment, and a complete export system.

---

# 📐 **Pipeline Diagram**

<img width="393" height="2268" alt="image" src="https://github.com/user-attachments/assets/015e87e6-5358-42b8-9ad1-6f0ddec394da" />

# **1. Overview**

The model transforms raw ticket and agent data into:

- Accurate monthly ticket forecasts  
- Einstein‑adjusted and bias‑calibrated demand  
- Agent‑driven capacity & productivity metrics  
- Department‑level operational boards  
- PowerBI/analytics-ready consolidated dataset  

---

# **2. Architecture**

### Full pipeline sections:

1. Configuration & paths  
2. Helper utilities  
3. Forecast engines  
4. Input loading  
5. Recommended model mapping  
6. Daily forecasting  
7. Monthly aggregation  
8. Einstein deduction  
9. Bias calibration  
10. Agents metrics  
11. Capacity consolidation  
12. KPI Board creation  
13. Consolidated outputs  
14. Excel export  

---

# **3. Input Requirements**

## **3.1 Mandatory Inputs**

| File | Purpose |
|------|---------|
| `Incoming_new.xlsx` | Daily raw ticket activity |
| `department.xlsx` | Mapping of departments (ID, Name, Vertical) |

## **3.2 Optional Inputs**

| File | Purpose |
|------|---------|
| `einstein.xlsx` | Einstein-solved tickets |
| `inventory_month.xlsx` | Daily open ticket inventory |
| `agent_language_n_target.xlsx` | Historical capacity (fallback) |
| `productivity_agents.xlsx` | Agent productivity & target data |

All files are normalized and validated automatically.

---

# **4. Forecasting Logic**

## **4.1 Daily Engines**

### **STL Baseline (default)**
- Weekly seasonality  
- Robust trend extraction  
- Log smoothing  
- Uses last 7 days pattern as fallback  

### **SARIMAX‑7**
- SARIMA (1,0,1) × (1,0,1)_7  
- Log space modeling  
- Uses empirical residual variance  

**Selection rule:**  
Department → recommended engine → else STL.

---

## **4.2 Monthly Aggregation**

| Step | Description |
|------|-------------|
| Summation | Daily p50/p05/p95 → monthly totals |
| Validation | Ensures: **p05 ≤ p50 ≤ p95** |
| Smoothing | Reduces noise from irregular months |

---

# **5. Einstein Deduction**

| Concept | Description |
|--------|-------------|
| Solved tickets | Taken from `einstein.xlsx` |
| Rate formula | `einstein_rate = solved / actuals` |
| 3‑month rolling | Smoothes noise |
| Deduction | `forecast *= (1 - rate_recent)` |
| Safety | Clip 0 → 0.9 |

---

# **6. Bias-Based Calibration**

| Field | Meaning |
|-------|---------|
| `mape_pct` | Mean Absolute Percentage Error |
| `wape_pct` | Weighted Absolute Percentage Error |
| `bias_pct` | Systematic over/under‑forecast |

Calibration formula:

```
calib_factor = 1 - bias_pct / 100
Range: 0.70 → 1.30
```

Applied to:

- Monthly forecast  
- P05  
- P95  

---

# **7. Agents-Based Capacity & Productivity (v17.3 Key Upgrade)**

| KPI | Definition |
|------|-----------|
| `agents_gt1_day` | Agents with meaningful targets (>1) |
| `target_mean_gt1_day` | Mean target of productive agents |
| `prod_sum_day` | Total daily productivity |

Monthly aggregation formulas:

```
capacity_agents = mean(agents_gt1_day) * mean(target_mean_gt1_day)
productivity_agents = sum(prod_sum_day)
```

---

# **8. Capacity Consolidation**

Priority rules:

```
capacity = agents_capacity if exists else historical_capacity
productivity = productivity_agents if exists else capacity
```

Future capacity:
- Mean of last 3 historical months per department.

---

# **9. long_dept Construction**

Includes:

- Einstein-adjusted & calibrated forecasts  
- Actuals  
- Capacity (agents + fallback)  
- Productivity  
- Inventory means  
- Standardized numeric columns  

---

# **10. Boards (Board_[Department]_2627)**

### Columns  
`J‑26, F‑26, M‑26, A‑26, M‑26, J‑26 … F‑27`

### Rows (KPIs)

| KPI | Description |
|------|-------------|
| Forecast | Monthly calibrated forecast |
| Actual Volume | Actual ticket volume |
| Forecast Accuracy | `100 − |F − A| / A * 100` |
| Capacity | Monthly capacity |
| Productivity | Monthly productivity |
| Diff Capacity vs Productivity | `(Cap - Prod) / Cap` |
| Expected Forecast vs Capacity | FC − CAP |
| Actual Volume vs Productivity | `(Actual - Prod) / Actual` |
| Inventory | Mean daily open cases |
| Comments | Empty |

All metrics include division-by-zero protection.

---

# **11. Consolidated Output: `capacity_forecast`**

| Column | Description |
|--------|-------------|
| Month | `J-26` → `F-27` |
| department_name | Department |
| KPI | Metric |
| Total | Value |

Designed for BI tools such as Power BI and Tableau.

---

# **12. Excel Export**

Features:

- Full sheet-name sanitization  
- Replace sheets or create file as needed  
- Forced indexing for row labels in boards  
- Graceful fallback on empty tables  
- Writes all summary and detailed sheets  

Sheets generated:

- `Monthly_Actuals`  
- `Monthly_Forecast_RAW`  
- `Monthly_Forecast_CAL`  
- `Monthly_Capacity_Hist`  
- `Monthly_Capacity_All`  
- `Monthly_Inventory`  
- `Model_Used_and_Error`  
- `capacity_forecast`  
- `Board_[department]_2627` (multiple)

---

# **13. Improvements vs v17.2**

| Area | v17.2 | v17.3 |
|------|-------|--------|
| Agents-based capacity | ❌ | ✔ Full integration |
| Einstein deduction | Simple | ✔ 3‑month smoothing |
| Calibration | Manual | ✔ Auto + bounded |
| Sheet export | Basic | ✔ Robust & sanitized |
| KPI Board | Partial | ✔ Complete |
| Quantile guard | Weak | ✔ Strict enforcement |

---

# **14. How to Run**

```bash
python model_v17_3.py
```

Output generated at:

```
outputs/capacity_forecast_v17_3.xlsx
```

---

# **15. Author & Context**

Designed for Continuous Improvement & Operations teams across:

- Payments  
- Partners  
- Hospitality  

Focus areas:

- High‑reliability forecasting  
- Workforce & capacity planning  
- Operational KPI visibility  
- Robustness to inconsistent data  

---
