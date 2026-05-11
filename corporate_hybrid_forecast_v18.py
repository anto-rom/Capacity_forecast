# coding: utf-8
# corporate_hybrid_forecast_v8_SQL (refactor) — FIXED FULL SCRIPT
#
# Key refactor:
# - Replace Excel inputs with Azure SQL Server (APP_FLOW.CustomerService.*)
# - Keep the pipeline logic aligned with v17_x: daily forecast -> monthly sum -> adjustments -> export
#
# Notes:
# - Comments intentionally in English (per your preference).
# - Password must be provided via env var: SQL_PASSWORD

from pathlib import Path
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlalchemy as sa
from urllib.parse import quote_plus

# -----------------------------
# 1) Configuration
# -----------------------------
OUTPUT_DIR = os.getenv("CAPACITY_OUTPUT_DIR", str(Path.cwd() / "outputs"))
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "capacity_forecast_v18_SQL.xlsx")

HORIZON_MONTHS = 12
HORIZON_DAYS = 365

VERTICALS_TARGET = ["Payments", "Partners", "Hospitality"]
TARGET_TPH = 6.0

EXCLUDE_DEPARTMENT_NAME_TOKENS = ["PROJ", "DIST", "KEY", "PROXIMIS"]

# If True: uplift training by Calls Not Indexed (history)
APPLY_CALLS_NOT_INDEXED_TO_TRAINING = True

# Recent-rate lookback for calls-not-indexed and repeats ratios (safe default)
RECENT_RATE_MONTHS = 3

print("OUTPUT_XLSX →", OUTPUT_XLSX)

# -----------------------------
# 1b) SQL Connection (Azure SQL Server)
# -----------------------------
SQL_SERVER = "wlms-sql-pay-appflow-dev.database.windows.net"
SQL_DATABASE = "APP_FLOW"
SQL_SCHEMA = "CustomerService"
SQL_USER = "CS_USER"
SQL_PASSWORD = os.getenv("SQL_PASSWORD")

if not SQL_PASSWORD:
    raise EnvironmentError(
        "Missing SQL_PASSWORD environment variable. Set it in your OS before running the script."
    )

pwd = quote_plus(SQL_PASSWORD)
ODBC_DRIVER = os.getenv("SQL_ODBC_DRIVER", "ODBC Driver 18 for SQL Server")
ODBC_DRIVER_ENC = quote_plus(ODBC_DRIVER)

engine = sa.create_engine(
    f"mssql+pyodbc://{SQL_USER}:{pwd}@{SQL_SERVER}/{SQL_DATABASE}"
    f"?driver={ODBC_DRIVER_ENC}&Encrypt=yes&TrustServerCertificate=no",
    fast_executemany=True
)

# -----------------------------
# 2) Helpers
# -----------------------------
def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df

def pick_col(df: pd.DataFrame, candidates):
    """Pick the first present column from candidates (case-insensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None

def to_month_start(dt_series):
    s = pd.to_datetime(dt_series, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp(how="start")

def safe_numeric(s, default=0.0):
    """
    Safe numeric conversion.
    - Works with Series, arrays, or scalars.
    """
    if isinstance(s, (int, float, np.number)):
        return default if pd.isna(s) else s
    return pd.to_numeric(s, errors="coerce").fillna(default)

def validate_quantiles(dfm: pd.DataFrame):
    needed = {"forecast_p05_dept", "forecast_monthly_dept", "forecast_p95_dept"}
    if not needed.issubset(dfm.columns):
        return
    viol = dfm[
        (dfm["forecast_p05_dept"] > dfm["forecast_monthly_dept"]) |
        (dfm["forecast_monthly_dept"] > dfm["forecast_p95_dept"])
    ]
    if not viol.empty:
        raise ValueError("Quantile order violation in monthly aggregation.")

def read_table(query: str) -> pd.DataFrame:
    df = pd.read_sql(query, engine)
    return std_cols(df)

# -----------------------------
# 3) Forecast engines
# -----------------------------
def _seasonal_naive_forecast(y: pd.Series, horizon: int, season_length: int):
    """Seasonal naive forecast repeating last season_length values."""
    y = y.copy()
    if len(y) < season_length:
        last = float(y.iloc[-1]) if len(y) else 0.0
        return np.full(horizon, last, dtype=float)
    pattern = y.iloc[-season_length:].to_numpy(dtype=float)
    reps = int(np.ceil(horizon / season_length))
    return np.tile(pattern, reps)[:horizon]

def forecast_daily_baseline(y: pd.Series, horizon_days: int = 365) -> pd.DataFrame:
    """
    Baseline forecast:
    - STL decomposition on log1p with weekly seasonality (7) if available
    - trend forecast: flat last trend value
    - season forecast: repeat last 7 seasonal values
    - p05/p95: +/- 1.645 * std(residual)
    Fallback: seasonal naive on original scale.
    """
    y = y.asfreq("D").fillna(0.0).astype(float)
    idx_future = pd.date_range(y.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    # If too short, fallback to seasonal naive directly
    if len(y) < 56:
        p50 = _seasonal_naive_forecast(y, horizon_days, 7)
        resid = (y[7:] - y.shift(7)[7:]).dropna()
        std = float(resid.std()) if len(resid) else max(1.0, np.sqrt(max(float(y.mean()), 1.0)))
        p05 = np.clip(p50 - 1.645 * std, 0, None)
        p95 = p50 + 1.645 * std
        return pd.DataFrame({"date": idx_future, "p50": p50, "p05": p05, "p95": p95})

    try:
        from statsmodels.tsa.seasonal import STL

        ly = np.log1p(y)
        res = STL(ly, period=7, robust=True).fit()

        trend = res.trend
        seas = res.seasonal
        resid = res.resid

        trend_f = np.full(horizon_days, float(trend.iloc[-1]), dtype=float)
        seas_f = _seasonal_naive_forecast(seas, horizon_days, 7)

        mu_log = trend_f + seas_f
        std = float(resid.std()) if float(resid.std()) > 0 else 0.5

        p50 = np.expm1(mu_log)
        p05 = np.expm1(mu_log - 1.645 * std)
        p95 = np.expm1(mu_log + 1.645 * std)

        p50 = np.clip(p50, 0, None)
        p05 = np.clip(p05, 0, None)
        p95 = np.clip(p95, 0, None)

        return pd.DataFrame({"date": idx_future, "p50": p50, "p05": p05, "p95": p95})

    except Exception:
        # Safe fallback
        p50 = _seasonal_naive_forecast(y, horizon_days, 7)
        resid = (y[7:] - y.shift(7)[7:]).dropna()
        std = float(resid.std()) if len(resid) else max(1.0, np.sqrt(max(float(y.mean()), 1.0)))
        p05 = np.clip(p50 - 1.645 * std, 0, None)
        p95 = p50 + 1.645 * std
        return pd.DataFrame({"date": idx_future, "p50": p50, "p05": p05, "p95": p95})

def forecast_daily_sarimax(y: pd.Series, horizon_days: int = 365) -> pd.DataFrame | None:
    """
    SARIMAX weekly seasonality.
    If statsmodels is unavailable or fit fails, return None to allow fallback.
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")

        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = y.asfreq("D").fillna(0.0).astype(float)
        idx_future = pd.date_range(y.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")

        # Simple robust configuration (weekly seasonality via seasonal_order period=7)
        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)

        pred = res.get_forecast(steps=horizon_days)
        mean = pred.predicted_mean.to_numpy(dtype=float)

        # Approx CI -> p05/p95 using normal approx
        se = pred.se_mean.to_numpy(dtype=float)
        p50 = np.clip(mean, 0, None)
        p05 = np.clip(mean - 1.645 * se, 0, None)
        p95 = np.clip(mean + 1.645 * se, 0, None)

        return pd.DataFrame({"date": idx_future, "p50": p50, "p05": p05, "p95": p95})

    except Exception:
        return None

def forecast_monthly_inventory_baseline(y: pd.Series, horizon_months: int = 12) -> pd.DataFrame:
    """
    Monthly inventory forecast (backlog):
    - If enough history: STL on log1p with 12-month seasonality
    - Else: last-value naive
    Returns DataFrame with columns: ['month','inventory_forecast']
    """
    y = y.asfreq("MS").fillna(0.0).astype(float)
    last_month = y.index.max()
    future_idx = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")

    if len(y) < 24:
        last = float(y.iloc[-1]) if len(y) else 0.0
        f = np.full(horizon_months, last, dtype=float)
        return pd.DataFrame({"month": future_idx, "inventory_forecast": np.clip(f, 0, None)})

    try:
        from statsmodels.tsa.seasonal import STL

        ly = np.log1p(y)
        res = STL(ly, period=12, robust=True).fit()
        trend = res.trend
        seas = res.seasonal
        resid = res.resid

        trend_f = np.full(horizon_months, float(trend.iloc[-1]), dtype=float)
        seas_f = _seasonal_naive_forecast(seas, horizon_months, 12)

        mu_log = trend_f + seas_f
        f = np.expm1(mu_log)
        return pd.DataFrame({"month": future_idx, "inventory_forecast": np.clip(f, 0, None)})

    except Exception:
        last = float(y.iloc[-1]) if len(y) else 0.0
        f = np.full(horizon_months, last, dtype=float)
        return pd.DataFrame({"month": future_idx, "inventory_forecast": np.clip(f, 0, None)})

# -----------------------------
# 4) Load inputs from SQL
# -----------------------------
# 4a) Incoming tickets (daily)
incoming_raw = read_table(f"""
SELECT
    [Date] AS date,
    department_id,
    ticket_total
FROM {SQL_SCHEMA}.incoming_tickets
WHERE department_id IS NOT NULL
  AND [Date] IS NOT NULL
""")

incoming = incoming_raw.copy()
incoming["date"] = pd.to_datetime(incoming["date"], errors="coerce")
incoming["month"] = to_month_start(incoming["date"])
incoming["department_id"] = pd.to_numeric(incoming["department_id"], errors="coerce").astype("Int64")
incoming["ticket_total"] = pd.to_numeric(incoming["ticket_total"], errors="coerce").fillna(0)

# 4b) Department map
dept_map = read_table(f"""
SELECT
    department_id,
    department_name,
    vertical
FROM {SQL_SCHEMA}.department
""")
dept_map["department_id"] = pd.to_numeric(dept_map["department_id"], errors="coerce").astype("Int64")

incoming = incoming.merge(dept_map, on="department_id", how="left")

# Scope & exclusions
incoming = incoming[incoming["vertical"].isin(VERTICALS_TARGET)].copy()

mask_excl = pd.Series(False, index=incoming.index)
for tok in EXCLUDE_DEPARTMENT_NAME_TOKENS:
    mask_excl = mask_excl | incoming["department_name"].astype(str).str.upper().str.contains(tok.upper(), na=False)

incoming = incoming.loc[~mask_excl].copy()
print("Loaded incoming rows =", len(incoming))

# 4c) Calls Not Indexed (monthly)
calls_ni_month = read_table(f"""
SELECT
    department_id,
    DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1) AS month,
    COUNT(*) AS calls_not_indexed
FROM {SQL_SCHEMA}.calls_not_indexed
WHERE department_id IS NOT NULL
  AND [Date] IS NOT NULL
GROUP BY
    department_id,
    DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1)
""")
calls_ni_month["month"] = pd.to_datetime(calls_ni_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

# 4d) Repeats (monthly, derived from DAILY_RR_new)
repeats_month = read_table(f"""
SELECT
    department_id,
    DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1) AS month,
    COUNT(DISTINCT case_number) AS repeats_workload
FROM {SQL_SCHEMA}.DAILY_RR_new
WHERE department_id IS NOT NULL
  AND [Date] IS NOT NULL
GROUP BY
    department_id,
    DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1)
""")
repeats_month["month"] = pd.to_datetime(repeats_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

# 4e) Monthly actuals (cases)
monthly_actuals = (
    incoming
    .groupby(["vertical", "department_id", "department_name", "month"], as_index=False)
    .agg(actual_volume=("ticket_total", "sum"))
)

# Optional uplift of training signal (cases + calls not indexed)
if APPLY_CALLS_NOT_INDEXED_TO_TRAINING and isinstance(calls_ni_month, pd.DataFrame) and not calls_ni_month.empty:
    monthly_actuals = monthly_actuals.merge(
        calls_ni_month,
        on=["department_id", "month"],
        how="left"
    )
    monthly_actuals["calls_not_indexed"] = safe_numeric(monthly_actuals.get("calls_not_indexed", 0.0), 0.0)
    # Keep actual_volume unchanged; calls_not_indexed used only for recent-rate estimation / optional training variants

# 4f) Inventory (monthly backlog – SQL view)
inventory_month = read_table(f"""
SELECT
    department_id,
    month,
    inventory_cases AS inventory
FROM {SQL_SCHEMA}.vw_capacity_inventory_monthly
""")
inventory_month["month"] = pd.to_datetime(inventory_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
print("Inventory monthly rows:", len(inventory_month))

# 4g) Einstein (monthly) — loaded but Einstein deduction step is currently disabled later
ein_month = read_table(f"""
WITH ein AS (
    SELECT
        me.Week_Year,
        me.Month_number,
        me.department,
        SUM(me.Not_Reopened) AS einstein_solved
    FROM {SQL_SCHEMA}.MONTHLY_einstein me
    GROUP BY me.Week_Year, me.Month_number, me.department
),
yr AS (
    SELECT
        d.Week_Year,
        d.Month_number,
        YEAR(MIN(d.[Date])) AS yr
    FROM {SQL_SCHEMA}.[date] d
    GROUP BY d.Week_Year, d.Month_number
)
SELECT
    dp.department_id,
    ein.department AS department_name,
    DATEFROMPARTS(yr.yr, ein.Month_number, 1) AS [month],
    ein.einstein_solved
FROM ein
JOIN yr
  ON ein.Week_Year = yr.Week_Year
 AND ein.Month_number = yr.Month_number
LEFT JOIN {SQL_SCHEMA}.department dp
  ON dp.department_name = ein.department;
""")
ein_month["month"] = pd.to_datetime(ein_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

# ------------------------------------------------------------
# Agent-level daily productivity (for historical averages)
# ------------------------------------------------------------
pa_agent = read_table(f"""
SELECT
    department_id,
    agent_id,
    CAST([Date] AS date) AS [date],
    CAST(resolved_total + transfer_total AS FLOAT) AS prod_total_model
FROM {SQL_SCHEMA}.agent_productivity_new_daily
WHERE department_id IS NOT NULL
  AND [Date] IS NOT NULL
""")
pa_agent["date"] = pd.to_datetime(pa_agent["date"], errors="coerce")

# ------------------------------------------------------------
# Historical productivity per agent (daily average)
# ------------------------------------------------------------
prod_per_agent_day = (
    pa_agent
    .groupby("department_id", as_index=False)
    .agg(avg_cases_per_agent_day=("prod_total_model", "mean"))
)
print("Avg productivity per agent/day:")
print(prod_per_agent_day.head())

# ------------------------------------------------------------
# Department-day capacity/productivity (derived)
# ------------------------------------------------------------
pa_raw = read_table(f"""
SELECT
    department_id,
    CAST([Date] AS date) AS [date],
    COUNT(DISTINCT agent_id) AS capacity_agents,
    COUNT(DISTINCT CASE WHEN (resolved_total + transfer_total) > 0 THEN agent_id END) AS productivity_agents
FROM {SQL_SCHEMA}.agent_productivity_new_daily
WHERE department_id IS NOT NULL
  AND [Date] IS NOT NULL
GROUP BY
    department_id,
    CAST([Date] AS date)
""")
pa_raw["date"] = pd.to_datetime(pa_raw["date"], errors="coerce")

# ------------------------------------------------------------
# Historical MONTHLY cap/prod (mean daily agents per month)
# NOTE: history only
# ------------------------------------------------------------
cap_prod_hist = (
    pa_raw
    .assign(month=lambda df: df["date"].values.astype("datetime64[M]"))
    .groupby(["department_id", "month"], as_index=False)
    .agg(
        capacity_agents=("capacity_agents", "mean"),
        productivity_agents=("productivity_agents", "mean")
    )
)
cap_prod_hist["month"] = pd.to_datetime(cap_prod_hist["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
print("Historical cap/prod months:", len(cap_prod_hist))
print(cap_prod_hist.head())

# -----------------------------
# 7) Monthly aggregation (build monthly_fc_raw)  ✅
# -----------------------------
required_cols = {"date", "department_id", "ticket_total", "vertical", "department_name"}
missing = required_cols - set(incoming.columns)
if missing:
    raise KeyError(f"incoming is missing required columns: {missing}")

daily_hist = (
    incoming
    .groupby(["vertical", "department_id", "department_name", "date"], as_index=False)
    .agg(y=("ticket_total", "sum"))
)
daily_hist["date"] = pd.to_datetime(daily_hist["date"], errors="coerce")
daily_hist["y"] = safe_numeric(daily_hist["y"], 0.0).clip(lower=0.0)

daily_fc_rows = []
for (vertical, dept_id, dept_name), g in daily_hist.groupby(["vertical", "department_id", "department_name"]):
    g = g.sort_values("date")
    y = g.set_index("date")["y"].asfreq("D").fillna(0.0)

    fc = forecast_daily_sarimax(y, horizon_days=HORIZON_DAYS)
    model_used = "SARIMAX" if fc is not None else "BASELINE"

    if fc is None:
        fc = forecast_daily_baseline(y, horizon_days=HORIZON_DAYS)

    fc = fc.copy()
    fc["date"] = pd.to_datetime(fc["date"], errors="coerce")
    for c in ["p50", "p05", "p95"]:
        fc[c] = safe_numeric(fc.get(c, 0.0), 0.0).clip(lower=0.0)

    fc["vertical"] = vertical
    fc["department_id"] = dept_id
    fc["department_name"] = dept_name
    fc["model_used"] = model_used
    daily_fc_rows.append(fc)

daily_fc = pd.concat(daily_fc_rows, ignore_index=True) if daily_fc_rows else pd.DataFrame(
    columns=["date", "p50", "p05", "p95", "vertical", "department_id", "department_name", "model_used"]
)

daily_fc["month"] = to_month_start(daily_fc["date"])

monthly_fc_only = (
    daily_fc
    .groupby(["vertical", "department_id", "department_name", "month"], as_index=False)
    .agg(
        forecast_monthly_dept=("p50", "sum"),
        forecast_p05_dept=("p05", "sum"),
        forecast_p95_dept=("p95", "sum"),
    )
)

monthly_actuals["month"] = pd.to_datetime(monthly_actuals["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
last_hist_month = monthly_actuals["month"].max()

future_months = pd.date_range(
    last_hist_month + pd.offsets.MonthBegin(1),
    periods=HORIZON_MONTHS,
    freq="MS"
)
all_months = pd.DatetimeIndex(sorted(set(monthly_actuals["month"].unique()).union(set(future_months))))

dept_dim = monthly_actuals[["vertical", "department_id", "department_name"]].drop_duplicates()
spine = (
    dept_dim.assign(_k=1)
    .merge(pd.DataFrame({"month": all_months, "_k": 1}), on="_k", how="outer")
    .drop(columns=["_k"])
)

monthly_fc_raw = spine.merge(
    monthly_actuals[["vertical", "department_id", "department_name", "month", "actual_volume"]],
    on=["vertical", "department_id", "department_name", "month"],
    how="left"
).merge(
    monthly_fc_only,
    on=["vertical", "department_id", "department_name", "month"],
    how="left"
)

# Fill history with actuals to stabilize downstream (optional but recommended)
mask_hist = monthly_fc_raw["actual_volume"].notna()
for c in ["forecast_monthly_dept", "forecast_p05_dept", "forecast_p95_dept"]:
    monthly_fc_raw[c] = safe_numeric(monthly_fc_raw.get(c, 0.0), 0.0)
    monthly_fc_raw.loc[mask_hist, c] = safe_numeric(monthly_fc_raw.loc[mask_hist, "actual_volume"], 0.0)

monthly_fc_raw["actual_volume"] = safe_numeric(monthly_fc_raw["actual_volume"], 0.0).clip(lower=0.0)

validate_quantiles(monthly_fc_raw)

monthly_fc_raw = monthly_fc_raw.sort_values(["vertical", "department_name", "month"]).reset_index(drop=True)
print("Built monthly_fc_raw rows:", len(monthly_fc_raw))
print(monthly_fc_raw.tail())

# -----------------------------
# 8) Einstein: build monthly_adj (Einstein deduction temporarily disabled, but monthly_adj is required)
# -----------------------------
monthly_adj = monthly_fc_raw.copy()
monthly_adj["month"] = pd.to_datetime(monthly_adj["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

# -----------------------------
# 9) Calls Not Indexed: recent ratio and forecast additional workload
# -----------------------------
calls_recent = pd.DataFrame(columns=["department_id", "calls_not_indexed_rate_recent"])

if isinstance(calls_ni_month, pd.DataFrame) and not calls_ni_month.empty:
    hist_calls = monthly_actuals.merge(calls_ni_month, on=["department_id", "month"], how="left")
    hist_calls["calls_not_indexed"] = safe_numeric(hist_calls.get("calls_not_indexed", 0.0), 0.0)
    hist_calls["actual_volume"] = safe_numeric(hist_calls["actual_volume"], 0.0)

    # Rate = calls_not_indexed / actual_volume (safe), recent = mean last N non-zero months per dept
    hist_calls["rate"] = 0.0
    nz = hist_calls["actual_volume"] > 0
    hist_calls.loc[nz, "rate"] = (hist_calls.loc[nz, "calls_not_indexed"] / hist_calls.loc[nz, "actual_volume"])

    hist_calls = hist_calls.sort_values(["department_id", "month"])
    calls_recent = (
        hist_calls.groupby("department_id", as_index=False)
        .apply(lambda g: g.tail(RECENT_RATE_MONTHS)[["rate"]].mean())
        .reset_index()
    )
    # When using groupby+apply, output columns can vary; normalize:
    if "rate" in calls_recent.columns:
        calls_recent = calls_recent.rename(columns={"rate": "calls_not_indexed_rate_recent"})
    else:
        # Fallback: rebuild via aggregation
        calls_recent = (
            hist_calls.groupby("department_id", as_index=False)
            .tail(RECENT_RATE_MONTHS)
            .groupby("department_id", as_index=False)["rate"].mean()
            .rename(columns={"rate": "calls_not_indexed_rate_recent"})
        )

monthly_adj = monthly_adj.merge(calls_recent, on="department_id", how="left")
monthly_adj["calls_not_indexed_rate_recent"] = safe_numeric(
    monthly_adj.get("calls_not_indexed_rate_recent", 0.0), 0.0
).clip(0, 5.0)

monthly_adj["calls_not_indexed_forecast"] = (
    safe_numeric(monthly_adj["forecast_monthly_dept"], 0.0)
    * monthly_adj["calls_not_indexed_rate_recent"]
)

# -----------------------------
# 9b) Repeats: recent ratio and forecast additional workload
# -----------------------------
repeats_recent = pd.DataFrame(columns=["department_id", "repeats_rate_recent"])

if isinstance(repeats_month, pd.DataFrame) and not repeats_month.empty:
    hist_rr = monthly_actuals.merge(repeats_month, on=["department_id", "month"], how="left")
    hist_rr["repeats_workload"] = safe_numeric(hist_rr.get("repeats_workload", 0.0), 0.0)
    hist_rr["actual_volume"] = safe_numeric(hist_rr["actual_volume"], 0.0)

    hist_rr["rate"] = 0.0
    nz = hist_rr["actual_volume"] > 0
    hist_rr.loc[nz, "rate"] = (hist_rr.loc[nz, "repeats_workload"] / hist_rr.loc[nz, "actual_volume"])

    hist_rr = hist_rr.sort_values(["department_id", "month"])
    repeats_recent = (
        hist_rr.groupby("department_id", as_index=False)
        .apply(lambda g: g.tail(RECENT_RATE_MONTHS)[["rate"]].mean())
        .reset_index()
    )
    if "rate" in repeats_recent.columns:
        repeats_recent = repeats_recent.rename(columns={"rate": "repeats_rate_recent"})
    else:
        repeats_recent = (
            hist_rr.groupby("department_id", as_index=False)
            .tail(RECENT_RATE_MONTHS)
            .groupby("department_id", as_index=False)["rate"].mean()
            .rename(columns={"rate": "repeats_rate_recent"})
        )

monthly_adj = monthly_adj.merge(repeats_recent, on="department_id", how="left")
monthly_adj["repeats_rate_recent"] = safe_numeric(
    monthly_adj.get("repeats_rate_recent", 0.0), 0.0
).clip(0, 5.0)

monthly_adj["repeats_forecast"] = (
    safe_numeric(monthly_adj["forecast_monthly_dept"], 0.0)
    * monthly_adj["repeats_rate_recent"]
)

# -----------------------------
# 9c) Inventory: forecast and attach
# -----------------------------
inventory_fc = pd.DataFrame(columns=["department_id", "month", "inventory_forecast"])

if isinstance(inventory_month, pd.DataFrame) and not inventory_month.empty:
    inv_fc_rows = []
    for dpt_id, g in inventory_month.groupby("department_id"):
        y = g.set_index("month")["inventory"].sort_index()
        y = safe_numeric(y, 0.0)
        fc = forecast_monthly_inventory_baseline(y, horizon_months=HORIZON_MONTHS)
        fc["department_id"] = int(dpt_id)
        inv_fc_rows.append(fc)

    inventory_fc = pd.concat(inv_fc_rows, ignore_index=True) if inv_fc_rows else inventory_fc
    inventory_fc["month"] = pd.to_datetime(inventory_fc["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

monthly_adj["month"] = pd.to_datetime(monthly_adj["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
inventory_month["month"] = pd.to_datetime(inventory_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

monthly_adj = monthly_adj.merge(
    inventory_month.rename(columns={"inventory": "inventory_actual"}),
    on=["department_id", "month"],
    how="left"
)

monthly_adj = monthly_adj.merge(
    inventory_fc[["department_id", "month", "inventory_forecast"]] if not inventory_fc.empty else inventory_fc,
    on=["department_id", "month"],
    how="left"
)

monthly_adj["inventory_final"] = monthly_adj["inventory_actual"]
mask = monthly_adj["inventory_final"].isna()
monthly_adj.loc[mask, "inventory_final"] = monthly_adj.loc[mask, "inventory_forecast"]
monthly_adj["inventory_final"] = safe_numeric(monthly_adj["inventory_final"], 0.0).clip(lower=0.0)

# ------------------------------------------------------------
# ✅ Capacity spine + ffill (ONLY ONCE) — aligned to monthly_adj months (includes future)
# ------------------------------------------------------------
dept_month_spine = monthly_adj[["department_id", "month"]].drop_duplicates().copy()

cap_prod_month = (
    dept_month_spine.merge(
        cap_prod_hist,
        on=["department_id", "month"],
        how="left"
    )
    .sort_values(["department_id", "month"])
)

cap_prod_month[["capacity_agents", "productivity_agents"]] = (
    cap_prod_month
    .groupby("department_id")[["capacity_agents", "productivity_agents"]]
    .ffill()
)

cap_prod_month[["capacity_agents", "productivity_agents"]] = (
    cap_prod_month[["capacity_agents", "productivity_agents"]].fillna(0.0)
)

cap_prod_month["days_in_month"] = cap_prod_month["month"].dt.to_period("M").dt.days_in_month

cap_prod_month = cap_prod_month.merge(
    prod_per_agent_day,
    on="department_id",
    how="left"
)

cap_prod_month["avg_cases_per_agent_day"] = pd.to_numeric(
    cap_prod_month["avg_cases_per_agent_day"], errors="coerce"
).fillna(0.0)

cap_prod_month["monthly_resolution_capacity"] = (
    cap_prod_month["avg_cases_per_agent_day"]
    * cap_prod_month["productivity_agents"]
    * cap_prod_month["days_in_month"]
)

print("Cap/prod months aligned to monthly_adj:", len(cap_prod_month))
print(cap_prod_month.tail())

# -----------------------------
# 10) Bias-based calibration (kept as your hardcoded table)
# -----------------------------
model_used_error_df = pd.DataFrame([
    {"vertical": "Hospitality", "department_id": 7, "department_name": "CS_PMSH_L1", "model_used": "STL", "backtest_months": 12, "mape_pct": 29.35, "wape_pct": 26.00, "bias_pct": -0.40},
    {"vertical": "Hospitality", "department_id": 8, "department_name": "CS_PMSP_CLOUD_L1", "model_used": "STL", "backtest_months": 12, "mape_pct": 43.80, "wape_pct": 39.12, "bias_pct": 12.64},
    {"vertical": "Hospitality", "department_id": 11, "department_name": "CS_PMSP_CLOUD_L2", "model_used": "STL", "backtest_months": 12, "mape_pct": 27.76, "wape_pct": 22.27, "bias_pct": -4.06},
    {"vertical": "Hospitality", "department_id": 23, "department_name": "CS_PMSP_FRANCE", "model_used": "STL", "backtest_months": 12, "mape_pct": 31.44, "wape_pct": 25.79, "bias_pct": -1.79},
    {"vertical": "Hospitality", "department_id": 5, "department_name": "CS_PMSP_INTEG", "model_used": "STL", "backtest_months": 12, "mape_pct": 20.88, "wape_pct": 17.68, "bias_pct": -2.70},
    {"vertical": "Hospitality", "department_id": 9, "department_name": "CS_PMSP_PREM_L1", "model_used": "STL", "backtest_months": 12, "mape_pct": 17.29, "wape_pct": 14.44, "bias_pct": -3.07},
    {"vertical": "Hospitality", "department_id": 10, "department_name": "CS_PMSP_PREM_L2", "model_used": "STL", "backtest_months": 12, "mape_pct": 28.77, "wape_pct": 23.23, "bias_pct": -1.12},
    {"vertical": "Partners", "department_id": 12, "department_name": "CS_PART_APAC", "model_used": "STL", "backtest_months": 12, "mape_pct": 20.88, "wape_pct": 20.99, "bias_pct": -5.14},
    {"vertical": "Partners", "department_id": 13, "department_name": "CS_PART_EMEA", "model_used": "STL", "backtest_months": 12, "mape_pct": 27.76, "wape_pct": 25.91, "bias_pct": 7.36},
    {"vertical": "Partners", "department_id": 14, "department_name": "CS_PART_LATAM", "model_used": "STL", "backtest_months": 12, "mape_pct": 26.60, "wape_pct": 21.38, "bias_pct": 5.29},
    {"vertical": "Partners", "department_id": 15, "department_name": "CS_PART_US", "model_used": "STL", "backtest_months": 12, "mape_pct": 40.03, "wape_pct": 30.97, "bias_pct": -9.47},
    {"vertical": "Payments", "department_id": 3, "department_name": "CA_PYAC", "model_used": "STL", "backtest_months": 12, "mape_pct": 18.63, "wape_pct": 18.70, "bias_pct": 1.28},
    {"vertical": "Payments", "department_id": 1, "department_name": "CS_GT3C_EU", "model_used": "STL", "backtest_months": 12, "mape_pct": 24.80, "wape_pct": 22.19, "bias_pct": 0.58},
    {"vertical": "Payments", "department_id": 18, "department_name": "Datatrans L2 Customer Support", "model_used": "STL", "backtest_months": 12, "mape_pct": 54.89, "wape_pct": 38.55, "bias_pct": -7.28},
    {"vertical": "Payments", "department_id": 2, "department_name": "L2 Customer Support", "model_used": "STL", "backtest_months": 12, "mape_pct": 28.56, "wape_pct": 24.46, "bias_pct": -1.36},
    {"vertical": "Payments", "department_id": 21, "department_name": "Specialist - L2 Customer Support", "model_used": "STL", "backtest_months": 12, "mape_pct": 42.25, "wape_pct": 41.25, "bias_pct": 17.26},
])

calib_from_bias = model_used_error_df[["department_id", "bias_pct"]].copy()
calib_from_bias["department_id"] = pd.to_numeric(calib_from_bias["department_id"], errors="coerce")
calib_from_bias["calib_factor"] = (1 - calib_from_bias["bias_pct"] / 100.0).clip(0.70, 1.30)

monthly_adj = monthly_adj.merge(calib_from_bias[["department_id", "calib_factor"]], on="department_id", how="left")
monthly_adj["calib_factor"] = monthly_adj["calib_factor"].fillna(1.0)

# Since Einstein deduction is disabled, define post_einstein columns as base forecast for compatibility
monthly_adj["forecast_monthly_dept_post_einstein"] = safe_numeric(monthly_adj["forecast_monthly_dept"], 0.0)
monthly_adj["forecast_p05_dept_post_einstein"] = safe_numeric(monthly_adj["forecast_p05_dept"], 0.0)
monthly_adj["forecast_p95_dept_post_einstein"] = safe_numeric(monthly_adj["forecast_p95_dept"], 0.0)

monthly_adj["forecast_monthly_dept_post_einstein_cal"] = monthly_adj["forecast_monthly_dept_post_einstein"] * monthly_adj["calib_factor"]
monthly_adj["forecast_p05_dept_post_einstein_cal"] = monthly_adj["forecast_p05_dept_post_einstein"] * monthly_adj["calib_factor"]
monthly_adj["forecast_p95_dept_post_einstein_cal"] = monthly_adj["forecast_p95_dept_post_einstein"] * monthly_adj["calib_factor"]

# -----------------------------
# 12) Build long_dept and final export frame (SAFE)
# -----------------------------
fc_board = monthly_adj.copy()

if "forecast_monthly_dept" not in fc_board.columns:
    raise KeyError("Missing column 'forecast_monthly_dept' in monthly_adj.")

fc_board["forecast_base"] = safe_numeric(fc_board["forecast_monthly_dept"], 0.0)

if "forecast_monthly_dept_post_einstein_cal" not in fc_board.columns:
    raise KeyError("Missing 'forecast_monthly_dept_post_einstein_cal' in monthly_adj.")

fc_board["forecast_human_cases_cal"] = safe_numeric(fc_board["forecast_monthly_dept_post_einstein_cal"], 0.0)

# Einstein rate is disabled -> keep 0.0
fc_board["einstein_rate_recent"] = 0.0
fc_board["einstein_solved_forecast"] = fc_board["forecast_base"] * fc_board["einstein_rate_recent"]

# Safety columns
for col in [
    "calls_not_indexed_rate_recent", "calls_not_indexed_forecast",
    "repeats_rate_recent", "repeats_forecast",
    "inventory_final"
]:
    if col not in fc_board.columns:
        fc_board[col] = 0.0
    fc_board[col] = safe_numeric(fc_board[col], 0.0)

# Month alignment
fc_board["month"] = pd.to_datetime(fc_board["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
monthly_actuals["month"] = pd.to_datetime(monthly_actuals["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

# Build long_dept core
long_dept = (
    fc_board[[
        "vertical", "department_id", "department_name", "month",
        "forecast_base",
        "forecast_human_cases_cal",
        "einstein_rate_recent", "einstein_solved_forecast",
        "calls_not_indexed_rate_recent", "calls_not_indexed_forecast",
        "repeats_rate_recent", "repeats_forecast",
        "inventory_final"
    ]]
    .merge(
        monthly_actuals[["vertical", "department_id", "department_name", "month", "actual_volume"]],
        on=["vertical", "department_id", "department_name", "month"],
        how="left"
    )
)

# Attach calls not indexed actual (history)
if isinstance(calls_ni_month, pd.DataFrame) and not calls_ni_month.empty:
    calls_ni_month["month"] = pd.to_datetime(calls_ni_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
    long_dept = long_dept.merge(
        calls_ni_month.rename(columns={"calls_not_indexed": "calls_not_indexed_actual"}),
        on=["department_id", "month"],
        how="left"
    )
else:
    long_dept["calls_not_indexed_actual"] = np.nan

# Attach repeats actual (history)
if isinstance(repeats_month, pd.DataFrame) and not repeats_month.empty:
    repeats_month["month"] = pd.to_datetime(repeats_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
    long_dept = long_dept.merge(
        repeats_month.rename(columns={"repeats_workload": "repeats_actual"}),
        on=["department_id", "month"],
        how="left"
    )
else:
    long_dept["repeats_actual"] = 0.0

# Attach capacity/productivity (agents)
if isinstance(cap_prod_month, pd.DataFrame) and not cap_prod_month.empty:
    cap_prod_month["month"] = pd.to_datetime(cap_prod_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
    long_dept = long_dept.merge(cap_prod_month, on=["department_id", "month"], how="left")
else:
    long_dept["capacity_agents"] = np.nan
    long_dept["productivity_agents"] = np.nan
    long_dept["monthly_resolution_capacity"] = np.nan
    long_dept["avg_cases_per_agent_day"] = np.nan
    long_dept["days_in_month"] = np.nan

# Numeric safety
num_cols = [
    "actual_volume",
    "forecast_base", "forecast_human_cases_cal",
    "einstein_rate_recent", "einstein_solved_forecast",
    "calls_not_indexed_actual", "calls_not_indexed_forecast",
    "repeats_actual", "repeats_forecast",
    "inventory_final",
    "capacity_agents", "productivity_agents",
    "monthly_resolution_capacity",
    "avg_cases_per_agent_day",
    "days_in_month",
]
for c in num_cols:
    if c in long_dept.columns:
        long_dept[c] = pd.to_numeric(long_dept[c], errors="coerce")

long_dept = (
    long_dept
    .sort_values(["vertical", "department_name", "month"])
    .reset_index(drop=True)
)

print("Built long_dept rows:", len(long_dept))

# -----------------------------
# 13) Final export table (capacity_forecast)
# -----------------------------
cap_wide = long_dept.copy()

cap_wide["Month"] = cap_wide["month"]
cap_wide["Vertical"] = cap_wide["vertical"]
cap_wide["Department_name"] = cap_wide["department_name"]

cap_wide["Actual Volume"] = cap_wide["actual_volume"]
cap_wide["Forecast (Cases)"] = cap_wide["forecast_base"]
cap_wide["Forecast after Einstein (Human cases)"] = cap_wide["forecast_human_cases_cal"]
cap_wide["Einstein solved forecast"] = cap_wide["einstein_solved_forecast"]

cap_wide["Calls not indexed (actual)"] = safe_numeric(cap_wide.get("calls_not_indexed_actual", 0.0), 0.0)
cap_wide["Calls not indexed (forecast)"] = safe_numeric(cap_wide.get("calls_not_indexed_forecast", 0.0), 0.0)

cap_wide["Repeats (actual)"] = safe_numeric(cap_wide.get("repeats_actual", 0.0), 0.0)
cap_wide["Repeats (forecast)"] = safe_numeric(cap_wide.get("repeats_forecast", 0.0), 0.0)

cap_wide["Inventory"] = safe_numeric(cap_wide.get("inventory_final", 0.0), 0.0)

# Agents
cap_wide["Productivity"] = pd.to_numeric(cap_wide.get("productivity_agents", np.nan), errors="coerce")
cap_wide["Capacity"] = pd.to_numeric(cap_wide.get("capacity_agents", np.nan), errors="coerce")

# Workloads
cap_wide["Actual Workload (Cases + calls not indexed + repeats)"] = (
    safe_numeric(cap_wide["Actual Volume"], 0.0)
    + cap_wide["Calls not indexed (actual)"]
    + cap_wide["Repeats (actual)"]
)

cap_wide["Workload Forecast (Humans + calls not indexed + repeats + inventory)"] = (
    safe_numeric(cap_wide["Forecast after Einstein (Human cases)"], 0.0)
    + cap_wide["Calls not indexed (forecast)"]
    + cap_wide["Repeats (forecast)"]
    + cap_wide["Inventory"]
)

# Backward-compatible (without inventory)
cap_wide["Workload Forecast (Humans + calls not indexed + repeats)"] = (
    safe_numeric(cap_wide["Forecast after Einstein (Human cases)"], 0.0)
    + cap_wide["Calls not indexed (forecast)"]
    + cap_wide["Repeats (forecast)"]
)

# Capacity in cases
cap_wide["Forecast Resolvable Cases (Capacity)"] = safe_numeric(
    cap_wide.get("monthly_resolution_capacity", 0.0), 0.0
)

cap_wide["Staffing Balance (Capacity - Workload)"] = (
    cap_wide["Forecast Resolvable Cases (Capacity)"]
    - cap_wide["Workload Forecast (Humans + calls not indexed + repeats + inventory)"]
)

cap_wide["Expected Workload vs Capacity"] = (
    cap_wide["Workload Forecast (Humans + calls not indexed + repeats + inventory)"]
    - cap_wide["Forecast Resolvable Cases (Capacity)"]
)

capacity_forecast_display = cap_wide[[
    "Month", "Vertical", "Department_name",
    "Actual Volume",
    "Forecast (Cases)",
    "Forecast after Einstein (Human cases)",
    "Einstein solved forecast",
    "Calls not indexed (actual)",
    "Calls not indexed (forecast)",
    "Repeats (actual)",
    "Repeats (forecast)",
    "Inventory",
    "Productivity",
    "Capacity",
    "Actual Workload (Cases + calls not indexed + repeats)",
    "Workload Forecast (Humans + calls not indexed + repeats + inventory)",
    "Forecast Resolvable Cases (Capacity)",
    "Staffing Balance (Capacity - Workload)",
    "Expected Workload vs Capacity",
    "Workload Forecast (Humans + calls not indexed + repeats)"
]].copy()

# -----------------------------
# Export
# -----------------------------
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as writer:
    monthly_adj.to_excel(writer, sheet_name="Monthly_Forecast_CAL", index=False)
    model_used_error_df.to_excel(writer, sheet_name="Model_Used_and_Error", index=False)
    capacity_forecast_display.to_excel(writer, sheet_name="capacity_forecast", index=False)

print("Export complete →", OUTPUT_XLSX)