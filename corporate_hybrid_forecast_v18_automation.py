# coding: utf-8
# corporate_hybrid_forecast_v18_automation.py
# Notes:
# - Reads inputs from Azure SQL (APP_FLOW.CustomerService.*)
# - Writes base output to CustomerService.capacity_forecast (existing Power BI table)
# - Writes extended metrics (shrinkage, productivity, FTE, risk heatmap) to CustomerService.capacity_forecast_risk
# - Requires env vars: SQL_SERVER, SQL_DATABASE, SQL_USER, SQL_PASSWORD

from pathlib import Path
import os
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from urllib.parse import quote_plus

# -----------------------------
# 1) Configuration
# -----------------------------
SQL_SERVER = os.getenv("SQL_SERVER", "wlms-sql-pay-appflow-dev.database.windows.net")
SQL_DATABASE = os.getenv("SQL_DATABASE", "APP_FLOW")
SQL_SCHEMA = os.getenv("SQL_SCHEMA", "CustomerService")
SQL_USER = os.getenv("SQL_USER", "CS_USER")
SQL_PASSWORD = os.getenv("SQL_PASSWORD")

SQL_ODBC_DRIVER = os.getenv("SQL_ODBC_DRIVER", "ODBC Driver 17 for SQL Server")
if not SQL_PASSWORD:
    raise EnvironmentError("Missing SQL_PASSWORD environment variable.")

HORIZON_MONTHS = int(os.getenv("HORIZON_MONTHS", "12"))
RECENT_RATE_MONTHS = int(os.getenv("RECENT_RATE_MONTHS", "3"))

# Capacity day basis: calendar|business
CAPACITY_DAY_BASIS = os.getenv("CAPACITY_DAY_BASIS", "calendar").lower()

# Work hours per day used to convert shrinkage time into % of capacity time
WORK_HOURS_PER_DAY = float(os.getenv("WORK_HOURS_PER_DAY", "8"))

# Business scope
VERTICALS_TARGET = ["Payments", "Hospitality", "Onboarding"]
ONBOARDING_DEPARTMENTS = [d.strip() for d in os.getenv("ONBOARDING_DEPARTMENTS", "OB_CUST").split(",") if d.strip()]
EXCLUDE_DEPARTMENT_NAME_TOKENS = ["PROJ", "DIST", "KEY", "PROXIMIS"]

# Optional Excel output
EXPORT_EXCEL = os.getenv("EXPORT_EXCEL", "true").lower() == "true"
OUTPUT_DIR = os.getenv("CAPACITY_OUTPUT_DIR", str(Path.cwd() / "outputs"))
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "capacity_forecast_v18_SQL.xlsx")

# Target tables
TARGET_TABLE = os.getenv("TARGET_TABLE", "capacity_forecast")                 # base table (existing)
TARGET_TABLE_RISK = os.getenv("TARGET_TABLE_RISK", "capacity_forecast_risk")  # extended metrics for heatmap

print("OUTPUT_XLSX →", OUTPUT_XLSX)
print("SQL_SERVER →", SQL_SERVER)
print("SQL_DB →", SQL_DATABASE)
print("SQL_SCHEMA →", SQL_SCHEMA)
print("TARGET_TABLE →", f"{SQL_SCHEMA}.{TARGET_TABLE}")
print("TARGET_TABLE_RISK →", f"{SQL_SCHEMA}.{TARGET_TABLE_RISK}")

# -----------------------------
# 2) SQL Engine
# -----------------------------
odbc_params = (
    f"DRIVER={{{SQL_ODBC_DRIVER}}};"
    f"SERVER={SQL_SERVER};"
    f"PORT=1433;"
    f"DATABASE={SQL_DATABASE};"
    f"UID={SQL_USER};"
    f"PWD={SQL_PASSWORD};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)

engine = sa.create_engine(
    "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc_params),
    fast_executemany=True
)

with engine.connect() as conn:
    conn.execute(text("SELECT 1"))
print("✅ SQL connection OK")

# -----------------------------
# 3) Helpers
# -----------------------------
def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df

def safe_numeric(s, default=0.0):
    if isinstance(s, (int, float, np.number)):
        return default if pd.isna(s) else s
    return pd.to_numeric(s, errors="coerce").fillna(default)

def to_month_start(dt_series):
    s = pd.to_datetime(dt_series, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp(how="start")

def read_table(sql: str) -> pd.DataFrame:
    return std_cols(pd.read_sql(sql, engine))

def seasonal_naive_forecast(y: pd.Series, horizon_days: int, season_len: int = 7):
    y = y.asfreq("D").fillna(0.0).astype(float)
    if len(y) == 0:
        return np.zeros(horizon_days, dtype=float)
    if len(y) < season_len:
        return np.full(horizon_days, float(y.iloc[-1]), dtype=float)
    pattern = y.iloc[-season_len:].to_numpy(dtype=float)
    reps = int(np.ceil(horizon_days / season_len))
    return np.tile(pattern, reps)[:horizon_days]

def forecast_daily(y: pd.Series, horizon_days: int = 365) -> pd.DataFrame:
    """
    Lightweight forecast:
    - p50: weekly seasonal naive
    - p05/p95: +/- 1.645 * std(residuals last 90 days)
    """
    y = y.asfreq("D").fillna(0.0).astype(float)
    last_date = y.index.max()
    future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    p50 = seasonal_naive_forecast(y, horizon_days=horizon_days, season_len=7)

    window = min(90, len(y))
    if window >= 14:
        hist_tail = y.iloc[-window:].copy()
        # crude residuals based on repeating last-week pattern
        backcast = seasonal_naive_forecast(y.iloc[:-7] if len(y) > 7 else y, horizon_days=window, season_len=7)
        resid = (hist_tail.to_numpy(dtype=float) - backcast)
        sigma = float(np.nanstd(resid)) if np.isfinite(np.nanstd(resid)) else 0.0
    else:
        sigma = 0.0

    delta = 1.645 * sigma
    p05 = np.clip(p50 - delta, 0, None)
    p95 = np.clip(p50 + delta, 0, None)

    return pd.DataFrame({"date": future_idx, "p50": p50, "p05": p05, "p95": p95})

def recent_rate(df_hist: pd.DataFrame, value_col: str, base_col: str, months: int) -> pd.DataFrame:
    """
    rate = sum(value) / sum(base) over last N months, per department_id
    """
    if df_hist.empty:
        return pd.DataFrame(columns=["department_id", f"{value_col}_rate_recent"])

    df = df_hist.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
    max_month = df["month"].max()
    min_month = (max_month - pd.offsets.MonthBegin(months))
    df = df[df["month"] >= min_month].copy()

    df[value_col] = safe_numeric(df[value_col], 0.0)
    df[base_col] = safe_numeric(df[base_col], 0.0)

    g = df.groupby("department_id", as_index=False).agg(
        v_sum=(value_col, "sum"),
        b_sum=(base_col, "sum")
    )
    g[f"{value_col}_rate_recent"] = np.where(g["b_sum"] > 0, g["v_sum"] / g["b_sum"], 0.0)
    return g[["department_id", f"{value_col}_rate_recent"]]

def month_day_multiplier(month_start: pd.Timestamp, basis: str) -> int:
    ms = pd.Timestamp(month_start).normalize()
    me = (ms + pd.offsets.MonthEnd(1)).normalize()
    if basis == "business":
        # np.busday_count excludes end date, add 1 day to include it
        return int(np.busday_count(ms.date(), (me + pd.Timedelta(days=1)).date()))
    return int((me - ms).days + 1)

def ensure_risk_table(conn):
    """
    Create extended table if it does not exist.
    Kept minimal types and names to be Power BI friendly.
    """
    ddl = f"""
    IF OBJECT_ID('{SQL_SCHEMA}.{TARGET_TABLE_RISK}', 'U') IS NULL
    BEGIN
        CREATE TABLE [{SQL_SCHEMA}].[{TARGET_TABLE_RISK}] (
            run_date date NULL,
            [Month] date NULL,
            Vertical varchar(50) NULL,
            Department_name varchar(100) NULL,

            -- Demand / forecast pieces
            [Actual Incoming] int NULL,
            [Forecast (Incoming)] int NULL,
            [Calls not indexed (forecast)] int NULL,
            [Repeats (forecast)] int NULL,
            Inventory int NULL,

            -- Produced / productivity
            [Actual Produced] int NULL,
            [Productivity per FTE per day] decimal(18,4) NULL,

            -- Targets / capacity
            [Target per FTE per day] decimal(18,4) NULL,
            [Capacity Gross] int NULL,
            [Shrinkage %] decimal(10,6) NULL,
            [Headcount after shrinkage] decimal(18,4) NULL,
            [Capacity Net after shrinkage] int NULL,

            -- FTE needed / gaps
            [FTE actual] decimal(18,4) NULL,
            [FTE required] decimal(18,4) NULL,
            [FTE gap] decimal(18,4) NULL,

            -- Risk & heatmap
            [Workload Forecast Total] int NULL,
            [Workload minus Net Capacity] int NULL,
            [RiskScore] int NULL,
            [RiskBand] varchar(20) NULL,
            [Status] varchar(20) NULL,

            inserted_at datetime2(0) NOT NULL DEFAULT SYSDATETIME()
        );
    END
    """
    conn.execute(text(ddl))

# -----------------------------
# 4) Load inputs from SQL
# -----------------------------
# 4.1 Incoming (Actual Volume) -> DAILY_Incoming_Total.total_incoming
incoming_raw = read_table(f"""
SELECT
    [Date] AS [date],
    department_id,
    total_incoming AS incoming_total
FROM [{SQL_SCHEMA}].[DAILY_Incoming_Total]
WHERE department_id IS NOT NULL
  AND [Date] IS NOT NULL
""")

dept_map = read_table(f"""
SELECT
    department_id,
    department_name,
    vertical
FROM [{SQL_SCHEMA}].[department]
""")

incoming = incoming_raw.copy()
incoming["date"] = pd.to_datetime(incoming["date"], errors="coerce")
incoming["month"] = to_month_start(incoming["date"])
incoming["department_id"] = pd.to_numeric(incoming["department_id"], errors="coerce").astype("Int64")
incoming["incoming_total"] = pd.to_numeric(incoming["incoming_total"], errors="coerce").fillna(0)

dept_map["department_id"] = pd.to_numeric(dept_map["department_id"], errors="coerce").astype("Int64")
incoming = incoming.merge(dept_map, on="department_id", how="left")

# Scope filter
incoming = incoming[incoming["vertical"].isin(VERTICALS_TARGET)].copy()
incoming = incoming[
    (incoming["vertical"] != "Onboarding") |
    (incoming["department_name"].isin(ONBOARDING_DEPARTMENTS))
].copy()

mask_excl = pd.Series(False, index=incoming.index)
for tok in EXCLUDE_DEPARTMENT_NAME_TOKENS:
    mask_excl = mask_excl | incoming["department_name"].astype(str).str.upper().str.contains(tok.upper(), na=False)
incoming = incoming.loc[~mask_excl].copy()

print("Loaded incoming rows =", len(incoming))

# 4.2 Calls Not Indexed (monthly)
calls_ni_month = read_table(f"""
SELECT
    department_id,
    DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1) AS [month],
    COUNT(*) AS calls_not_indexed
FROM [{SQL_SCHEMA}].[calls_not_indexed]
WHERE department_id IS NOT NULL
  AND [Date] IS NOT NULL
GROUP BY department_id, DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1)
""")
calls_ni_month["month"] = pd.to_datetime(calls_ni_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

# 4.3 Repeats (monthly)
repeats_month = read_table(f"""
SELECT
    department_id,
    DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1) AS [month],
    COUNT(DISTINCT case_number) AS repeats_workload
FROM [{SQL_SCHEMA}].[DAILY_RR_new]
WHERE department_id IS NOT NULL
  AND [Date] IS NOT NULL
GROUP BY department_id, DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1)
""")
repeats_month["month"] = pd.to_datetime(repeats_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

# 4.4 Inventory (monthly)
inventory_month = read_table(f"""
SELECT
    department_id,
    [month],
    inventory_cases AS inventory
FROM [{SQL_SCHEMA}].[vw_capacity_inventory_monthly]
""")
inventory_month["month"] = pd.to_datetime(inventory_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

# -----------------------------
# 4.5 Capacity targets (daily per agent)
# -----------------------------
targets = read_table(f"""
SELECT
    agent_id,
    department_id,
    item_target
FROM [{SQL_SCHEMA}].[agent_item_target]
WHERE department_id IS NOT NULL
""")

targets["department_id"] = pd.to_numeric(targets["department_id"], errors="coerce").astype("Int64")
targets["agent_id"] = pd.to_numeric(targets["agent_id"], errors="coerce").astype("Int64")
targets["item_target"] = pd.to_numeric(targets["item_target"], errors="coerce").fillna(0.0)

# Department gross capacity target per day
dept_target_day = targets.groupby("department_id", as_index=False)\
    .agg(capacity_target_day=("item_target", "sum"))

# Average target per FTE/day
dept_target_avg = targets.groupby("department_id", as_index=False)\
    .agg(target_per_fte_day=("item_target", "mean"))


# -----------------------------
# 4.6 Productivity (IMPORTANT - must come BEFORE FTE calc)
# -----------------------------
prod_daily = read_table(f"""
SELECT
    [Date] AS [date],
    agent_id,
    department_id,
    department_name,
    prod_total_model
FROM [{SQL_SCHEMA}].[agent_productivity_new_daily]
WHERE department_id IS NOT NULL
  AND agent_id IS NOT NULL
  AND [Date] IS NOT NULL
""")

prod_daily["date"] = pd.to_datetime(prod_daily["date"], errors="coerce")
prod_daily["month"] = to_month_start(prod_daily["date"])
prod_daily["department_id"] = pd.to_numeric(prod_daily["department_id"], errors="coerce").astype("Int64")
prod_daily["agent_id"] = pd.to_numeric(prod_daily["agent_id"], errors="coerce").astype("Int64")
prod_daily["prod_total_model"] = pd.to_numeric(prod_daily["prod_total_model"], errors="coerce").fillna(0)


# ✅ PRODUCED WORKLOAD
prod_month = (
    prod_daily.groupby(["department_id", "month"], as_index=False)
    .agg(actual_produced=("prod_total_model", "sum"))
)


# ✅ FTE REAL (FIX DEL PROBLEMA)
active_daily = (
    prod_daily.groupby(["department_id", "date"], as_index=False)
    .agg(active_agents=("agent_id", "nunique"))
)

dept_fte_month = (
    active_daily
    .assign(month=to_month_start(active_daily["date"]))
    .groupby(["department_id", "month"], as_index=False)
    .agg(fte_actual=("active_agents", "mean"))
)


# -----------------------------
# 4.7 Shrinkage
# -----------------------------
shrink = read_table(f"""
SELECT
    [Date] AS [date],
    agent_id,
    shrinkage_time
FROM [{SQL_SCHEMA}].[shrinkage]
WHERE agent_id IS NOT NULL
  AND [Date] IS NOT NULL
""")

shrink["date"] = pd.to_datetime(shrink["date"], errors="coerce")
shrink["month"] = to_month_start(shrink["date"])
shrink["agent_id"] = pd.to_numeric(shrink["agent_id"], errors="coerce").astype("Int64")
shrink["shrinkage_time"] = pd.to_numeric(shrink["shrinkage_time"], errors="coerce").fillna(0)

# Map agent → department
agent_dim = read_table(f"""
SELECT
    agent_id,
    department_id
FROM [{SQL_SCHEMA}].[agent]
WHERE agent_id IS NOT NULL
""")

agent_dim["agent_id"] = pd.to_numeric(agent_dim["agent_id"], errors="coerce").astype("Int64")
agent_dim["department_id"] = pd.to_numeric(agent_dim["department_id"], errors="coerce").astype("Int64")

shrink = shrink.merge(agent_dim, on="agent_id", how="left")
shrink = shrink[shrink["department_id"].notna()].copy()

shrink_month = (
    shrink.groupby(["department_id", "month"], as_index=False)
    .agg(shrinkage_seconds=("shrinkage_time", "sum"))
)

shrink_month["shrinkage_hours"] = shrink_month["shrinkage_seconds"] / 3600.0

# -----------------------------
# 5) Monthly actual incoming (Actual Volume)
# -----------------------------
monthly_actuals = (
    incoming.groupby(["vertical", "department_id", "department_name", "month"], as_index=False)
    .agg(actual_incoming=("incoming_total", "sum"))
)
monthly_actuals["actual_incoming"] = safe_numeric(monthly_actuals["actual_incoming"], 0.0).clip(lower=0.0)

# -----------------------------
# 6) Forecast incoming (daily -> monthly)
# -----------------------------
daily_hist = (
    incoming.groupby(["vertical", "department_id", "department_name", "date"], as_index=False)
    .agg(y=("incoming_total", "sum"))
)
daily_hist["date"] = pd.to_datetime(daily_hist["date"], errors="coerce")
daily_hist["y"] = safe_numeric(daily_hist["y"], 0.0).clip(lower=0.0)

daily_fc_rows = []
for (vertical, dept_id, dept_name), g in daily_hist.groupby(["vertical", "department_id", "department_name"]):
    g = g.sort_values("date")
    y = g.set_index("date")["y"].asfreq("D").fillna(0.0)
    fc = forecast_daily(y, horizon_days=365)
    fc["vertical"] = vertical
    fc["department_id"] = dept_id
    fc["department_name"] = dept_name
    daily_fc_rows.append(fc)

daily_fc = pd.concat(daily_fc_rows, ignore_index=True) if daily_fc_rows else pd.DataFrame(
    columns=["date", "p50", "p05", "p95", "vertical", "department_id", "department_name"]
)
daily_fc["month"] = to_month_start(daily_fc["date"])

monthly_fc_only = (
    daily_fc.groupby(["vertical", "department_id", "department_name", "month"], as_index=False)
    .agg(
        forecast_monthly_incoming=("p50", "sum"),
        forecast_p05=("p05", "sum"),
        forecast_p95=("p95", "sum"),
    )
)

# Create month spine (history + future)
last_hist_month = monthly_actuals["month"].max()
future_months = pd.date_range(last_hist_month + pd.offsets.MonthBegin(1), periods=HORIZON_MONTHS, freq="MS")
all_months = pd.DatetimeIndex(sorted(set(monthly_actuals["month"]).union(set(future_months))))

dept_dim = monthly_actuals[["vertical", "department_id", "department_name"]].drop_duplicates()
spine = dept_dim.assign(_k=1).merge(pd.DataFrame({"month": all_months, "_k": 1}), on="_k", how="outer").drop(columns=["_k"])

monthly_fc_raw = (
    spine.merge(
        monthly_actuals[["vertical", "department_id", "department_name", "month", "actual_incoming"]],
        on=["vertical", "department_id", "department_name", "month"],
        how="left",
    )
    .merge(
        monthly_fc_only,
        on=["vertical", "department_id", "department_name", "month"],
        how="left",
    )
)

# Fill history with actual incoming
mask_hist = monthly_fc_raw["actual_incoming"].notna()
for c in ["forecast_monthly_incoming", "forecast_p05", "forecast_p95"]:
    monthly_fc_raw[c] = safe_numeric(monthly_fc_raw.get(c, 0.0), 0.0)
    monthly_fc_raw.loc[mask_hist, c] = safe_numeric(monthly_fc_raw.loc[mask_hist, "actual_incoming"], 0.0)

monthly_adj = monthly_fc_raw.copy()
monthly_adj["actual_incoming"] = safe_numeric(monthly_adj["actual_incoming"], 0.0).clip(lower=0.0)

# -----------------------------
# 7) Recent ratios (Calls NI / Repeats) applied to incoming forecast
# -----------------------------
hist_calls = monthly_actuals.merge(calls_ni_month, on=["department_id", "month"], how="left")
hist_calls["calls_not_indexed"] = safe_numeric(hist_calls.get("calls_not_indexed", 0.0), 0.0)
calls_rate = recent_rate(hist_calls, "calls_not_indexed", "actual_incoming", RECENT_RATE_MONTHS)

hist_rr = monthly_actuals.merge(repeats_month, on=["department_id", "month"], how="left")
hist_rr["repeats_workload"] = safe_numeric(hist_rr.get("repeats_workload", 0.0), 0.0)
repeats_rate = recent_rate(hist_rr, "repeats_workload", "actual_incoming", RECENT_RATE_MONTHS)

monthly_adj = monthly_adj.merge(calls_rate, on="department_id", how="left")
monthly_adj = monthly_adj.merge(repeats_rate, on="department_id", how="left")

monthly_adj["calls_not_indexed_rate_recent"] = safe_numeric(monthly_adj.get("calls_not_indexed_rate_recent", 0.0), 0.0).clip(0, 5.0)
monthly_adj["repeats_workload_rate_recent"] = safe_numeric(monthly_adj.get("repeats_workload_rate_recent", 0.0), 0.0).clip(0, 5.0)

monthly_adj["calls_not_indexed_forecast"] = safe_numeric(monthly_adj["forecast_monthly_incoming"], 0.0) * monthly_adj["calls_not_indexed_rate_recent"]
monthly_adj["repeats_forecast"] = safe_numeric(monthly_adj["forecast_monthly_incoming"], 0.0) * monthly_adj["repeats_workload_rate_recent"]

# -----------------------------
# 8) Inventory
# -----------------------------
inventory_month["inventory"] = safe_numeric(inventory_month["inventory"], 0.0)
monthly_adj = monthly_adj.merge(
    inventory_month.rename(columns={"inventory": "inventory_actual"}),
    on=["department_id", "month"],
    how="left",
)
monthly_adj["inventory_final"] = safe_numeric(monthly_adj["inventory_actual"], 0.0)

# -----------------------------
# 9) Days multiplier (calendar or business)
# -----------------------------
month_multiplier = pd.DataFrame({"month": all_months})
month_multiplier["days_multiplier"] = month_multiplier["month"].apply(lambda m: month_day_multiplier(m, CAPACITY_DAY_BASIS))

monthly_adj = monthly_adj.merge(month_multiplier, on="month", how="left")
monthly_adj["days_multiplier"] = pd.to_numeric(monthly_adj["days_multiplier"], errors="coerce").fillna(0).astype(int)

# -----------------------------
# 10) Capacity (gross) from targets
# -----------------------------
monthly_adj = monthly_adj.merge(dept_target_day, on="department_id", how="left")
monthly_adj["capacity_target_day"] = safe_numeric(monthly_adj.get("capacity_target_day", 0.0), 0.0)

monthly_adj["capacity_gross"] = np.round(
    monthly_adj["capacity_target_day"] * monthly_adj["days_multiplier"]
).astype("Int64")

# -----------------------------
# 11) Productivity (actual produced) + productivity per FTE per day
# -----------------------------
monthly_adj = monthly_adj.merge(prod_month, on=["department_id", "month"], how="left")
monthly_adj["actual_produced"] = safe_numeric(monthly_adj.get("actual_produced", 0.0), 0.0).round(0).astype("Int64")


monthly_adj = monthly_adj.merge(
    dept_fte_month,
    on=["department_id", "month"],
    how="left"
)

monthly_adj["fte_actual"] = safe_numeric(monthly_adj.get("fte_actual", 0.0), 0.0)

monthly_adj["productivity_per_fte_day"] = np.where(
    (monthly_adj["fte_actual"] > 0) & (monthly_adj["days_multiplier"] > 0),
    monthly_adj["actual_produced"] / (monthly_adj["fte_actual"] * monthly_adj["days_multiplier"]),
    0.0
)

# -----------------------------
# 12) Shrinkage % (monthly)
# -----------------------------
monthly_adj = monthly_adj.merge(shrink_month[["department_id", "month", "shrinkage_hours"]], on=["department_id", "month"], how="left")
monthly_adj["shrinkage_hours"] = safe_numeric(monthly_adj.get("shrinkage_hours", 0.0), 0.0)

# shrinkage_pct = shrinkage_hours / (fte_actual * work_hours_per_day * days_in_month)
monthly_adj["shrinkage_pct"] = np.where(
    (monthly_adj["fte_actual"] > 0) & (monthly_adj["days_multiplier"] > 0),
    monthly_adj["shrinkage_hours"] / (monthly_adj["fte_actual"] * WORK_HOURS_PER_DAY * monthly_adj["days_multiplier"]),
    0.0
)

# Keep shrinkage in a sensible range
monthly_adj["shrinkage_pct"] = pd.to_numeric(monthly_adj["shrinkage_pct"], errors="coerce").fillna(0.0).clip(0.0, 0.90)

monthly_adj["headcount_after_shrinkage"] = monthly_adj["fte_actual"] * (1.0 - monthly_adj["shrinkage_pct"])

# -----------------------------
# 13) Net capacity after shrinkage (for risk & staffing)
# -----------------------------
monthly_adj["capacity_net"] = np.round(
    safe_numeric(monthly_adj["capacity_gross"], 0.0) * (1.0 - monthly_adj["shrinkage_pct"])
).astype("Int64")

# -----------------------------
# 14) FTE required (net, corrected) and alerts
# -----------------------------
monthly_adj = monthly_adj.merge(dept_target_avg, on="department_id", how="left")
monthly_adj["target_per_fte_day"] = safe_numeric(monthly_adj.get("target_per_fte_day", 0.0), 0.0)

monthly_adj["target_per_fte_month_net"] = monthly_adj["target_per_fte_day"] * monthly_adj["days_multiplier"] * (1.0 - monthly_adj["shrinkage_pct"])

monthly_adj["workload_forecast_total"] = (
    safe_numeric(monthly_adj["forecast_monthly_incoming"], 0.0)
    + safe_numeric(monthly_adj["calls_not_indexed_forecast"], 0.0)
    + safe_numeric(monthly_adj["repeats_forecast"], 0.0)
    + safe_numeric(monthly_adj["inventory_final"], 0.0)
).round(0).astype("Int64")

monthly_adj["fte_required"] = np.where(
    monthly_adj["target_per_fte_month_net"] > 0,
    safe_numeric(monthly_adj["workload_forecast_total"], 0.0) / monthly_adj["target_per_fte_month_net"],
    0.0
)

# Keep as decimal to avoid losing precision in Power BI; we also provide a ceiling-based variant implicitly via RiskScore/gap.
monthly_adj["fte_required"] = pd.to_numeric(monthly_adj["fte_required"], errors="coerce").fillna(0.0)

monthly_adj["fte_gap"] = monthly_adj["fte_required"] - monthly_adj["headcount_after_shrinkage"]

def classify_status(row):
    if row["fte_gap"] > 0.5:
        return "UNDER_CAPACITY"
    elif row["fte_gap"] < -2.0:
        return "OVER_CAPACITY"
    return "BALANCED"

monthly_adj["status"] = monthly_adj.apply(classify_status, axis=1)

# -----------------------------
# 15) RiskScore (numeric) + RiskBand (for Heatmap)
# -----------------------------
# Workload minus net capacity
monthly_adj["workload_minus_net_capacity"] = (
    safe_numeric(monthly_adj["workload_forecast_total"], 0.0) - safe_numeric(monthly_adj["capacity_net"], 0.0)
).round(0).astype("Int64")

# Normalized components
workload_ratio = np.where(
    safe_numeric(monthly_adj["workload_forecast_total"], 0.0) > 0,
    np.clip(safe_numeric(monthly_adj["workload_minus_net_capacity"], 0.0) / safe_numeric(monthly_adj["workload_forecast_total"], 1.0), 0, 1),
    0.0
)

fte_ratio = np.where(
    monthly_adj["fte_required"] > 0,
    np.clip(np.maximum(monthly_adj["fte_gap"], 0.0) / monthly_adj["fte_required"], 0, 1),
    0.0
)

# Weighted risk score in [0..100]
monthly_adj["risk_score"] = np.round(100.0 * (0.65 * workload_ratio + 0.35 * fte_ratio)).astype(int)

def risk_band(score: int) -> str:
    if score >= 60:
        return "RED"
    if score >= 30:
        return "AMBER"
    return "GREEN"

monthly_adj["risk_band"] = monthly_adj["risk_score"].apply(risk_band)

# -----------------------------
# 16) Base output to CustomerService.capacity_forecast (keep existing schema)
# -----------------------------
cap = monthly_adj.copy()
cap["Month"] = cap["month"]
cap["Vertical"] = cap["vertical"]
cap["Department_name"] = cap["department_name"]

# Column names must match your SQL table exactly
cap["Actual Volume"] = safe_numeric(cap["actual_incoming"], 0.0).round(0).astype("Int64")
cap["Forecast (Cases)"] = safe_numeric(cap["forecast_monthly_incoming"], 0.0).round(0).astype("Int64")
cap["Calls not indexed (forecast)"] = safe_numeric(cap["calls_not_indexed_forecast"], 0.0).round(0).astype("Int64")
cap["Repeats (forecast)"] = safe_numeric(cap["repeats_forecast"], 0.0).round(0).astype("Int64")
cap["Capacity"] = safe_numeric(cap["capacity_gross"], 0.0).round(0).astype("Int64")
cap["Inventory"] = safe_numeric(cap["inventory_final"], 0.0).round(0).astype("Int64")

cap["Actual Workload (Cases + calls not indexed + repeats)"] = cap["Actual Volume"]
cap["Workload Forecast (Humans + calls not indexed + repeats )"] = (
    safe_numeric(cap["Forecast (Cases)"], 0.0)
    + safe_numeric(cap["Calls not indexed (forecast)"], 0.0)
    + safe_numeric(cap["Repeats (forecast)"], 0.0)
).round(0).astype("Int64")

RUN_DATE = pd.Timestamp.today().normalize()
cap["run_date"] = RUN_DATE

capacity_forecast_display = cap[[
    "run_date",
    "Month",
    "Vertical",
    "Department_name",
    "Actual Volume",
    "Forecast (Cases)",
    "Calls not indexed (forecast)",
    "Repeats (forecast)",
    "Capacity",
    "Inventory",
    "Actual Workload (Cases + calls not indexed + repeats)",
    "Workload Forecast (Humans + calls not indexed + repeats )",
]].copy()

# -----------------------------
# 17) Extended output for Heatmap -> CustomerService.capacity_forecast_risk
# -----------------------------
risk_out = monthly_adj.copy()
risk_out["Month"] = risk_out["month"]
risk_out["Vertical"] = risk_out["vertical"]
risk_out["Department_name"] = risk_out["department_name"]
risk_out["run_date"] = RUN_DATE

risk_out_df = pd.DataFrame({
    "run_date": risk_out["run_date"],
    "Month": risk_out["Month"],
    "Vertical": risk_out["Vertical"],
    "Department_name": risk_out["Department_name"],

    "Actual Incoming": safe_numeric(risk_out["actual_incoming"], 0.0).round(0).astype("Int64"),
    "Forecast (Incoming)": safe_numeric(risk_out["forecast_monthly_incoming"], 0.0).round(0).astype("Int64"),
    "Calls not indexed (forecast)": safe_numeric(risk_out["calls_not_indexed_forecast"], 0.0).round(0).astype("Int64"),
    "Repeats (forecast)": safe_numeric(risk_out["repeats_forecast"], 0.0).round(0).astype("Int64"),
    "Inventory": safe_numeric(risk_out["inventory_final"], 0.0).round(0).astype("Int64"),

    "Actual Produced": safe_numeric(risk_out["actual_produced"], 0.0).round(0).astype("Int64"),
    "Productivity per FTE per day": pd.to_numeric(risk_out["productivity_per_fte_day"], errors="coerce").fillna(0.0),

    "Target per FTE per day": pd.to_numeric(risk_out["target_per_fte_day"], errors="coerce").fillna(0.0),
    "Capacity Gross": safe_numeric(risk_out["capacity_gross"], 0.0).round(0).astype("Int64"),
    "Shrinkage %": pd.to_numeric(risk_out["shrinkage_pct"], errors="coerce").fillna(0.0),
    "Headcount after shrinkage": pd.to_numeric(risk_out["headcount_after_shrinkage"], errors="coerce").fillna(0.0),
    "Capacity Net after shrinkage": safe_numeric(risk_out["capacity_net"], 0.0).round(0).astype("Int64"),

    "FTE actual": pd.to_numeric(risk_out["fte_actual"], errors="coerce").fillna(0.0),
    "FTE required": pd.to_numeric(risk_out["fte_required"], errors="coerce").fillna(0.0),
    "FTE gap": pd.to_numeric(risk_out["fte_gap"], errors="coerce").fillna(0.0),

    "Workload Forecast Total": safe_numeric(risk_out["workload_forecast_total"], 0.0).round(0).astype("Int64"),
    "Workload minus Net Capacity": safe_numeric(risk_out["workload_minus_net_capacity"], 0.0).round(0).astype("Int64"),
    "RiskScore": pd.to_numeric(risk_out["risk_score"], errors="coerce").fillna(0).astype(int),
    "RiskBand": risk_out["risk_band"].astype(str),
    "Status": risk_out["status"].astype(str),
})

# -----------------------------
# 18) Optional Excel export
# -----------------------------
if EXPORT_EXCEL:
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as writer:
        monthly_adj.to_excel(writer, sheet_name="Monthly_Forecast_CAL", index=False)
        capacity_forecast_display.to_excel(writer, sheet_name="capacity_forecast_base", index=False)
        risk_out_df.to_excel(writer, sheet_name="capacity_forecast_risk", index=False)

# -----------------------------
# 19) SQL export (FULL OVERWRITE with TRUNCATE)
# -----------------------------
with engine.begin() as conn:

    # Base table - FULL overwrite
    conn.execute(
        text(f"TRUNCATE TABLE [{SQL_SCHEMA}].[{TARGET_TABLE}]")
    )

    # Risk table - ensure exists + FULL overwrite
    ensure_risk_table(conn)

    conn.execute(
        text(f"TRUNCATE TABLE [{SQL_SCHEMA}].[{TARGET_TABLE_RISK}]")
    )

# Insert fresh data (clean state)
capacity_forecast_display.to_sql(
    TARGET_TABLE,
    engine,
    schema=SQL_SCHEMA,
    if_exists="append",
    index=False,
)

risk_out_df.to_sql(
    TARGET_TABLE_RISK,
    engine,
    schema=SQL_SCHEMA,
    if_exists="append",
    index=False,
)

# -----------------------------
# Verification
# -----------------------------
with engine.connect() as conn:

    inserted_base = conn.execute(
        text(f"SELECT COUNT(*) FROM [{SQL_SCHEMA}].[{TARGET_TABLE}]")
    ).scalar()

    inserted_risk = conn.execute(
        text(f"SELECT COUNT(*) FROM [{SQL_SCHEMA}].[{TARGET_TABLE_RISK}]")
    ).scalar()

if inserted_base == 0:
    raise RuntimeError("Base export finished but 0 rows found")

if inserted_risk == 0:
    raise RuntimeError("Risk export finished but 0 rows found")

print(f"SQL export OK → base rows={inserted_base}, risk rows={inserted_risk}")
print("Done.")