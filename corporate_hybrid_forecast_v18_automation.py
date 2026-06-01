# coding: utf-8
# corporate_hybrid_forecast_v18_automation.py
# Production-ready version:
# - Robust SQLAlchemy engine (pool_pre_ping + recycle)
# - TRUNCATE + INSERT in separate transactions
# - Safe inserts: method=None + small chunksize
# - Automatic retry on transient network/ODBC failures (08S01/10054)
# - Standard internal naming + explicit mapping to existing SQL column names

from __future__ import annotations

from pathlib import Path
import os
import time
import random
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, DBAPIError
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

HORIZON_MONTHS = int(os.getenv("HORIZON_MONTHS", "12"))
RECENT_RATE_MONTHS = int(os.getenv("RECENT_RATE_MONTHS", "3"))                # calls/repeats ratios
RECENT_ACTIVE_RATIO_MONTHS = int(os.getenv("RECENT_ACTIVE_RATIO_MONTHS", "3"))# active FTE projection
RECENT_SHRINK_MONTHS = int(os.getenv("RECENT_SHRINK_MONTHS", "3"))            # shrinkage projection

CAPACITY_DAY_BASIS = os.getenv("CAPACITY_DAY_BASIS", "calendar").lower()      # "calendar" or "business"
WORK_HOURS_PER_DAY = float(os.getenv("WORK_HOURS_PER_DAY", "8"))

# Business scope
VERTICALS_TARGET = ["Payments", "Hospitality", "Onboarding"]
ONBOARDING_DEPARTMENTS = [
    d.strip() for d in os.getenv("ONBOARDING_DEPARTMENTS", "OB_CUST").split(",") if d.strip()
]
EXCLUDE_DEPARTMENT_NAME_TOKENS = ["PROJ", "DIST", "KEY", "PROXIMIS"]

# Optional Excel output
EXPORT_EXCEL = os.getenv("EXPORT_EXCEL", "true").lower() == "true"
OUTPUT_DIR = os.getenv("CAPACITY_OUTPUT_DIR", str(Path.cwd() / "outputs"))
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "capacity_forecast_v18_SQL.xlsx")

# Target tables (existing)
TARGET_TABLE = os.getenv("TARGET_TABLE", "capacity_forecast")
TARGET_TABLE_RISK = os.getenv("TARGET_TABLE_RISK", "capacity_forecast_risk")

# Insert robustness
CHUNKSIZE = int(os.getenv("SQL_INSERT_CHUNKSIZE", "200"))  # key to avoid 08S01/10054
MAX_RETRIES = int(os.getenv("SQL_MAX_RETRIES", "5"))
BASE_BACKOFF_SECONDS = float(os.getenv("SQL_RETRY_BACKOFF_BASE", "2.0"))


if not SQL_PASSWORD:
    raise EnvironmentError("Missing SQL_PASSWORD environment variable.")

print("OUTPUT_XLSX ->", OUTPUT_XLSX)
print("SQL_SERVER  ->", SQL_SERVER)
print("SQL_DB      ->", SQL_DATABASE)
print("SQL_SCHEMA  ->", SQL_SCHEMA)
print("TARGET_BASE ->", TARGET_TABLE)
print("TARGET_RISK ->", TARGET_TABLE_RISK)


# -----------------------------
# 2) SQL Engine (robust)
# -----------------------------

def build_engine() -> sa.Engine:
    # Keep Connection Timeout explicit; add keepalive-like behaviors through pool_pre_ping + recycle.
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

    # pool_recycle: closes and recreates connections periodically to avoid stale sockets.
    return sa.create_engine(
        "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc_params),
        fast_executemany=True,
        pool_pre_ping=True,
        pool_recycle=1800,   # 30 min
        pool_size=5,
        max_overflow=10,
    )


engine = build_engine()

with engine.connect() as conn:
    conn.execute(text("SELECT 1"))
print("✅ SQL connection OK")


# -----------------------------
# 3) Helpers
# -----------------------------

def fq(table_name: str) -> str:
    return f"[{SQL_SCHEMA}].[{table_name}]"


def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_numeric(x, default=0.0):
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").fillna(default)
    if isinstance(x, (int, float, np.number)):
        return default if pd.isna(x) else x
    try:
        v = pd.to_numeric(x, errors="coerce")
        return default if pd.isna(v) else v
    except Exception:
        return default


def to_month_start(dt_series) -> pd.Series:
    s = pd.to_datetime(dt_series, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp(how="start")


def read_table(sql: str) -> pd.DataFrame:
    return std_cols(pd.read_sql(text(sql), engine))


def month_day_multiplier(month_start: pd.Timestamp, basis: str) -> int:
    ms = pd.Timestamp(month_start).normalize()
    me = (ms + pd.offsets.MonthEnd(1)).normalize()
    if basis == "business":
        return int(np.busday_count(ms.date(), (me + pd.Timedelta(days=1)).date()))
    return int((me - ms).days + 1)


def seasonal_naive_forecast(y: pd.Series, horizon_days: int, season_len: int = 7) -> np.ndarray:
    y = y.asfreq("D").fillna(0.0).astype(float)
    if len(y) == 0:
        return np.zeros(horizon_days, dtype=float)
    if len(y) < season_len:
        return np.full(horizon_days, float(y.iloc[-1]), dtype=float)
    pattern = y.iloc[-season_len:].to_numpy(dtype=float)
    reps = int(np.ceil(horizon_days / season_len))
    return np.tile(pattern, reps)[:horizon_days]


def forecast_daily(y: pd.Series, horizon_days: int) -> pd.DataFrame:
    """
    Lightweight forecast:
    - p50: weekly seasonal naive
    - p05/p95: +/- 1.645 * std(residuals last 90 days)
    """
    y = y.asfreq("D").fillna(0.0).astype(float)

    if y.empty:
        future_idx = pd.date_range(pd.Timestamp.today().normalize(), periods=horizon_days, freq="D")
        return pd.DataFrame({"date": future_idx, "p50": 0.0, "p05": 0.0, "p95": 0.0})

    base = seasonal_naive_forecast(y, horizon_days)
    smooth = y.rolling(7, min_periods=1).mean()
    residuals = (y - smooth).tail(90)
    std = float(residuals.std()) if residuals.notna().any() else 0.0

    start = y.index.max() + pd.Timedelta(days=1)
    future_idx = pd.date_range(start, periods=horizon_days, freq="D")

    p50 = base
    p05 = np.maximum(base - 1.645 * std, 0.0)
    p95 = base + 1.645 * std

    return pd.DataFrame({"date": future_idx, "p50": p50, "p05": p05, "p95": p95})


def recent_rate(df_hist: pd.DataFrame, value_col: str, base_col: str, months: int) -> pd.DataFrame:
    out_col = f"{value_col}_rate_recent"
    if df_hist.empty or "month" not in df_hist.columns:
        return pd.DataFrame(columns=["department_id", out_col])

    df = df_hist.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    max_month = df["month"].max()
    if pd.isna(max_month):
        return pd.DataFrame(columns=["department_id", out_col])

    cutoff = max_month - pd.DateOffset(months=months)
    df = df[df["month"] >= cutoff].copy()

    df[value_col] = safe_numeric(df.get(value_col, 0.0), 0.0)
    df[base_col] = safe_numeric(df.get(base_col, 0.0), 0.0)

    agg = df.groupby("department_id", as_index=False).agg(
        val_sum=(value_col, "sum"),
        base_sum=(base_col, "sum"),
    )
    agg[out_col] = np.where(agg["base_sum"] > 0, agg["val_sum"] / agg["base_sum"], 0.0)
    return agg[["department_id", out_col]]


def recent_mean_per_dept(df: pd.DataFrame, value_col: str, months: int, out_name: str) -> pd.DataFrame:
    if df.empty or "month" not in df.columns:
        return pd.DataFrame(columns=["department_id", out_name])

    tmp = df.dropna(subset=[value_col]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["department_id", out_name])

    tmp["month"] = pd.to_datetime(tmp["month"], errors="coerce")
    max_month = tmp["month"].max()
    if pd.isna(max_month):
        return pd.DataFrame(columns=["department_id", out_name])

    cutoff = max_month - pd.DateOffset(months=months)
    tmp = tmp[tmp["month"] >= cutoff].copy()

    out = tmp.groupby("department_id", as_index=False).agg(val=(value_col, "mean"))
    out.rename(columns={"val": out_name}, inplace=True)
    return out


def is_transient_sql_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    transient_tokens = [
        "08s01", "10054", "communication link failure", "tcp provider",
        "connection forcibly closed", "connection reset", "se ha forzado la interrupción",
        "deadlock victim", "timeout", "transport-level error"
    ]
    return any(t in msg for t in transient_tokens)


def rebuild_engine():
    global engine
    try:
        engine.dispose()
    except Exception:
        pass
    engine = build_engine()


def with_retry(func, *, max_retries: int = MAX_RETRIES, base_backoff: float = BASE_BACKOFF_SECONDS, label: str = "operation"):
    """
    Execute func() with retry on transient DB/ODBC failures.
    Rebuilds engine between attempts to force a new connection.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except (OperationalError, DBAPIError) as e:
            if not is_transient_sql_error(e) or attempt == max_retries:
                raise
            sleep_s = base_backoff * (2 ** (attempt - 1)) + random.random()
            print(f"⚠️ Transient SQL error during {label}. Retry {attempt}/{max_retries} in {sleep_s:.1f}s")
            rebuild_engine()
            time.sleep(sleep_s)


# -----------------------------
# 4) Load inputs from SQL
# -----------------------------

def load_inputs():
    incoming_raw = read_table(f"""
        SELECT
            [Date] AS [date],
            department_id,
            total_incoming AS incoming_total
        FROM {fq('DAILY_Incoming_Total')}
        WHERE department_id IS NOT NULL
          AND [Date] IS NOT NULL
    """)

    dept_map = read_table(f"""
        SELECT
            department_id,
            department_name,
            vertical
        FROM {fq('department')}
    """)

    incoming = incoming_raw.copy()
    incoming["date"] = pd.to_datetime(incoming["date"], errors="coerce")
    incoming["month"] = to_month_start(incoming["date"])
    incoming["department_id"] = pd.to_numeric(incoming["department_id"], errors="coerce").astype("Int64")
    incoming["incoming_total"] = pd.to_numeric(incoming["incoming_total"], errors="coerce").fillna(0.0)

    dept_map["department_id"] = pd.to_numeric(dept_map["department_id"], errors="coerce").astype("Int64")
    incoming = incoming.merge(dept_map, on="department_id", how="left")

    # Scope filter
    incoming = incoming[incoming["vertical"].isin(VERTICALS_TARGET)].copy()

    # Only specific departments for Onboarding
    incoming = incoming[
        (incoming["vertical"] != "Onboarding") |
        (incoming["department_name"].isin(ONBOARDING_DEPARTMENTS))
    ].copy()

    # Exclusions
    mask_excl = pd.Series(False, index=incoming.index)
    for tok in EXCLUDE_DEPARTMENT_NAME_TOKENS:
        mask_excl = mask_excl | incoming["department_name"].astype(str).str.upper().str.contains(tok.upper(), na=False)
    incoming = incoming.loc[~mask_excl].copy()

    calls_ni_month = read_table(f"""
        SELECT
            department_id,
            DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1) AS [month],
            COUNT(*) AS calls_not_indexed
        FROM {fq('calls_not_indexed')}
        WHERE department_id IS NOT NULL
          AND [Date] IS NOT NULL
        GROUP BY department_id, DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1)
    """)
    calls_ni_month["month"] = pd.to_datetime(calls_ni_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

    repeats_month = read_table(f"""
        SELECT
            department_id,
            DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1) AS [month],
            COUNT(DISTINCT case_number) AS repeats_workload
        FROM {fq('DAILY_RR_new')}
        WHERE department_id IS NOT NULL
          AND [Date] IS NOT NULL
        GROUP BY department_id, DATEFROMPARTS(YEAR([Date]), MONTH([Date]), 1)
    """)
    repeats_month["month"] = pd.to_datetime(repeats_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

    inventory_month = read_table(f"""
        SELECT
            department_id,
            [month],
            inventory_cases AS inventory
        FROM {fq('vw_capacity_inventory_monthly')}
    """)
    inventory_month["month"] = pd.to_datetime(inventory_month["month"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")

    targets = read_table(f"""
        SELECT
            agent_id,
            department_id,
            item_target
        FROM {fq('agent_item_target')}
        WHERE department_id IS NOT NULL
    """)
    targets["department_id"] = pd.to_numeric(targets["department_id"], errors="coerce").astype("Int64")
    targets["agent_id"] = pd.to_numeric(targets["agent_id"], errors="coerce").astype("Int64")
    targets["item_target"] = pd.to_numeric(targets["item_target"], errors="coerce").fillna(0.0)

    dept_target_avg = (
        targets.groupby("department_id", as_index=False)
        .agg(target_per_fte_day=("item_target", "mean"))
    )

    prod_daily = read_table(f"""
        SELECT
            [Date] AS [date],
            agent_id,
            department_id,
            department_name,
            prod_total_model
        FROM {fq('agent_productivity_new_daily')}
        WHERE department_id IS NOT NULL
          AND agent_id IS NOT NULL
          AND [Date] IS NOT NULL
    """)
    prod_daily["date"] = pd.to_datetime(prod_daily["date"], errors="coerce")
    prod_daily["month"] = to_month_start(prod_daily["date"])
    prod_daily["department_id"] = pd.to_numeric(prod_daily["department_id"], errors="coerce").astype("Int64")
    prod_daily["agent_id"] = pd.to_numeric(prod_daily["agent_id"], errors="coerce").astype("Int64")
    prod_daily["prod_total_model"] = pd.to_numeric(prod_daily["prod_total_model"], errors="coerce").fillna(0.0)

    prod_month = (
        prod_daily.groupby(["department_id", "month"], as_index=False)
        .agg(actual_produced=("prod_total_model", "sum"))
    )

    shrink = read_table(f"""
        SELECT
            [Date] AS [date],
            agent_id,
            shrinkage_time
        FROM {fq('shrinkage')}
        WHERE agent_id IS NOT NULL
          AND [Date] IS NOT NULL
    """)
    shrink["date"] = pd.to_datetime(shrink["date"], errors="coerce")
    shrink["month"] = to_month_start(shrink["date"])
    shrink["agent_id"] = pd.to_numeric(shrink["agent_id"], errors="coerce").astype("Int64")
    shrink["shrinkage_time"] = pd.to_numeric(shrink["shrinkage_time"], errors="coerce").fillna(0.0)

    agents = read_table(f"""
        SELECT
            agent_id,
            department_id,
            tl_id,
            start_date,
            fte_pct
        FROM {fq('agent')}
        WHERE agent_id IS NOT NULL
          AND department_id IS NOT NULL
    """)
    agents["agent_id"] = pd.to_numeric(agents["agent_id"], errors="coerce").astype("Int64")
    agents["department_id"] = pd.to_numeric(agents["department_id"], errors="coerce").astype("Int64")
    agents["tl_id"] = pd.to_numeric(agents["tl_id"], errors="coerce").fillna(0).astype(int)
    agents["start_date"] = pd.to_datetime(agents["start_date"], errors="coerce")
    agents["fte_pct"] = pd.to_numeric(agents["fte_pct"], errors="coerce").fillna(1.0)

    agents["is_employed"] = agents["tl_id"] != 0

    return incoming, calls_ni_month, repeats_month, inventory_month, dept_target_avg, prod_daily, prod_month, shrink, agents


# -----------------------------
# 5) Model build
# -----------------------------

def build_model(
    incoming: pd.DataFrame,
    calls_ni_month: pd.DataFrame,
    repeats_month: pd.DataFrame,
    inventory_month: pd.DataFrame,
    dept_target_avg: pd.DataFrame,
    prod_daily: pd.DataFrame,
    prod_month: pd.DataFrame,
    shrink: pd.DataFrame,
    agents: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    monthly_actuals = (
        incoming.groupby(["vertical", "department_id", "department_name", "month"], as_index=False)
        .agg(actual_incoming=("incoming_total", "sum"))
    )
    monthly_actuals["actual_incoming"] = safe_numeric(monthly_actuals["actual_incoming"], 0.0).clip(lower=0.0)

    daily_hist = (
        incoming.groupby(["vertical", "department_id", "department_name", "date"], as_index=False)
        .agg(y=("incoming_total", "sum"))
    )
    daily_hist["date"] = pd.to_datetime(daily_hist["date"], errors="coerce")
    daily_hist["y"] = safe_numeric(daily_hist["y"], 0.0).clip(lower=0.0)

    last_hist_date = pd.to_datetime(daily_hist["date"], errors="coerce").max()
    if pd.isna(last_hist_date):
        last_hist_date = pd.Timestamp.today().normalize()

    last_hist_month = monthly_actuals["month"].max()
    if pd.isna(last_hist_month):
        last_hist_month = pd.Timestamp.today().to_period("M").to_timestamp(how="start")

    future_months = pd.date_range(last_hist_month + pd.offsets.MonthBegin(1), periods=HORIZON_MONTHS, freq="MS")
    max_future_date = (future_months.max() + pd.offsets.MonthEnd(1)).normalize() if len(future_months) else last_hist_date
    horizon_days = max(1, int((max_future_date - last_hist_date).days))

    daily_fc_rows = []
    for (vertical, dept_id, dept_name), g in daily_hist.groupby(["vertical", "department_id", "department_name"]):
        g = g.sort_values("date")
        y = g.set_index("date")["y"].asfreq("D").fillna(0.0)
        fc = forecast_daily(y, horizon_days=horizon_days)
        fc["vertical"] = vertical
        fc["department_id"] = dept_id
        fc["department_name"] = dept_name
        daily_fc_rows.append(fc)

    daily_fc = (
        pd.concat(daily_fc_rows, ignore_index=True)
        if daily_fc_rows
        else pd.DataFrame(columns=["date", "p50", "p05", "p95", "vertical", "department_id", "department_name"])
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

    all_months = pd.DatetimeIndex(sorted(set(monthly_actuals["month"]).union(set(future_months))))
    dept_dim = monthly_actuals[["vertical", "department_id", "department_name"]].drop_duplicates()

    spine = (
        dept_dim.assign(_k=1)
        .merge(pd.DataFrame({"month": all_months, "_k": 1}), on="_k", how="outer")
        .drop(columns=["_k"])
    )

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

    mask_hist = monthly_fc_raw["actual_incoming"].notna()
    for c in ["forecast_monthly_incoming", "forecast_p05", "forecast_p95"]:
        monthly_fc_raw[c] = safe_numeric(monthly_fc_raw.get(c, 0.0), 0.0)
        monthly_fc_raw.loc[mask_hist, c] = safe_numeric(monthly_fc_raw.loc[mask_hist, "actual_incoming"], 0.0)

    monthly_adj = monthly_fc_raw.copy()
    monthly_adj["actual_incoming"] = safe_numeric(monthly_adj["actual_incoming"], 0.0).clip(lower=0.0)

    # Calls NI + repeats ratios
    hist_calls = monthly_actuals.merge(calls_ni_month, on=["department_id", "month"], how="left")
    hist_calls["calls_not_indexed"] = safe_numeric(hist_calls.get("calls_not_indexed", 0.0), 0.0)
    calls_rate = recent_rate(hist_calls, "calls_not_indexed", "actual_incoming", RECENT_RATE_MONTHS)

    hist_rr = monthly_actuals.merge(repeats_month, on=["department_id", "month"], how="left")
    hist_rr["repeats_workload"] = safe_numeric(hist_rr.get("repeats_workload", 0.0), 0.0)
    repeats_rate = recent_rate(hist_rr, "repeats_workload", "actual_incoming", RECENT_RATE_MONTHS)

    monthly_adj = monthly_adj.merge(calls_rate, on="department_id", how="left")
    monthly_adj = monthly_adj.merge(repeats_rate, on="department_id", how="left")

    monthly_adj["calls_not_indexed_rate_recent"] = safe_numeric(
        monthly_adj.get("calls_not_indexed_rate_recent", 0.0), 0.0
    ).clip(0, 5.0)
    monthly_adj["repeats_workload_rate_recent"] = safe_numeric(
        monthly_adj.get("repeats_workload_rate_recent", 0.0), 0.0
    ).clip(0, 5.0)

    monthly_adj["calls_not_indexed_forecast"] = safe_numeric(monthly_adj["forecast_monthly_incoming"], 0.0) * monthly_adj["calls_not_indexed_rate_recent"]
    monthly_adj["repeats_forecast"] = safe_numeric(monthly_adj["forecast_monthly_incoming"], 0.0) * monthly_adj["repeats_workload_rate_recent"]

    # Inventory
    inventory_month["inventory"] = safe_numeric(inventory_month["inventory"], 0.0)
    monthly_adj = monthly_adj.merge(
        inventory_month.rename(columns={"inventory": "inventory_actual"}),
        on=["department_id", "month"],
        how="left",
    )
    monthly_adj["inventory_final"] = safe_numeric(monthly_adj["inventory_actual"], 0.0)

    # Days multiplier
    month_multiplier = pd.DataFrame({"month": all_months})
    month_multiplier["days_multiplier"] = month_multiplier["month"].apply(lambda m: month_day_multiplier(m, CAPACITY_DAY_BASIS))
    monthly_adj = monthly_adj.merge(month_multiplier, on="month", how="left")
    monthly_adj["days_multiplier"] = pd.to_numeric(monthly_adj["days_multiplier"], errors="coerce").fillna(0).astype(int)

    # Headcount inventory projection
    month_spine = pd.DataFrame({"month": all_months})
    month_spine["month_end"] = (pd.to_datetime(month_spine["month"]) + pd.offsets.MonthEnd(1)).dt.normalize()

    agents_small = agents.copy()
    agents_small["_k"] = 1
    month_spine["_k"] = 1
    axm = agents_small.merge(month_spine, on="_k", how="outer").drop(columns=["_k"])

    axm = axm[
        (axm["is_employed"]) &
        (axm["start_date"].isna() | (axm["start_date"] <= axm["month_end"]))
    ].copy()

    headcount_inv = (
        axm.groupby(["department_id", "month"], as_index=False)
        .agg(headcount_inventory=("fte_pct", "sum"))
    )

    monthly_adj = monthly_adj.merge(headcount_inv, on=["department_id", "month"], how="left")
    monthly_adj["headcount_inventory"] = safe_numeric(monthly_adj.get("headcount_inventory", 0.0), 0.0)

    # Active FTE observed + projection
    agent_fte_map = agents[["agent_id", "fte_pct", "is_employed"]].copy()
    prod_daily_fte = prod_daily.merge(agent_fte_map, on="agent_id", how="left")
    prod_daily_fte["fte_pct"] = pd.to_numeric(prod_daily_fte["fte_pct"], errors="coerce").fillna(1.0)
    prod_daily_fte["is_employed"] = prod_daily_fte["is_employed"].fillna(True)
    prod_daily_fte = prod_daily_fte[prod_daily_fte["is_employed"]].copy()

    active_fte_daily = (
        prod_daily_fte.groupby(["department_id", "date"], as_index=False)
        .agg(active_fte=("fte_pct", "sum"))
    )
    active_fte_daily["month"] = to_month_start(active_fte_daily["date"])

    fte_actual_obs = (
        active_fte_daily.groupby(["department_id", "month"], as_index=False)
        .agg(fte_actual_obs=("active_fte", "mean"))
    )
    monthly_adj = monthly_adj.merge(fte_actual_obs, on=["department_id", "month"], how="left")

    monthly_adj["active_ratio_obs"] = np.where(
        (monthly_adj["headcount_inventory"] > 0) & (monthly_adj["fte_actual_obs"].notna()),
        monthly_adj["fte_actual_obs"] / monthly_adj["headcount_inventory"],
        np.nan
    )

    active_ratio_recent = recent_mean_per_dept(
        monthly_adj, "active_ratio_obs", RECENT_ACTIVE_RATIO_MONTHS, "active_ratio_recent_mean"
    )
    monthly_adj = monthly_adj.merge(active_ratio_recent, on="department_id", how="left")

    monthly_adj["fte_actual"] = np.where(
        monthly_adj["fte_actual_obs"].notna(),
        monthly_adj["fte_actual_obs"],
        np.where(
            monthly_adj["active_ratio_recent_mean"].notna(),
            monthly_adj["headcount_inventory"] * np.clip(monthly_adj["active_ratio_recent_mean"], 0, 1),
            monthly_adj["headcount_inventory"]
        )
    )

    # Shrinkage (fix FutureWarning by using pandas BooleanDtype)
    shrink_map = shrink.merge(
        agents[["agent_id", "department_id", "is_employed"]],
        on="agent_id",
        how="left"
    )
    shrink_map = shrink_map[shrink_map["department_id"].notna()].copy()

    # Boolean dtype to avoid silent downcasting warnings
    shrink_map["is_employed"] = shrink_map["is_employed"].astype("boolean").fillna(True)
    shrink_map = shrink_map[shrink_map["is_employed"]].copy()

    shrink_month = (
        shrink_map.groupby(["department_id", "month"], as_index=False)
        .agg(shrinkage_seconds=("shrinkage_time", "sum"))
    )
    shrink_month["shrinkage_hours"] = shrink_month["shrinkage_seconds"] / 3600.0

    monthly_adj = monthly_adj.merge(
        shrink_month[["department_id", "month", "shrinkage_hours"]],
        on=["department_id", "month"],
        how="left"
    )
    monthly_adj["shrinkage_hours"] = safe_numeric(monthly_adj.get("shrinkage_hours", 0.0), 0.0)

    monthly_adj["shrinkage_pct_obs"] = np.where(
        (monthly_adj["headcount_inventory"] > 0) & (monthly_adj["days_multiplier"] > 0),
        monthly_adj["shrinkage_hours"] / (monthly_adj["headcount_inventory"] * WORK_HOURS_PER_DAY * monthly_adj["days_multiplier"]),
        np.nan
    )
    monthly_adj["shrinkage_pct_obs"] = pd.to_numeric(monthly_adj["shrinkage_pct_obs"], errors="coerce").clip(0.0, 0.90)

    shrink_recent = recent_mean_per_dept(
        monthly_adj, "shrinkage_pct_obs", RECENT_SHRINK_MONTHS, "shrinkage_pct_recent_mean"
    )
    monthly_adj = monthly_adj.merge(shrink_recent, on="department_id", how="left")

    monthly_adj["shrinkage_pct"] = np.where(
        monthly_adj["shrinkage_pct_obs"].notna(),
        monthly_adj["shrinkage_pct_obs"],
        monthly_adj["shrinkage_pct_recent_mean"]
    )
    monthly_adj["shrinkage_pct"] = pd.to_numeric(monthly_adj["shrinkage_pct"], errors="coerce").fillna(0.0).clip(0.0, 0.90)

    monthly_adj["fte_after_shrinkage"] = monthly_adj["headcount_inventory"] * (1.0 - monthly_adj["shrinkage_pct"])

    # Capacity
    monthly_adj = monthly_adj.merge(dept_target_avg, on="department_id", how="left")
    monthly_adj["target_per_fte_day"] = safe_numeric(monthly_adj.get("target_per_fte_day", 0.0), 0.0)

    monthly_adj["capacity_day"] = monthly_adj["target_per_fte_day"] * monthly_adj["headcount_inventory"]
    monthly_adj["capacity_gross"] = np.round(monthly_adj["capacity_day"] * monthly_adj["days_multiplier"]).astype("Int64")
    monthly_adj["capacity_net"] = np.round(monthly_adj["capacity_gross"] * (1.0 - monthly_adj["shrinkage_pct"])).astype("Int64")

    # Productivity
    monthly_adj = monthly_adj.merge(prod_month, on=["department_id", "month"], how="left")
    monthly_adj["actual_produced"] = pd.to_numeric(monthly_adj["actual_produced"], errors="coerce")  # keep NaN future

    monthly_adj["productivity_per_fte_day"] = np.where(
        (monthly_adj["fte_actual"] > 0) &
        (monthly_adj["days_multiplier"] > 0) &
        (monthly_adj["actual_produced"].notna()),
        monthly_adj["actual_produced"] / (monthly_adj["fte_actual"] * monthly_adj["days_multiplier"]),
        np.nan
    )

    # Workload forecast total + FTE required
    monthly_adj["workload_forecast_total"] = (
        safe_numeric(monthly_adj["forecast_monthly_incoming"], 0.0)
        + safe_numeric(monthly_adj["calls_not_indexed_forecast"], 0.0)
        + safe_numeric(monthly_adj["repeats_forecast"], 0.0)
        + safe_numeric(monthly_adj["inventory_final"], 0.0)
    )
    monthly_adj["workload_forecast_total"] = np.round(monthly_adj["workload_forecast_total"]).astype("Int64")

    monthly_adj["target_per_fte_month_net"] = (
        monthly_adj["target_per_fte_day"] * monthly_adj["days_multiplier"] * (1.0 - monthly_adj["shrinkage_pct"])
    )

    monthly_adj["fte_required"] = np.where(
        monthly_adj["target_per_fte_month_net"] > 0,
        safe_numeric(monthly_adj["workload_forecast_total"], 0.0) / monthly_adj["target_per_fte_month_net"],
        0.0
    )
    monthly_adj["fte_required"] = pd.to_numeric(monthly_adj["fte_required"], errors="coerce").fillna(0.0)
    monthly_adj["fte_gap"] = monthly_adj["fte_required"] - monthly_adj["fte_after_shrinkage"]

    def classify_status(row):
        if pd.isna(row["fte_gap"]):
            return "UNKNOWN"
        if row["fte_gap"] > 0.5:
            return "UNDER_CAPACITY"
        if row["fte_gap"] < -2.0:
            return "OVER_CAPACITY"
        return "BALANCED"

    monthly_adj["status"] = monthly_adj.apply(classify_status, axis=1)

    # Risk score
    monthly_adj["workload_minus_net_capacity"] = (
        safe_numeric(monthly_adj["workload_forecast_total"], 0.0) - safe_numeric(monthly_adj["capacity_net"], 0.0)
    )
    monthly_adj["workload_minus_net_capacity"] = np.round(monthly_adj["workload_minus_net_capacity"]).astype("Int64")

    workload_ratio = np.where(
        safe_numeric(monthly_adj["workload_forecast_total"], 0.0) > 0,
        np.clip(
            safe_numeric(monthly_adj["workload_minus_net_capacity"], 0.0) /
            safe_numeric(monthly_adj["workload_forecast_total"], 1.0),
            0, 1
        ),
        0.0
    )
    fte_ratio = np.where(
        monthly_adj["fte_required"] > 0,
        np.clip(np.maximum(monthly_adj["fte_gap"], 0.0) / monthly_adj["fte_required"], 0, 1),
        0.0
    )
    monthly_adj["risk_score"] = np.round(100.0 * (0.65 * workload_ratio + 0.35 * fte_ratio)).astype(int)

    def risk_band(score: int) -> str:
        if score >= 60:
            return "RED"
        if score >= 30:
            return "AMBER"
        return "GREEN"

    monthly_adj["risk_band"] = monthly_adj["risk_score"].apply(risk_band)

    # -----------------------------
    # Build BASE DF (standard internal -> SQL columns)
    # -----------------------------
    run_date = pd.Timestamp.today().normalize()

    base_df = pd.DataFrame({
        "run_date": run_date,
        "Month": monthly_adj["month"],
        "Vertical": monthly_adj["vertical"],
        "Department_name": monthly_adj["department_name"],

        "Actual Volume": safe_numeric(monthly_adj["actual_incoming"], 0.0).round(0).astype("Int64"),
        "Forecast (Cases)": safe_numeric(monthly_adj["forecast_monthly_incoming"], 0.0).round(0).astype("Int64"),
        "Calls not indexed (forecast)": safe_numeric(monthly_adj["calls_not_indexed_forecast"], 0.0).round(0).astype("Int64"),
        "Repeats (forecast)": safe_numeric(monthly_adj["repeats_forecast"], 0.0).round(0).astype("Int64"),
        "Capacity": safe_numeric(monthly_adj["capacity_gross"], 0.0).round(0).astype("Int64"),
        "Inventory": safe_numeric(monthly_adj["inventory_final"], 0.0).round(0).astype("Int64"),
        "Workload Forecast (Humans + calls not indexed + repeats )": (
            safe_numeric(monthly_adj["forecast_monthly_incoming"], 0.0)
            + safe_numeric(monthly_adj["calls_not_indexed_forecast"], 0.0)
            + safe_numeric(monthly_adj["repeats_forecast"], 0.0)
        ).round(0).astype("Int64"),
        # Your new SQL column
        "Actual_Workload": safe_numeric(monthly_adj["actual_incoming"], 0.0).round(0).astype("Int64"),
    })

    base_sql_order = [
        "run_date", "Month", "Vertical", "Department_name",
        "Actual Volume", "Forecast (Cases)", "Calls not indexed (forecast)", "Repeats (forecast)",
        "Capacity", "Inventory", "Workload Forecast (Humans + calls not indexed + repeats )",
        "Actual_Workload"
    ]
    base_df = base_df[base_sql_order].copy()

    # -----------------------------
    # Build RISK DF (match your SQL table exactly)
    # -----------------------------
    risk_df = pd.DataFrame({
        "run_date": run_date,
        "department_id": monthly_adj["department_id"].astype("Int64"),
        "Month": monthly_adj["month"],
        "Vertical": monthly_adj["vertical"],
        "Department_name": monthly_adj["department_name"],

        "Actual Incoming": safe_numeric(monthly_adj["actual_incoming"], 0.0).round(0).astype("Int64"),
        "Forecast (Incoming)": safe_numeric(monthly_adj["forecast_monthly_incoming"], 0.0).round(0).astype("Int64"),
        "Calls not indexed (forecast)": safe_numeric(monthly_adj["calls_not_indexed_forecast"], 0.0).round(0).astype("Int64"),
        "Repeats (forecast)": safe_numeric(monthly_adj["repeats_forecast"], 0.0).round(0).astype("Int64"),
        "Inventory": safe_numeric(monthly_adj["inventory_final"], 0.0).round(0).astype("Int64"),
        "Workload Forecast Total": safe_numeric(monthly_adj["workload_forecast_total"], 0.0).round(0).astype("Int64"),

        "Actual Produced": pd.to_numeric(monthly_adj["actual_produced"], errors="coerce").round(0).astype("Int64"),

        # decimals in SQL -> round to 6
        "Productivity per active FTE per day": pd.to_numeric(monthly_adj["productivity_per_fte_day"], errors="coerce").round(6),
        "Target per FTE per day": pd.to_numeric(monthly_adj["target_per_fte_day"], errors="coerce").round(6),
        "Headcount": pd.to_numeric(monthly_adj["headcount_inventory"], errors="coerce").round(6),
        "FTE actual": pd.to_numeric(monthly_adj["fte_actual"], errors="coerce").round(6),
        "Shrinkage %": pd.to_numeric(monthly_adj["shrinkage_pct"], errors="coerce").round(6),
        "FTE after shrinkage": pd.to_numeric(monthly_adj["fte_after_shrinkage"], errors="coerce").round(6),

        "Capacity Gross": safe_numeric(monthly_adj["capacity_gross"], 0.0).round(0).astype("Int64"),
        "Capacity Net": safe_numeric(monthly_adj["capacity_net"], 0.0).round(0).astype("Int64"),
        "FTE required": pd.to_numeric(monthly_adj["fte_required"], errors="coerce").round(6),
        "FTE gap": pd.to_numeric(monthly_adj["fte_gap"], errors="coerce").round(6),
        "Workload minus Net Capacity": safe_numeric(monthly_adj["workload_minus_net_capacity"], 0.0).round(0).astype("Int64"),

        "RiskScore": pd.to_numeric(monthly_adj["risk_score"], errors="coerce").fillna(0).astype(int),
        "RiskBand": monthly_adj["risk_band"].astype(str),
        "Status": monthly_adj["status"].astype(str),
    })

    risk_sql_order = [
        "run_date", "department_id", "Month", "Vertical", "Department_name",
        "Actual Incoming", "Forecast (Incoming)", "Calls not indexed (forecast)", "Repeats (forecast)",
        "Inventory", "Workload Forecast Total", "Actual Produced",
        "Productivity per active FTE per day", "Target per FTE per day",
        "Headcount", "FTE actual", "Shrinkage %", "FTE after shrinkage",
        "Capacity Gross", "Capacity Net", "FTE required", "FTE gap",
        "Workload minus Net Capacity", "RiskScore", "RiskBand", "Status"
    ]
    risk_df = risk_df[risk_sql_order].copy()

    return monthly_adj, base_df, risk_df


# -----------------------------
# 6) Export (production-safe)
# -----------------------------

def export_outputs(monthly_adj: pd.DataFrame, base_df: pd.DataFrame, risk_df: pd.DataFrame):
    # Excel export (optional)
    if EXPORT_EXCEL:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as writer:
            monthly_adj.to_excel(writer, sheet_name="Monthly_Model", index=False)
            base_df.to_excel(writer, sheet_name="capacity_forecast", index=False)
            risk_df.to_excel(writer, sheet_name="capacity_forecast_risk", index=False)

    # TRUNCATE in its own transaction
    def do_truncate():
        with engine.begin() as conn:
            print("🔄 Truncating tables...")
            conn.execute(text(f"TRUNCATE TABLE {fq(TARGET_TABLE)}"))
            conn.execute(text(f"TRUNCATE TABLE {fq(TARGET_TABLE_RISK)}"))

    with_retry(do_truncate, label="truncate")

    # Insert base in separate transaction/connection
    def do_insert_base():
        with engine.begin() as conn:
            print("⬆️ Uploading capacity_forecast...")
            base_df.to_sql(
                TARGET_TABLE,
                engine,
                schema=SQL_SCHEMA,
                if_exists="append",
                index=False,
                method=None,          # avoid multi-row INSERT statement issues
                chunksize=CHUNKSIZE,  # key to avoid 08S01/10054
            )

    with_retry(do_insert_base, label="insert base")

    # Insert risk in separate transaction/connection
    def do_insert_risk():
        with engine.begin() as conn:
            print("⬆️ Uploading capacity_forecast_risk...")
            risk_df.to_sql(
                TARGET_TABLE_RISK,
                engine,
                schema=SQL_SCHEMA,
                if_exists="append",
                index=False,
                method=None,
                chunksize=CHUNKSIZE,
            )

    with_retry(do_insert_risk, label="insert risk")

    # Post-write verification
    def do_verify():
        with engine.connect() as conn:
            base_count = conn.execute(text(f"SELECT COUNT(*) FROM {fq(TARGET_TABLE)}")).scalar()
            risk_count = conn.execute(text(f"SELECT COUNT(*) FROM {fq(TARGET_TABLE_RISK)}")).scalar()
        return int(base_count or 0), int(risk_count or 0)

    base_count, risk_count = with_retry(do_verify, label="verify counts")

    if base_count == 0:
        raise RuntimeError("❌ Base table empty after insert")
    if risk_count == 0:
        raise RuntimeError("❌ Risk table empty after insert")

    print(f"✅ SUCCESS -> base={base_count}, risk={risk_count}")
    print("Done.")


# -----------------------------
# Main
# -----------------------------

def main():
    incoming, calls_ni_month, repeats_month, inventory_month, dept_target_avg, prod_daily, prod_month, shrink, agents = load_inputs()
    monthly_adj, base_df, risk_df = build_model(
        incoming=incoming,
        calls_ni_month=calls_ni_month,
        repeats_month=repeats_month,
        inventory_month=inventory_month,
        dept_target_avg=dept_target_avg,
        prod_daily=prod_daily,
        prod_month=prod_month,
        shrink=shrink,
        agents=agents,
    )
    export_outputs(monthly_adj, base_df, risk_df)


if __name__ == "__main__":
    main()