
# %% [markdown]
# # Corporate Hybrid Forecast Notebook — v6_4 (Hybrid + Global ML + Exogenous)
# 
# **Generated:** 2026-02-09 09:13 UTC  
# This notebook extends the existing hybrid pipeline (Prophet / ARIMA / TBATS-ETS) with:
# 
# - Weekly aggregation candidate for noisy series (Thu→Wed weeks)
# - Robust outlier detection/correction **before** modeling
# - **Global LightGBM per vertical** with temporal features (multi-series)
# - Naive and seasonal-naive baselines
# - Dynamic language splitting (use actual language shares when available; else fixed shares)
# - **Exogenous features**: EU core holidays & events (auto generator or optional CSV files)

# %% [markdown]
# ## 1. Imports & Config

# %%
import os
import math
import warnings
from typing import Optional, Dict, Tuple, List
from datetime import date, timedelta

import numpy as np
import pandas as pd
import time

# Forecasting libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

# Prophet
try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet  # legacy
    except Exception:
        Prophet = None

# TBATS
try:
    from tbats import TBATS
except Exception:
    TBATS = None

# Machine Learning
try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error
except Exception:
    lgb = None

warnings.filterwarnings("ignore")

# ==================== Configuration ====================

BASE_DIR = r"C:\Users\pt3canro\Desktop\CAPACITY"
INPUT_DIR = os.path.join(BASE_DIR, "input_model")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

INCOMING_SOURCE_PATH = os.path.join(INPUT_DIR, "Incoming_new.xlsx")  # Sheet 'Main'
INCOMING_SHEET = "Main"

DEPT_MAP_PATH = os.path.join(INPUT_DIR, "department.xlsx")
DEPT_MAP_SHEET = "map"

PRODUCTIVITY_PATH = os.path.join(INPUT_DIR, "productivity_agents.xlsx")

OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "capacity_forecast_hybrid.xlsx")

# Horizons
H_MONTHS = 12           # monthly horizon
DAILY_HORIZON_DAYS = 90 # daily plan horizon
REPORT_START_MONTH = "2025-01"

# Reconciliation daily (top-down from monthly)
USE_DAILY_FROM_MONTHLY = True

# Organization-specific week: Thursday→Wednesday (end on Wed)
WEEKLY_FREQ = "W-WED"

# Language handling: "from_column" (keep Incoming language) or "fixed_shares"
LANGUAGE_STRATEGY = "from_column"

# Fixed language shares (fallback)
LANGUAGE_SHARES = {
    "English": 0.6435, "French": 0.0741, "German": 0.0860,
    "Italian": 0.0667, "Portuguese": 0.0162, "Spanish": 0.1135
}

# Outliers & noisy series
OUTLIER_METHOD = "IQR"  # "IQR" or "STL"
IQR_LO = 0.01
IQR_HI = 0.99
NOISE_SCORE_THRESH = 1.00  # robust CV>1 => consider "high-noise"

# Weekly modeling for noisy series (candidate for the ensemble)
ENABLE_WEEKLY_CANDIDATE = True

# Global model per vertical (LightGBM)
ENABLE_GLOBAL_LGB = True
LGB_LAGS = [7, 14, 28]
LGB_ROLLS = [7, 28]
LGB_N_ESTIMATORS = 400
LGB_LEARNING_RATE = 0.04

# ==================== Exogenous (Holidays / Events) ====================
EXOG_ENABLE = True
EXOG_SOURCE = "auto_core"  # "auto_core" | "files"
EXOG_FILES = {
    "holidays": os.path.join(INPUT_DIR, "holidays_eu.csv"),  # optional
    "events": os.path.join(INPUT_DIR, "events_eu.csv"),      # optional
}
# Countries for auto_core generator (EU core set) - not used country-specific in minimal rules
EXOG_COUNTRIES = ["ES", "FR", "DE", "IT", "PT", "GB"]

# %% [markdown]
# ## 2. Helpers, metrics and cleaning

# %%
# ==================== Utilities ====================

def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percentage."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom[denom == 0] = 1.0
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)

def smape_df(df: pd.DataFrame, y_col: str, yhat_col: str) -> float:
    if df.empty:
        return np.nan
    return smape(df[y_col].values, df[yhat_col].values)

def business_days_in_month(year: int, month: int) -> int:
    """Count business days (Mon-Fri) in a given month."""
    rng = pd.date_range(
        start=pd.Timestamp(year=year, month=month, day=1),
        end=pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0),
        freq="D"
    )
    return int(np.sum(rng.weekday < 5))

def _easter_sunday(year: int) -> date:
    """Western Easter (Computus). Return date."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    L = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * L) // 451
    month = (h + L - 7 * m + 114) // 31
    day = ((h + L - 7 * m + 114) % 31) + 1
    return date(year, month, day)

def _last_friday_of_november(year: int) -> date:
    d = date(year, 11, 30)
    while d.weekday() != 4:  # 4=Friday
        d = d.replace(day=d.day - 1)
    return d

def expm1_safe(x, cap_original: Optional[float] = None):
    """Stable expm1 with optional capping on original scale."""
    a = np.array(x, dtype=float)
    a[~np.isfinite(a)] = -50.0
    a = np.maximum(a, -50.0)
    if cap_original and np.isfinite(cap_original) and cap_original > 0:
        log_cap = np.log1p(cap_original)
        a = np.minimum(a, log_cap)
    y = np.expm1(a)
    if cap_original and np.isfinite(cap_original) and cap_original > 0:
        y = np.minimum(y, cap_original)
    return np.clip(y, 0, None)

def compute_dynamic_cap(ts_m: pd.Series) -> float:
    """Soft cap to prevent explosions; 6x over robust baseline."""
    if ts_m.empty or (ts_m.max() <= 0):
        return np.inf
    m12 = float(ts_m.tail(12).mean()) if len(ts_m) >= 3 else float(ts_m.mean())
    med, mx = float(ts_m.median()), float(ts_m.max())
    base = max(1.0, m12, med, 1.1 * mx)
    return base * 6.0

def coalesce_language(df: pd.DataFrame) -> pd.DataFrame:
    """Use actual language column when available; else split using fixed shares."""
    if 'language' in df.columns and df['language'].notna().any() and LANGUAGE_STRATEGY == "from_column":
        df['language'] = (
            df['language']
            .astype(str).str.strip()
            .replace({'nan': None, 'None': None})
            .fillna('English')
        )
        return df
    parts = []
    for lang, w in LANGUAGE_SHARES.items():
        tmp = df.copy()
        tmp['language'] = lang
        tmp['ticket_total'] = tmp['ticket_total'] * float(w)
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True) if parts else df

def noise_score_daily(g: pd.DataFrame) -> float:
    """Robust daily noise score combining robust-CV and spike ratio."""
    s = g.sort_values('Date')['ticket_total'].astype(float)
    if len(s) < 30:
        return 0.0
    med = float(np.median(s))
    if med <= 0:
        return 0.0
    mad = float(np.median(np.abs(s - med)))
    robust_cv = (1.4826 * mad) / med
    p95 = float(np.percentile(s, 95))
    spike_ratio = (p95 / med) if med > 0 else 0.0
    return float(0.7 * robust_cv + 0.3 * (spike_ratio - 1.0))

def clean_outliers_daily(g: pd.DataFrame, method="IQR") -> pd.DataFrame:
    """Daily outlier correction: STL-residual clamp or IQR winsorization."""
    g = g.copy()
    s = g.sort_values('Date')['ticket_total'].astype(float)
    if method.upper() == "STL" and len(s) >= 60:
        ts = pd.Series(s.values, index=g.sort_values('Date')['Date'])
        try:
            stl = STL(ts, period=7, robust=True).fit()
            resid = stl.resid
            rstd = float(np.std(resid))
            y = np.where(np.abs(resid) > 3 * rstd, ts - np.sign(resid) * 3 * rstd, ts)
            g.loc[ts.index, 'ticket_total'] = np.clip(y, 0, None)
            return g
        except Exception:
            pass
    # IQR/winsorize-like clamp at 1%/99%
    ql = s.quantile(0.01)
    qh = s.quantile(0.99)
    g['ticket_total'] = s.clip(ql, qh).values
    return g

# %% [markdown]
# ## 3. Exogenous features

# %%
# ==================== Exogenous features ====================

def build_eu_core_holidays(start_date: pd.Timestamp, end_date: pd.Timestamp,
                           countries: List[str]) -> pd.DataFrame:
    """
    Minimal EU-core holiday/event calendar for [start, end].
    - Holidays: NewYear, GoodFriday, EasterMonday, LabourDay (May 1), Christmas, BoxingDay.
    - Events: BlackFriday, CyberMonday.
    We don't differentiate by country here to keep it robust.
    """
    years = list(range(start_date.year, end_date.year + 1))
    rows = []
    for y in years:
        # Holidays
        rows.append({"ds": date(y, 1, 1), "name": "NewYear", "type": "holiday", "weight": 1.0})
        easter = _easter_sunday(y)
        rows.append({"ds": easter - timedelta(days=2), "name": "GoodFriday", "type": "holiday", "weight": 1.0})
        rows.append({"ds": easter + timedelta(days=1), "name": "EasterMonday", "type": "holiday", "weight": 1.0})
        rows.append({"ds": date(y, 5, 1), "name": "LabourDay", "type": "holiday", "weight": 1.0})
        rows.append({"ds": date(y, 12, 25), "name": "Christmas", "type": "holiday", "weight": 1.0})
        rows.append({"ds": date(y, 12, 26), "name": "BoxingDay", "type": "holiday", "weight": 1.0})
        # Commercial events
        bf = _last_friday_of_november(y)
        rows.append({"ds": bf, "name": "BlackFriday", "type": "event", "weight": 0.6})
        rows.append({"ds": bf + timedelta(days=4), "name": "CyberMonday", "type": "event", "weight": 0.6})

    exo = pd.DataFrame(rows)
    exo['ds'] = pd.to_datetime(exo['ds'])
    exo = exo[(exo['ds'] >= pd.to_datetime(start_date.date())) & (exo['ds'] <= pd.to_datetime(end_date.date()))]
    return exo.reset_index(drop=True)

def load_exogenous_from_files(files: Dict[str, str],
                              start_date: pd.Timestamp,
                              end_date: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for key in ["holidays", "events"]:
        path = files.get(key, None)
        if path and os.path.exists(path):
            df = pd.read_csv(path) if path.lower().endswith('.csv') else pd.read_excel(path)
            # expected columns: ds, name, weight (weight optional -> default 1.0)
            if 'ds' not in df.columns:
                continue
            df['ds'] = pd.to_datetime(df['ds'])
            if 'name' not in df.columns:
                df['name'] = key
            if 'weight' not in df.columns:
                df['weight'] = 1.0
            df['type'] = 'holiday' if key == 'holidays' else 'event'
            df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]
            frames.append(df[['ds', 'name', 'type', 'weight']])
    if frames:
        return pd.concat(frames, ignore_index=True).drop_duplicates()
    return pd.DataFrame(columns=['ds', 'name', 'type', 'weight'])

def build_exogenous_calendar(incoming: pd.DataFrame,
                             horizon_days: int = DAILY_HORIZON_DAYS
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    - exo_daily: [ds, is_holiday, is_event, weight_h, weight_e]
    - exo_monthly: [month, hol_count, evt_count, hol_weight_sum, evt_weight_sum]
    """
    start = incoming['Date'].min() - pd.Timedelta(days=365)
    end = incoming['Date'].max() + pd.Timedelta(days=horizon_days)

    if not EXOG_ENABLE:
        exo_daily = pd.DataFrame({'ds': pd.date_range(start, end, freq='D')})
        exo_daily['is_holiday'] = 0
        exo_daily['is_event'] = 0
        exo_daily['weight_h'] = 0.0
        exo_daily['weight_e'] = 0.0
    else:
        exo = (load_exogenous_from_files(EXOG_FILES, start, end)
               if EXOG_SOURCE == "files"
               else build_eu_core_holidays(start, end, EXOG_COUNTRIES))
        cal = pd.DataFrame({'ds': pd.date_range(start, end, freq='D')})
        exo_daily = cal.merge(exo, on='ds', how='left')
        exo_daily['is_holiday'] = (exo_daily['type'] == 'holiday').astype(int)
        exo_daily['is_event'] = (exo_daily['type'] == 'event').astype(int)
        exo_daily['weight_h'] = np.where(exo_daily['is_holiday'] == 1, exo_daily['weight'].fillna(1.0), 0.0)
        exo_daily['weight_e'] = np.where(exo_daily['is_event'] == 1, exo_daily['weight'].fillna(1.0), 0.0)
        exo_daily = exo_daily[['ds', 'is_holiday', 'is_event', 'weight_h', 'weight_e']]

    exo_daily['month'] = exo_daily['ds'].dt.to_period('M')
    exo_monthly = (
        exo_daily.groupby('month', as_index=False)
        .agg(hol_count=('is_holiday', 'sum'),
             evt_count=('is_event', 'sum'),
             hol_weight_sum=('weight_h', 'sum'),
             evt_weight_sum=('weight_e', 'sum'))
    )
    return exo_daily, exo_monthly

# %% [markdown]
# ## 4. Loaders and mapping

# %%
# ==================== Loaders & mapping ====================

def load_incoming(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load daily incoming volumes from Excel/CSV.
    Expected columns (min): Date, department_id, ticket_total (or derivable).
    Coalesces language according to LANGUAGE_STRATEGY.
    """
    if not os.path.exists(path):
        msg = (
            "Incoming file not found:\n"
            f"{path}\n"
            "Please update INCOMING_SOURCE_PATH to the correct location."
        )
        raise FileNotFoundError(msg)

    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xlsm", ".xls"]:
        if not sheet_name:
            raise ValueError("Excel file detected but no sheet_name provided (e.g., 'Main').")
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported extension for incoming data: {ext}")

    # Required columns
    required = {"Date", "department_id"}
    missing = required - set(df.columns)
    if missing:
        msg = (
            "Incoming must contain columns: "
            f"{sorted(list(required))}. "
            f"Found: {list(df.columns)}. "
            f"Missing: {sorted(list(missing))}"
        )
        raise ValueError(msg)

    # ticket_total creation if missing
    if "ticket_total" not in df.columns:
        if "total_incoming" in df.columns:
            df["ticket_total"] = pd.to_numeric(df["total_incoming"], errors="coerce").fillna(0)
        elif {"incoming_from_customers", "incoming_from_transfers"}.issubset(df.columns):
            df["ticket_total"] = (
                pd.to_numeric(df["incoming_from_customers"], errors="coerce").fillna(0)
                + pd.to_numeric(df["incoming_from_transfers"], errors="coerce").fillna(0)
            )
        else:
            msg = (
                "Incoming must contain 'ticket_total' or alternatives "
                "('total_incoming' or both 'incoming_from_customers' and 'incoming_from_transfers'). "
                f"Found columns: {list(df.columns)}"
            )
            raise ValueError(msg)

    # Dtypes & clean
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        bad = df.loc[df["Date"].isna()].head(5)
        raise ValueError(f"Some Date values could not be parsed. Example rows:\n{bad}")

    df["department_id"] = df["department_id"].astype(str).str.strip()
    df["ticket_total"] = pd.to_numeric(df["ticket_total"], errors="coerce").fillna(0.0).astype(float)

    # Optional columns that downstream expects
    if "department_name" not in df.columns:
        df["department_name"] = None
    if "vertical" not in df.columns:
        df["vertical"] = None
    if "language" not in df.columns:
        df["language"] = None

    # Language handling
    df = coalesce_language(df)
    return df

def load_dept_map(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=['department_id', 'department_name', 'vertical'])

    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        mp = pd.read_excel(path, sheet_name=sheet if sheet else 0, engine="openpyxl")
    else:
        mp = pd.read_csv(path)

    rename_map = {
        'dept_id': 'department_id', 'dept_name': 'department_name', 'name': 'department_name',
        'segment': 'vertical', 'vertical_name': 'vertical'
    }
    mp = mp.rename(columns={k: v for k, v in rename_map.items() if k in mp.columns})
    if 'department_id' not in mp.columns:
        raise ValueError(f"Department map must contain 'department_id'. Found: {list(mp.columns)}")

    mp['department_id'] = mp['department_id'].astype(str).str.strip()
    if 'department_name' in mp.columns:
        mp['department_name'] = mp['department_name'].astype(str).str.strip()
    if 'vertical' in mp.columns:
        mp['vertical'] = mp['vertical'].astype(str).str.strip()

    return mp[['department_id', 'department_name', 'vertical']].drop_duplicates('department_id')

def apply_mapping(incoming: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    merged = incoming.merge(mapping, on='department_id', how='left', suffixes=('', '_map'))

    if 'department_name' not in merged.columns:
        merged['department_name'] = None
    if 'department_name_map' not in merged.columns:
        merged['department_name_map'] = None
    merged['department_name'] = merged['department_name'].fillna(merged['department_name_map']).fillna("Unknown")

    if 'vertical' not in merged.columns:
        merged['vertical'] = None
    if 'vertical_map' not in merged.columns:
        merged['vertical_map'] = None
    merged['vertical'] = merged['vertical'].fillna(merged['vertical_map']).fillna("Unmapped")

    drop_cols = [c for c in merged.columns if c.endswith('_map')]
    merged.drop(columns=drop_cols, inplace=True, errors='ignore')
    return merged

def load_productivity(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Productivity file not found: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    req = {'Date', 'agent_id', 'department_id', 'prod_total_model'}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"productivity_agents.xlsx missing columns: {sorted(list(missing))}. Found: {list(df.columns)}")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['department_id'] = df['department_id'].astype(str).str.strip()
    df['prod_total_model'] = pd.to_numeric(df['prod_total_model'], errors='coerce')

    prod_dept = (
        df.groupby('department_id', as_index=False)['prod_total_model']
        .mean().rename(columns={'prod_total_model': 'avg_tickets_per_agent_day'})
    )
    return prod_dept

# %% [markdown]
# ## 5. Aggregations (montly/weekly) and profiles

# %%
# ==================== Aggregations ====================

def build_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['month'] = d['Date'].dt.to_period('M')
    monthly = (
        d.groupby(['department_id', 'language', 'month'], as_index=False)['ticket_total']
        .sum().rename(columns={'ticket_total': 'incoming_monthly'})
    )
    return monthly

def build_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().set_index('Date').sort_index()
    rows = []
    for (dept, lang), g in d.groupby([d['department_id'], d['language']]):
        s = g['ticket_total'].resample(WEEKLY_FREQ).sum().dropna()
        if not s.empty:
            tmp = pd.DataFrame({
                'department_id': dept,
                'language': lang,
                'week': s.index.to_period('W-WED'),
                'incoming_weekly': s.values
            })
            rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=['department_id', 'language', 'week', 'incoming_weekly']
    )

def dow_profile(g: pd.DataFrame) -> pd.Series:
    prof = (g.assign(dow=g['Date'].dt.dayofweek)
              .groupby('dow')['ticket_total'].mean())
    if prof.notna().sum() >= 3:
        return prof / prof.mean()
    return pd.Series(1.0, index=range(7))

# %% [markdown]
# ## 6. Monthly modeling (Prophet/ARIMA/TBATS-ETS) + blending

# %%
# ==================== Modeling (monthly) ====================

def winsorize_monthly(ts_m: pd.Series, lower_q: float = IQR_LO, upper_q: float = IQR_HI) -> pd.Series:
    if ts_m.empty:
        return ts_m
    lo = ts_m.quantile(lower_q)
    hi = ts_m.quantile(upper_q)
    return ts_m.clip(lower=lo, upper=hi)

def prepare_monthly_exog(exo_monthly: pd.DataFrame, ts_index: pd.PeriodIndex) -> pd.DataFrame:
    ex = exo_monthly.copy()
    if not pd.api.types.is_period_dtype(ex['month']):
        ex['month'] = pd.PeriodIndex(ex['month'], freq='M')
    ex = ex.set_index('month').reindex(ts_index).fillna(0.0)
    return ex[['hol_count', 'evt_count', 'hol_weight_sum', 'evt_weight_sum']]

def fit_prophet_monthly_log(ts_m: pd.Series, exo_m: Optional[pd.DataFrame] = None):
    if Prophet is None or len(ts_m) < 6:
        return None, None
    y = np.log1p(ts_m.values)
    dfp = pd.DataFrame({'ds': ts_m.index.to_timestamp(), 'y': y})
    if exo_m is not None:
        dfp = dfp.join(exo_m, on=ts_m.index).reset_index(drop=True)

    m = Prophet(weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
    if exo_m is not None:
        for col in exo_m.columns:
            m.add_regressor(col)
    m.fit(dfp)

    def fcast(h_months=H_MONTHS):
        future = m.make_future_dataframe(periods=h_months, freq='MS')
        if exo_m is not None:
            # forward-fill exo for horizon with last row
            idx_future = pd.period_range(ts_m.index[-1] + 1, periods=h_months, freq='M')
            pad = exo_m.iloc[[-1]].repeat(h_months)
            pad.index = idx_future
            ex_full = pd.concat([exo_m, pad], axis=0)
            for col in exo_m.columns:
                future[col] = ex_full[col].values

        pred_df = m.predict(future)
        pred_df.index = pd.PeriodIndex(pred_df['ds'], freq='M')
        pred = pred_df['yhat'].iloc[-h_months:]
        cap = compute_dynamic_cap(ts_m)
        vals = expm1_safe(pred.values, cap_original=cap)
        return pd.Series(vals, index=pred.index)

    return m, fcast

def fit_arima_monthly_log(ts_m: pd.Series, exo_m: Optional[pd.DataFrame] = None):
    y = np.log1p(ts_m)
    best_aic, best_model, best_exog = np.inf, None, None
    pqs = [0, 1]
    seasonal = len(ts_m) >= 12
    PsQs = [0] if seasonal else [0]
    for p in pqs:
        for d in ([1] if len(ts_m) < 24 else [0, 1]):
            for q in pqs:
                for P in PsQs:
                    for D in ([0, 1] if seasonal else [0]):
                        for Q in PsQs:
                            try:
                                model = SARIMAX(
                                    y, order=(p, d, q),
                                    seasonal_order=(P, D, Q, 12 if seasonal else 0),
                                    exog=exo_m.values if exo_m is not None else None,
                                    enforce_stationarity=False, enforce_invertibility=False
                                ).fit(disp=False)
                                if model.aic < best_aic:
                                    best_aic = model.aic
                                    best_model = model
                                    best_exog = exo_m
                            except Exception:
                                continue

    def fcast(h_months=H_MONTHS):
        if best_model is None:
            idx = pd.period_range(ts_m.index[-1] + 1, periods=h_months, freq='M')
            return pd.Series([float(np.exp(y).mean())] * h_months, index=idx)
        idx = pd.period_range(ts_m.index[-1] + 1, periods=h_months, freq='M')
        exog_future = None
        if best_exog is not None:
            pad = best_exog.iloc[[-1]].repeat(h_months)
            pad.index = idx
            exog_future = pad.values
        fc_log = best_model.get_forecast(h_months, exog=exog_future).predicted_mean
        cap = compute_dynamic_cap(ts_m)
        fc = expm1_safe(fc_log, cap_original=cap)
        return pd.Series(fc, index=idx)

    return best_model, fcast

def fit_tbats_or_ets_monthly_log(ts_m: pd.Series):
    y_log = np.log1p(ts_m)

    if TBATS is not None and len(ts_m) >= 12:
        y_log_ts = pd.Series(y_log.values, index=ts_m.index.to_timestamp())
        estimator = TBATS(use_arma_errors=False, seasonal_periods=[12])
        model = estimator.fit(y_log_ts)

        def fcast(h_months=H_MONTHS):
            vals_log = model.forecast(steps=h_months)
            idx = pd.period_range(ts_m.index[-1] + 1, periods=h_months, freq='M')
            cap = compute_dynamic_cap(ts_m)
            vals = expm1_safe(vals_log, cap_original=cap)
            return pd.Series(vals, index=idx)

        return model, fcast

    seasonal = 12 if len(ts_m) >= 24 else None
    model = ExponentialSmoothing(y_log, trend='add',
                                 seasonal=('add' if seasonal else None),
                                 seasonal_periods=seasonal).fit()

    def fcast(h_months=H_MONTHS):
        vals_log = model.forecast(h_months)
        idx = pd.period_range(ts_m.index[-1] + 1, periods=h_months, freq='M')
        cap = compute_dynamic_cap(ts_m)
        vals = expm1_safe(vals_log, cap_original=cap)
        return pd.Series(vals, index=idx)

    return model, fcast

def rolling_cv_monthly_adaptive(ts_m: pd.Series,
                                exo_monthly: Optional[pd.DataFrame] = None
                                ) -> Optional[Dict[str, float]]:
    """Small rolling CV for sMAPE across candidates; short series are skipped."""
    n = len(ts_m)
    if n < 9:
        return None
    h = 3 if n >= 15 else 1
    min_train = max(12, n - (h + 2))
    splits = []
    for start in range(min_train, n - h + 1):
        train = ts_m.iloc[:start]
        test = ts_m.iloc[start:start + h]
        metrics = {}
        ex_train = ex_test = None
        if exo_monthly is not None:
            ex_train = exo_monthly.iloc[:start]
            ex_test = exo_monthly.iloc[start:start + h]

        mp, fp = fit_prophet_monthly_log(train, ex_train)
        if fp is not None:
            try:
                pv = np.array(fp(h_months=h).values[:h], dtype=float)
                pv[~np.isfinite(pv)] = np.nan
                metrics['Prophet'] = 200.0 if np.isnan(pv).all() else smape(test.values, np.nan_to_num(pv, nan=0.0))
            except Exception:
                metrics['Prophet'] = 200.0

        try:
            ma, fa = fit_arima_monthly_log(train, ex_train)
            pv = np.array(fa(h_months=h).values[:h], dtype=float)
            pv[~np.isfinite(pv)] = np.nan
            metrics['ARIMA'] = 200.0 if np.isnan(pv).all() else smape(test.values, np.nan_to_num(pv, nan=0.0))
        except Exception:
            metrics['ARIMA'] = 200.0

        try:
            mt, ft = fit_tbats_or_ets_monthly_log(train)
            pv = np.array(ft(h_months=h).values[:h], dtype=float)
            pv[~np.isfinite(pv)] = np.nan
            metrics['TBATS/ETS'] = 200.0 if np.isnan(pv).all() else smape(test.values, np.nan_to_num(pv, nan=0.0))
        except Exception:
            metrics['TBATS/ETS'] = 200.0

        splits.append(metrics)
    dfm = pd.DataFrame(splits)
    return dfm.mean().to_dict()

def select_or_blend_forecasts(fc_dict: Dict[str, pd.Series],
                              cv_scores: Dict[str, float],
                              blend: bool = True):
    """
    Inverse-error weighted blending; robust to empty or NaN CV scores.
    Fallbacks:
      - If no CV scores at all -> uniform averaging across all candidates (if blend=True),
        else pick the first available model deterministically.
      - If all inverted weights sum to 0 -> fallback to winner by min score when available,
        otherwise uniform average.
    """
    # Safety: if no forecast candidates, raise early (should not happen upstream)
    if not fc_dict:
        raise ValueError("select_or_blend_forecasts: no forecast candidates provided.")

    # Filter/normalize scores
    scores = {k: float(v) for k, v in (cv_scores or {}).items()
              if v is not None and np.isfinite(v)}

    # If blending is disabled, pick deterministically the first model
    if not blend:
        winner = next(iter(fc_dict.keys()))
        return fc_dict[winner], {'winner': winner, 'weights': {winner: 1.0}}

    # Case A: No usable CV scores -> uniform average across all models
    if len(scores) == 0:
        # Uniform weights across available forecasts
        keys = list(fc_dict.keys())
        idx = None
        for s in fc_dict.values():
            idx = s.index if idx is None else idx.union(s.index)
        w = {k: 1.0 / len(keys) for k in keys}
        blended = sum(w[k] * fc_dict[k].reindex(idx).fillna(0) for k in keys)
        # Winner is arbitrary: choose the first key for traceability
        winner = keys[0]
        return blended, {'winner': winner, 'weights': w}

    # Case B: We have scores -> inverse-error weighting
    inv = {k: (1.0 / v if v > 0 else 0.0) for k, v in scores.items()}
    total = sum(inv.values())

    # If all weights collapsed to zero (e.g., all scores are zero/identical in a corner case)
    if total == 0:
        # Try pick the smallest score as winner if possible
        try:
            winner = min(scores, key=scores.get)
            return fc_dict[winner], {'winner': winner, 'weights': {winner: 1.0}}
        except ValueError:
            # Fallback again to uniform average across *all* candidates
            keys = list(fc_dict.keys())
            idx = None
            for s in fc_dict.values():
                idx = s.index if idx is None else idx.union(s.index)
            w = {k: 1.0 / len(keys) for k in keys}
            blended = sum(w[k] * fc_dict[k].reindex(idx).fillna(0) for k in keys)
            winner = keys[0]
            return blended, {'winner': winner, 'weights': w}

    # Normal path: compute normalized weights over the models for which we have scores
    w = {k: inv[k] / total for k in inv}

    # Build union index across all candidate forecasts to avoid NaNs
    idx = None
    for s in fc_dict.values():
        idx = s.index if idx is None else idx.union(s.index)

    # Note: if some model is missing in scores (no CV), give it tiny weight (0) implicitly
    blended = sum(w.get(k, 0.0) * fc_dict[k].reindex(idx).fillna(0) for k in fc_dict)

    # Winner: the one with minimum score among those with scores
    winner = min(scores, key=scores.get)
    return blended, {'winner': winner, 'weights': w}

# %% [markdown]
# ## 7. Week candidate (noisiest series)

# %%
# ==================== Weekly candidate (ETS weekly) ====================

def forecast_weekly_candidate(g_daily: pd.DataFrame, horizon_months=H_MONTHS) -> Optional[pd.Series]:
    if len(g_daily) < 60:
        return None

    w = (g_daily.set_index('Date').sort_index()['ticket_total']
         .resample(WEEKLY_FREQ).sum().dropna())
    if len(w) < 30:
        return None

    y_log = np.log1p(w)
    try:
        model = ExponentialSmoothing(y_log, trend='add', seasonal=None).fit()
    except Exception:
        return None

    last_day = g_daily['Date'].max()
    end_date = (last_day + pd.offsets.MonthEnd(horizon_months)).to_pydatetime()
    steps = int(np.ceil((pd.Timestamp(end_date) - w.index[-1].to_timestamp()).days / 7)) + 4
    vals_log = model.forecast(steps=max(steps, 12))
    vals = np.expm1(vals_log)
    wf = pd.Series(vals, index=pd.period_range(vals.index[0], periods=len(vals), freq='W-WED'))

    prof = dow_profile(g_daily)
    future_days = pd.date_range(start=last_day + pd.Timedelta(days=1), end=end_date, freq='D')
    df_daily = pd.DataFrame({'Date': future_days})
    df_daily['week'] = df_daily['Date'].to_period('W-WED')

    # Map weekly forecast to daily with DOW profile weights
    wk_map = wf.to_timestamp().rename('week_total')
    df_daily = df_daily.merge(
        wk_map.reset_index().rename(columns={'index': 'Date'}),
        left_on='week', right_on=pd.PeriodIndex(wf.index, freq='W-WED').to_timestamp(), how='left'
    )
    df_daily['dow'] = df_daily['Date'].dt.dayofweek
    df_daily['w_sum'] = df_daily.groupby('week')['dow'].transform(lambda x: (prof.reindex(x).fillna(1.0)).sum())
    df_daily['w_w'] = np.where(df_daily['w_sum'] > 0,
                               (prof.reindex(df_daily['dow']).fillna(1.0)).values / df_daily['w_sum'],
                               1.0)
    df_daily['forecast_daily'] = df_daily['week_total'] * df_daily['w_w']
    df_daily['month'] = df_daily['Date'].dt.to_period('M')
    monthly_candidate = df_daily.groupby('month')['forecast_daily'].sum()
    monthly_candidate = monthly_candidate.iloc[:H_MONTHS]
    monthly_candidate.index = pd.PeriodIndex(monthly_candidate.index, freq='M')
    return monthly_candidate

# %% [markdown]
# ## 8. Daily pannel, features and Global LightGBM

# %%
# ==================== Global LightGBM per vertical (with daily exogenous) ====================

def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['dayofweek'] = d['Date'].dt.dayofweek
    d['weekofyear'] = d['Date'].dt.isocalendar().week.astype(int)
    d['month'] = d['Date'].dt.month
    d['year'] = d['Date'].dt.year
    return d


def add_lags_and_rolls(d: pd.DataFrame, group_cols: List[str], target_col='y') -> pd.DataFrame:
    d = d.sort_values(['Date']).copy()
    for lag in LGB_LAGS:
        d[f'lag_{lag}'] = d.groupby(group_cols)[target_col].shift(lag)
    for w in LGB_ROLLS:
        d[f'roll_mean_{w}'] = d.groupby(group_cols)[target_col].shift(1).rolling(w).mean()
        d[f'roll_std_{w}']  = d.groupby(group_cols)[target_col].shift(1).rolling(w).std()
    return d


def prepare_daily_panel(incoming: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    df = incoming.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['department_id'] = df['department_id'].astype(str).str.strip()
    df['ticket_total'] = pd.to_numeric(df['ticket_total'], errors='coerce').fillna(0.0)

    df = apply_mapping(df, mapping)
    has_lang = ('language' in df.columns) and (df['language'].notna().any())
    if LANGUAGE_STRATEGY == 'from_column' and has_lang:
        df['language'] = df['language'].fillna('Unknown').astype(str).str.strip()
        panel = (
            df.groupby(['Date', 'department_id', 'language', 'vertical'], as_index=False)['ticket_total']
              .sum()
              .rename(columns={'ticket_total': 'y'})
        )
    else:
        base = df.groupby(['Date', 'department_id', 'vertical'], as_index=False)['ticket_total'].sum()
        parts = []
        for lang, w in LANGUAGE_SHARES.items():
            tmp = base.copy()
            tmp['language'] = lang
            tmp['y'] = tmp['ticket_total'] * float(w)
            parts.append(tmp)
        panel = pd.concat(parts, ignore_index=True)
        panel = panel[['Date', 'department_id', 'language', 'vertical', 'y']]

    panel['y'] = panel['y'].clip(lower=0.0)
    return panel


def merge_exogenous_daily(panel: pd.DataFrame, exo_daily: pd.DataFrame) -> pd.DataFrame:
    exo_d = exo_daily.rename(columns={'ds': 'Date'}).copy()
    d = panel.merge(exo_d, on='Date', how='left')
    d[['is_holiday', 'is_event', 'weight_h', 'weight_e']] = d[['is_holiday', 'is_event', 'weight_h', 'weight_e']].fillna(0.0)
    return d

def train_global_lgb_per_vertical(incoming_clean: pd.DataFrame, mapping: pd.DataFrame,
                                  exo_daily: pd.DataFrame, h_days=DAILY_HORIZON_DAYS
                                  ) -> Dict[str, pd.DataFrame]:
    """
    Train a global LightGBM model per vertical and produce daily forecasts by (department_id, language).
    Fix: ensure categorical features match exactly between training and prediction (dtype + categories).
    Además: endurecemos la construcción de 'future' tras el stack() para evitar 'Length mismatch'.
    """
    if not ENABLE_GLOBAL_LGB or lgb is None:
        return {}

    res = {}

    # Horizonte de días futuro
    last_date = incoming_clean['Date'].max()
    start = last_date + pd.Timedelta(days=1)
    future_idx = pd.date_range(start=start, periods=h_days, freq='D')

    # Aseguramos mapeo mínimo requerido
    data = apply_mapping(incoming_clean, mapping)
    data = data[['Date', 'department_id', 'language', 'vertical', 'ticket_total']].copy()

    # Panel diario + exógenas
    panel = prepare_daily_panel(data.rename(columns={'ticket_total': 'ticket_total'}), mapping)
    panel_exo = merge_exogenous_daily(panel, exo_daily)

    # Lookup exógeno para el horizonte
    exo_lookup = (
        exo_daily.set_index('ds')[['is_holiday', 'is_event', 'weight_h', 'weight_e']]
        .reindex(future_idx, fill_value=0.0)
    )

    # Entrenamiento por vertical
    for vert, g in panel_exo.groupby('vertical'):
        if len(g) < 60:
            continue

        g = build_time_features(g.copy())
        g = add_lags_and_rolls(g, ['department_id', 'language'], target_col='y')

        feat_cols = [
            'dayofweek', 'weekofyear', 'month', 'year', 'department_id', 'language',
            'is_holiday', 'is_event', 'weight_h', 'weight_e'
        ]
        feat_cols += [c for c in g.columns if c.startswith('lag_') or c.startswith('roll_')]

        # ---- Categóricas estables (conjunto "congelado") ----
        cat_cols = ['department_id', 'language', 'dayofweek', 'month']
        cat_levels = {
            'department_id': pd.Categorical(g['department_id'].astype(str)).categories,
            'language': pd.Categorical(g['language'].astype(str)).categories,
            'dayofweek': pd.Index(range(7)),     # 0..6
            'month': pd.Index(range(1, 13)),     # 1..12
        }

        # ---- Tabla de entrenamiento y casteo de dtypes categóricos ----
        g_train = g.dropna(subset=feat_cols + ['y']).copy()
        if len(g_train) < 200:
            continue

        X = g_train[feat_cols].copy()
        y = g_train['y'].copy()

        if 'department_id' in X.columns:
            X['department_id'] = pd.Categorical(X['department_id'].astype(str),
                                                categories=cat_levels['department_id'])
        if 'language' in X.columns:
            X['language'] = pd.Categorical(X['language'].astype(str),
                                           categories=cat_levels['language'])
        if 'dayofweek' in X.columns:
            X['dayofweek'] = pd.Categorical(X['dayofweek'].astype(int),
                                            categories=cat_levels['dayofweek'])
        if 'month' in X.columns:
            X['month'] = pd.Categorical(X['month'].astype(int),
                                        categories=cat_levels['month'])

        # ---- CV ligera ----
        tscv = TimeSeriesSplit(n_splits=2)
        cv_smape = []
        for tr, va in tscv.split(X):
            model = lgb.LGBMRegressor(
                n_estimators=LGB_N_ESTIMATORS,
                learning_rate=LGB_LEARNING_RATE,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(
                X.iloc[tr], y.iloc[tr],
                eval_set=[(X.iloc[va], y.iloc[va])],
                categorical_feature=[c for c in cat_cols if c in X.columns],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            pred = model.predict(X.iloc[va], num_iteration=model.best_iteration_)
            cv_smape.append(smape(y.iloc[va].values, np.maximum(pred, 0.0)))

        lgb_cv_smape = float(np.mean(cv_smape)) if cv_smape else np.nan

        # ---- Modelo final ----
        model = lgb.LGBMRegressor(
            n_estimators=LGB_N_ESTIMATORS,
            learning_rate=LGB_LEARNING_RATE,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y, categorical_feature=[c for c in cat_cols if c in X.columns])

        best_iter = getattr(model, "best_iteration_", None)
        pred_kwargs = {'num_iteration': best_iter} if best_iter is not None else {}

        # ---- Bucle autoregresivo por (department_id, language) ----
        hist = (
            g[['Date', 'department_id', 'language', 'y']]
            .pivot_table(index='Date', columns=['department_id', 'language'], values='y')
            .asfreq('D').fillna(0.0)
        )
        hist_future = pd.DataFrame(index=future_idx, columns=hist.columns, dtype=float)

        for d in future_idx:
            dayofweek = int(d.dayofweek)
            weekofyear = int(d.isocalendar().week)
            month = int(d.month)
            year = int(d.year)
            ex_row = exo_lookup.loc[d]

            preds = {}

            for (dept, lang) in hist.columns:
                series = (pd.concat([hist[(dept, lang)], hist_future[(dept, lang)]], axis=0).dropna())

                feats = {
                    'dayofweek': dayofweek,
                    'weekofyear': weekofyear,
                    'month': month,
                    'year': year,
                    'department_id': str(dept),
                    'language': str(lang),
                    'is_holiday': float(ex_row['is_holiday']),
                    'is_event': float(ex_row['is_event']),
                    'weight_h': float(ex_row['weight_h']),
                    'weight_e': float(ex_row['weight_e']),
                }

                # Lags
                for lag in LGB_LAGS:
                    if len(series) >= lag:
                        feats[f'lag_{lag}'] = float(series.iloc[-lag])
                    else:
                        feats[f'lag_{lag}'] = float(series.mean()) if len(series) else 0.0

                # Rolls
                for w in LGB_ROLLS:
                    if len(series) > 1:
                        tail = series.iloc[:-1].tail(w)
                        feats[f'roll_mean_{w}'] = float(tail.mean()) if len(tail) else float(series.mean())
                        feats[f'roll_std_{w}'] = float(tail.std()) if len(tail) else 0.0
                    else:
                        feats[f'roll_mean_{w}'] = float(series.mean()) if len(series) else 0.0
                        feats[f'roll_std_{w}'] = 0.0

                fx = pd.DataFrame([feats])

                # Match dtypes y categorías exactamente
                fx['department_id'] = pd.Categorical(fx['department_id'].astype(str), categories=cat_levels['department_id'])
                fx['language'] = pd.Categorical(fx['language'].astype(str), categories=cat_levels['language'])
                fx['dayofweek'] = pd.Categorical(fx['dayofweek'].astype(int), categories=cat_levels['dayofweek'])
                fx['month'] = pd.Categorical(fx['month'].astype(int), categories=cat_levels['month'])

                # Orden exacto de columnas como en entrenamiento
                fx = fx.reindex(columns=feat_cols, fill_value=0)

                # Predicción
                yhat = float(model.predict(fx, **pred_kwargs)[0])
                preds[(dept, lang)] = max(0.0, yhat)

            hist_future.loc[d] = pd.Series(preds)

        # --- Construcción robusta de 'future' evitando Length mismatch ---
        # (Opcional) Trazas previas para entender la forma si cambia
        print("DEBUG hist_future.shape:", hist_future.shape)
        try:
            print("DEBUG hist_future.columns levels:", getattr(hist_future.columns, 'names', None), type(hist_future.columns))
        except Exception:
            pass

        future = hist_future.stack().reset_index()

        # Trazas para validar
        print("DEBUG future.shape:", future.shape)
        print("DEBUG future.columns:", list(future.columns))

        # Normalización a las 4 columnas esperadas
        expected_cols = ['Date', 'department_id', 'language', 'forecast_daily_lgb']

        if len(future.columns) == 4:
            # Renombra por posición si vinieron como [0,1,2,0] o similares
            future.columns = expected_cols

        elif len(future.columns) > 4:
            # Intento 1: renombrado semántico por nombre
            rename_map = {}
            for c in future.columns:
                lc = str(c).lower()
                if lc in ('date', 'ds'):
                    rename_map[c] = 'Date'
                elif lc in ('department_id', 'dept_id', 'dept'):
                    rename_map[c] = 'department_id'
                elif lc in ('language', 'lang'):
                    rename_map[c] = 'language'
            future = future.rename(columns=rename_map)

            # Si ya están las 3 claves por nombre, identifica la columna de valores
            if set(['Date', 'department_id', 'language']).issubset(future.columns):
                value_col = None
                for c in future.columns:
                    if c not in ('Date', 'department_id', 'language'):
                        value_col = c
                        break
                if value_col is None:
                    raise ValueError(f"No value column found in 'future' after stack(); columns={list(future.columns)}")
                future = future[['Date', 'department_id', 'language', value_col]].copy()
                future.columns = expected_cols
            else:
                # Fallback por posición controlada
                future = future.iloc[:, :4].copy()
                future.columns = expected_cols

        else:
            # Menos de 4 columnas => error claro para diagnosticar aguas arriba
            raise ValueError(f"Unexpected shape for 'future': {future.shape} | columns={list(future.columns)}")

        # Metadatos
        future['vertical'] = str(vert)
        future['lgb_cv_smape'] = lgb_cv_smape

        res[str(vert)] = future

    return res


    # Build daily panel + exogenous
    panel = prepare_daily_panel(data.rename(columns={'ticket_total': 'ticket_total'}), mapping)
    panel_exo = merge_exogenous_daily(panel, exo_daily)

    # Exogenous lookup (vectorized)
    exo_lookup = (
        exo_daily.set_index('ds')[['is_holiday', 'is_event', 'weight_h', 'weight_e']]
        .reindex(future_idx, fill_value=0.0)
    )

    for vert, g in panel_exo.groupby('vertical'):
        if len(g) < 60:
            continue

        g = build_time_features(g.copy())
        g = add_lags_and_rolls(g, ['department_id', 'language'], target_col='y')

        feat_cols = [
            'dayofweek', 'weekofyear', 'month', 'year', 'department_id', 'language',
            'is_holiday', 'is_event', 'weight_h', 'weight_e'
        ]
        feat_cols += [c for c in g.columns if c.startswith('lag_') or c.startswith('roll_')]

        # ---- Define stable categorical levels (frozen) ----
        cat_cols = ['department_id', 'language', 'dayofweek', 'month']
        # From the full 'g' to include all categories even if some rows are dropped later
        cat_levels = {
            'department_id': pd.Categorical(g['department_id'].astype(str)).categories,
            'language':     pd.Categorical(g['language'].astype(str)).categories,
            'dayofweek':    pd.Index(range(7)),     # 0..6
            'month':        pd.Index(range(1, 13)), # 1..12
        }

        # ----- Build training table and align categorical dtypes to frozen levels -----
        g_train = g.dropna(subset=feat_cols + ['y']).copy()
        if len(g_train) < 200:
            continue

        X = g_train[feat_cols].copy()
        y = g_train['y'].copy()

        # Cast to 'category' with frozen categories
        if 'department_id' in X.columns:
            X['department_id'] = pd.Categorical(X['department_id'].astype(str), categories=cat_levels['department_id'])
        if 'language' in X.columns:
            X['language'] = pd.Categorical(X['language'].astype(str), categories=cat_levels['language'])
        if 'dayofweek' in X.columns:
            X['dayofweek'] = pd.Categorical(X['dayofweek'].astype(int), categories=cat_levels['dayofweek'])
        if 'month' in X.columns:
            X['month'] = pd.Categorical(X['month'].astype(int), categories=cat_levels['month'])

        # ---- CV (light) ----
        tscv = TimeSeriesSplit(n_splits=2)
        cv_smape = []
        for tr, va in tscv.split(X):
            model = lgb.LGBMRegressor(
                n_estimators=LGB_N_ESTIMATORS,
                learning_rate=LGB_LEARNING_RATE,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            # Early stopping for speed (optional, safe)
            model.fit(
                X.iloc[tr], y.iloc[tr],
                eval_set=[(X.iloc[va], y.iloc[va])],
                categorical_feature=[c for c in cat_cols if c in X.columns],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            pred = model.predict(X.iloc[va], num_iteration=model.best_iteration_)
            cv_smape.append(smape(y.iloc[va].values, np.maximum(pred, 0.0)))
        lgb_cv_smape = float(np.mean(cv_smape)) if cv_smape else np.nan

        # ---- Train final model (use best_iteration_ if available) ----
        model = lgb.LGBMRegressor(
            n_estimators=LGB_N_ESTIMATORS,
            learning_rate=LGB_LEARNING_RATE,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y, categorical_feature=[c for c in cat_cols if c in X.columns])

        # Keep a safe number of iterations for prediction (handles the case with early stopping in CV)
        best_iter = getattr(model, "best_iteration_", None)
        pred_kwargs = {'num_iteration': best_iter} if best_iter is not None else {}

        # ---- Autoregressive forecasting per (dept, lang) ----
        hist = (
            g[['Date', 'department_id', 'language', 'y']]
            .pivot_table(index='Date', columns=['department_id', 'language'], values='y')
            .asfreq('D').fillna(0.0)
        )
        hist_future = pd.DataFrame(index=future_idx, columns=hist.columns, dtype=float)

        for d in future_idx:
            dayofweek = int(d.dayofweek)
            weekofyear = int(d.isocalendar().week)
            month = int(d.month)
            year = int(d.year)

            ex_row = exo_lookup.loc[d]
            preds = {}

            # Iterate each (dept, lang) time series
            for (dept, lang) in hist.columns:
                series = (pd.concat([hist[(dept, lang)], hist_future[(dept, lang)]], axis=0).dropna())

                feats = {
                    'dayofweek': dayofweek,
                    'weekofyear': weekofyear,
                    'month': month,
                    'year': year,
                    'department_id': str(dept),
                    'language': str(lang),
                    'is_holiday': float(ex_row['is_holiday']),
                    'is_event': float(ex_row['is_event']),
                    'weight_h': float(ex_row['weight_h']),
                    'weight_e': float(ex_row['weight_e']),
                }
                # Lags
                for lag in LGB_LAGS:
                    if len(series) >= lag:
                        feats[f'lag_{lag}'] = float(series.iloc[-lag])
                    else:
                        feats[f'lag_{lag}'] = float(series.mean()) if len(series) else 0.0
                # Rolls
                for w in LGB_ROLLS:
                    if len(series) > 1:
                        tail = series.iloc[:-1].tail(w)
                        feats[f'roll_mean_{w}'] = float(tail.mean()) if len(tail) else float(series.mean())
                        feats[f'roll_std_{w}']  = float(tail.std()) if len(tail) else 0.0
                    else:
                        feats[f'roll_mean_{w}'] = float(series.mean()) if len(series) else 0.0
                        feats[f'roll_std_{w}']  = 0.0

                fx = pd.DataFrame([feats])

                # --- Critical: match categorical dtypes and categories exactly to training ----
                fx['department_id'] = pd.Categorical(fx['department_id'].astype(str), categories=cat_levels['department_id'])
                fx['language']      = pd.Categorical(fx['language'].astype(str),      categories=cat_levels['language'])
                fx['dayofweek']     = pd.Categorical(fx['dayofweek'].astype(int),     categories=cat_levels['dayofweek'])
                fx['month']         = pd.Categorical(fx['month'].astype(int),         categories=cat_levels['month'])

                # Ensure column order matches training features (for safety)
                fx = fx.reindex(columns=feat_cols, fill_value=0)

                # Predict
                yhat = float(model.predict(fx, **pred_kwargs)[0])
                preds[(dept, lang)] = max(0.0, yhat)

            hist_future.loc[d] = pd.Series(preds)

        future = hist_future.stack().reset_index()
        future.columns = ['Date', 'department_id', 'language', 'forecast_daily_lgb']
        future['vertical'] = str(vert)
        future['lgb_cv_smape'] = lgb_cv_smape
        res[str(vert)] = future

    return res

# %% [markdown]
# ## 9. Monthly reconciliation  - daily and language split (dynamic)

# %%
# ==================== Top-down reconciliation (monthly->daily) ====================

def disaggregate_month_to_days(group_hist: pd.DataFrame, month_period: pd.Period, target_sum: float) -> pd.DataFrame:
    start = month_period.start_time
    end = month_period.end_time
    days = pd.date_range(start=start, end=end, freq='D')
    hist = group_hist.sort_values('Date').tail(90)
    profile = dow_profile(hist)
    weights = np.array([profile.get(d.dayofweek, 1.0) for d in days], dtype=float)
    weights = np.maximum(weights, 1e-6)
    weights = weights / weights.sum()
    alloc = target_sum * weights
    return pd.DataFrame({'Date': days, 'forecast_daily': alloc})

def build_daily_from_monthly(incoming: pd.DataFrame, fc_monthly: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    last_date = incoming['Date'].max()
    start = last_date + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=horizon_days - 1)
    future_months = pd.period_range(start=start.to_period('M'), end=end.to_period('M'), freq='M')

    rows = []
    for (dept, lang), g in incoming.groupby(['department_id', 'language']):
        for m in future_months:
            fcm = fc_monthly[
                (fc_monthly['department_id'] == dept) &
                (fc_monthly['language'] == lang) &
                (fc_monthly['month'] == m)
            ]
            if fcm.empty:
                continue
            target = float(fcm['forecast_monthly'].iloc[0])
            if target <= 0:
                continue
            alloc_df = disaggregate_month_to_days(g[['Date', 'ticket_total']], m, target)
            alloc_df = alloc_df[(alloc_df['Date'] >= start) & (alloc_df['Date'] <= end)]
            alloc_df.insert(0, 'department_id', dept)
            alloc_df.insert(1, 'language', lang)
            rows.append(alloc_df)

    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=['department_id', 'language', 'Date', 'forecast_daily']
    )
    return df

# ==================== Language split (dynamic shares) ====================

def build_language_shares_from_actuals(incoming: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    if 'language' not in incoming.columns:
        return pd.DataFrame(columns=['department_id', 'language', 'share'])
    df = incoming.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    cutoff = df['Date'].max() - pd.Timedelta(days=window_days)
    df = df[df['Date'] >= cutoff]

    g = (df.groupby(['department_id', 'language'], as_index=False)['ticket_total']
           .sum().rename(columns={'ticket_total': 'sum_lang'}))
    tot = (g.groupby('department_id', as_index=False)['sum_lang'].sum()
             .rename(columns={'sum_lang': 'sum_dept'}))
    m = g.merge(tot, on='department_id', how='left')
    m['share'] = np.where(m['sum_dept'] > 0, m['sum_lang'] / m['sum_dept'], np.nan)
    return m[['department_id', 'language', 'share']]

def split_daily_by_language(daily_fc: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for lang, w in LANGUAGE_SHARES.items():
        tmp = daily_fc.copy()
        tmp['language'] = lang
        tmp['forecast_daily_language'] = tmp['forecast_daily'] * float(w)
        parts.append(tmp)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out

def split_daily_forecast_by_language_dynamic(daily_fc: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if LANGUAGE_STRATEGY != 'from_column' or 'language' not in incoming.columns:
        return split_daily_by_language(daily_fc)

    shares = build_language_shares_from_actuals(incoming)
    if shares.empty:
        return split_daily_by_language(daily_fc)

    rows = []
    for dept, g in daily_fc.groupby('department_id'):
        s = shares[shares['department_id'] == dept].dropna(subset=['share']).copy()
        if s.empty:
            rows.append(split_daily_by_language(g))
            continue
        s['share'] = s['share'] / s['share'].sum()
        for _, r in g.iterrows():
            for _, row in s.iterrows():
                rows.append({
                    'Date': r['Date'],
                    'department_id': dept,
                    'language': row['language'],
                    'forecast_daily_language': float(r['forecast_daily']) * float(row['share'])
                })
    out = pd.DataFrame(rows)
    return out

# %% [markdown]
# ## 10. Daily Blending (estatistic vs ML) and metrics

# %%
# ==================== Daily blending (stat vs ML) ====================

def compute_series_noise(panel: pd.DataFrame) -> pd.DataFrame:
    """Noise proxy using sMAPE of seasonal-naive(7d) on last h days."""
    scores = []
    for (dept, lang), g in panel.groupby(['department_id', 'language']):
        s = g.sort_values('Date')['y']
        if len(s) < 21:
            sc = np.nan
        else:
            h = min(7, len(s) // 4) or 1
            y_true = s.tail(h).values
            y_pred = s.shift(7).tail(h).fillna(method='ffill').fillna(0.0).values
            sc = smape(y_true, y_pred)
        scores.append({'department_id': dept, 'language': lang, 'noise_smape': sc})
    return pd.DataFrame(scores)

def blend_daily_predictions(daily_stat: pd.DataFrame, ml_forecasts: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    noise = compute_series_noise(panel)
    ml = ml_forecasts.copy()
    if not ml.empty:
        ml['Date'] = pd.to_datetime(ml['Date'])
        ml['department_id'] = ml['department_id'].astype(str)
        ml['language'] = ml['language'].astype(str)

    df = daily_stat.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['department_id'] = df['department_id'].astype(str)
    df['language'] = df['language'].astype(str)

    df = df.merge(ml, on=['Date', 'department_id', 'language'], how='left')
    if 'forecast_daily_lgb' not in df.columns:
        df['forecast_daily_lgb'] = np.nan
    df['forecast_daily_lgb'] = df['forecast_daily_lgb'].fillna(df['forecast_daily_language'])

    df = df.merge(noise, on=['department_id', 'language'], how='left')
    ns = df['noise_smape'].fillna(50.0).clip(0, 200) / 100.0
    w_ml = 0.3 + 0.5 * ns
    w_stat = 1.0 - w_ml
    df['forecast_daily_language_blend'] = w_ml * df['forecast_daily_lgb'] + w_stat * df['forecast_daily_language']
    return df[['Date', 'department_id', 'language',
               'forecast_daily_language', 'forecast_daily_lgb', 'forecast_daily_language_blend']]

# %% [markdown]
# ## 11. Monthly Forecast (dept, language) + CV/capacity_error tables

# %%
# ==================== Monthly per (dept, language) with exogenous ====================

def forecast_per_dept_lang_monthly(incoming_clean: pd.DataFrame, exo_monthly: pd.DataFrame) -> pd.DataFrame:
    out_rows = []

    for (dept, lang), g_daily in incoming_clean.groupby(['department_id', 'language']):
        g_daily = g_daily.sort_values('Date')
        g_daily_clean = clean_outliers_daily(g_daily, method=OUTLIER_METHOD)

        g_m = (g_daily_clean.assign(month=g_daily_clean['Date'].dt.to_period('M'))
               .groupby('month')['ticket_total'].sum())

        if not pd.api.types.is_period_dtype(g_m.index):
            g_m.index = pd.PeriodIndex(g_m.index, freq='M')
        if len(g_m) == 0:
            continue

        ts = winsorize_monthly(g_m, IQR_LO, IQR_HI)
        ex_m = prepare_monthly_exog(exo_monthly, ts.index)

        fc_dict, cv = {}, {}
        mp, fp = fit_prophet_monthly_log(ts, ex_m)
        if fp is not None:
            try:
                fc_dict['Prophet'] = fp(H_MONTHS)
            except Exception:
                pass

        try:
            ma, fa = fit_arima_monthly_log(ts, ex_m)
            fc_dict['ARIMA'] = fa(H_MONTHS)
        except Exception:
            pass

        try:
            mt, ft = fit_tbats_or_ets_monthly_log(ts)
            fc_dict['TBATS/ETS'] = ft(H_MONTHS)
        except Exception:
            pass

        if not fc_dict:
            idx = pd.period_range(ts.index[-1] + 1, periods=H_MONTHS, freq='M')
            val = max(0.0, float(ts.mean()))
            fc_dict['NaiveMean'] = pd.Series([val] * H_MONTHS, index=idx)

        try:
            cv = rolling_cv_monthly_adaptive(ts, ex_m) or {}
        except Exception:
            cv = {}

        noisy = noise_score_daily(g_daily_clean) >= NOISE_SCORE_THRESH
        if ENABLE_WEEKLY_CANDIDATE and noisy:
            wk = forecast_weekly_candidate(g_daily_clean, horizon_months=H_MONTHS)
            if wk is not None and len(wk) == H_MONTHS:
                fc_dict['WEEKLY'] = wk
                # If weekly candidate exists, use average of existing CVs as proxy
                cv['WEEKLY'] = np.mean([v for v in cv.values() if np.isfinite(v)]) if cv else 80.0

        blended, meta = select_or_blend_forecasts(fc_dict, cv_scores=cv, blend=True)
        for per, val in blended.items():
            out_rows.append({
                'department_id': dept, 'language': lang, 'month': per,
                'forecast_monthly': max(0.0, float(val)),
                'cv_prophet_smape': cv.get('Prophet', np.nan),
                'cv_arima_smape': cv.get('ARIMA', np.nan),
                'cv_tbats_ets_smape': cv.get('TBATS/ETS', np.nan),
                'cv_weekly_smape': cv.get('WEEKLY', np.nan),
                'winner_model': meta['winner'],
                'w_prophet': meta['weights'].get('Prophet', np.nan) if 'weights' in meta else np.nan,
                'w_arima': meta['weights'].get('ARIMA', np.nan) if 'weights' in meta else np.nan,
                'w_tbats_ets': meta['weights'].get('TBATS/ETS', np.nan) if 'weights' in meta else np.nan,
                'w_weekly': meta['weights'].get('WEEKLY', np.nan) if 'weights' in meta else np.nan,
            })

    df_out = pd.DataFrame(out_rows)
    if not df_out.empty:
        df_out['department_id'] = df_out['department_id'].astype(str)
        if not pd.api.types.is_period_dtype(df_out['month']):
            df_out['month'] = pd.PeriodIndex(df_out['month'], freq='M')
    return df_out

# ==================== CV table ====================

def build_cv_table(fc_monthly: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if fc_monthly is None or fc_monthly.empty:
        raise ValueError("fc_monthly is empty; cannot build CV table.")
    cols_keep = [
        'department_id', 'language',
        'cv_prophet_smape', 'cv_arima_smape', 'cv_tbats_ets_smape', 'cv_weekly_smape',
        'winner_model', 'w_prophet', 'w_arima', 'w_tbats_ets', 'w_weekly'
    ]
    df = (fc_monthly[cols_keep]
          .drop_duplicates(subset=['department_id', 'language']).copy())
    df = df.rename(columns={
        'cv_prophet_smape': 'sMAPE_Prophet_CV',
        'cv_arima_smape': 'sMAPE_ARIMA_CV',
        'cv_tbats_ets_smape': 'sMAPE_TBATS_ETS_CV',
        'cv_weekly_smape': 'sMAPE_WEEKLY_CV',
        'w_prophet': 'Weight_Prophet',
        'w_arima': 'Weight_ARIMA',
        'w_tbats_ets': 'Weight_TBATS_ETS',
        'w_weekly': 'Weight_WEEKLY'
    })
    df['department_id'] = df['department_id'].astype(str)
    df = apply_mapping(df, mapping)
    ordered_cols = [
        'department_id', 'department_name', 'vertical', 'language',
        'sMAPE_Prophet_CV', 'sMAPE_ARIMA_CV', 'sMAPE_TBATS_ETS_CV', 'sMAPE_WEEKLY_CV',
        'winner_model', 'Weight_Prophet', 'Weight_ARIMA', 'Weight_TBATS_ETS', 'Weight_WEEKLY'
    ]
    df = df[ordered_cols]
    return df.sort_values(['vertical', 'department_id', 'language'])

# ==================== capacity_error-like table ====================

def compute_monthly_accuracy_with_history(monthly: pd.DataFrame,
                                          fc_monthly: pd.DataFrame,
                                          report_start: str) -> pd.DataFrame:
    monthly = monthly.copy()
    monthly['department_id'] = monthly['department_id'].astype(str)
    if not pd.api.types.is_period_dtype(monthly['month']):
        monthly['month'] = pd.PeriodIndex(monthly['month'], freq='M')

    fc = fc_monthly.copy()
    fc['department_id'] = fc['department_id'].astype(str)
    if not pd.api.types.is_period_dtype(fc['month']):
        fc['month'] = pd.PeriodIndex(fc['month'], freq='M')

    start_per = pd.Period(report_start, freq='M')
    last_actual = monthly['month'].max()

    hist = (monthly.loc[monthly['month'] >= start_per, ['department_id', 'language', 'month', 'incoming_monthly']]
            .rename(columns={'incoming_monthly': 'Actual_Volume'}))
    hist['Forecast'] = np.nan

    fut = fc[['department_id', 'language', 'month', 'forecast_monthly',
              'cv_prophet_smape', 'cv_arima_smape', 'cv_tbats_ets_smape', 'cv_weekly_smape',
              'winner_model', 'w_prophet', 'w_arima', 'w_tbats_ets', 'w_weekly']].copy()
    fut = fut.loc[fut['month'] > last_actual]
    fut = fut.rename(columns={'forecast_monthly': 'Forecast'})
    fut['Actual_Volume'] = np.nan

    base = pd.concat([hist, fut], ignore_index=True, sort=False)

    base['Forecast_Accuracy'] = np.where(
        (base['Actual_Volume'].notna()) & (base['Forecast'].notna()) & (base['Actual_Volume'] > 0),
        (1 - (np.abs(base['Forecast'] - base['Actual_Volume']) / base['Actual_Volume'])) * 100.0,
        np.nan
    )
    return base

# %% [markdown]
# ## 12. Daily Capacity plan with ML Blending + write to Excel

# %%
# ==================== Daily capacity plan with ML blending ====================

def build_daily_capacity_plan(incoming: pd.DataFrame, mapping: pd.DataFrame, prod_dept: pd.DataFrame,
                              fc_monthly: pd.DataFrame, exo_daily: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    # 1) Statistical daily forecast (top-down or MA fallback)
    if USE_DAILY_FROM_MONTHLY:
        daily_fc = build_daily_from_monthly(incoming, fc_monthly, horizon_days)
    else:
        # Simple moving-average fallback
        d = incoming.copy().sort_values(['department_id', 'language', 'Date'])
        last_date = d['Date'].max()
        start = last_date + pd.Timedelta(days=1)
        idx_future = pd.date_range(start=start, periods=horizon_days, freq='D')
        rows = []
        for (dept, lang), g in d.groupby(['department_id', 'language']):
            g = g.sort_values('Date')
            base = float(g['ticket_total'].tail(28).mean()) if len(g) >= 28 else float(g['ticket_total'].mean())
            prof = (g.assign(dow=g['Date'].dt.dayofweek).groupby('dow')['ticket_total'].mean())
            prof = (prof / prof.mean()) if prof.notna().sum() >= 3 else pd.Series(1.0, index=range(7))
            vals = [max(0.0, base * float(prof.get(d.dayofweek, 1.0))) for d in idx_future]
            rows.append(pd.DataFrame({'department_id': dept, 'language': lang, 'Date': idx_future, 'forecast_daily': vals}))
        daily_fc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=['department_id', 'language', 'Date', 'forecast_daily']
        )

    # 2) Split by language (dynamic if available)
    daily_fc_lang = split_daily_forecast_by_language_dynamic(daily_fc, incoming)
    daily_fc_lang = apply_mapping(daily_fc_lang, mapping)

    # 3) Global ML per vertical (daily)
    lgb_dict = train_global_lgb_per_vertical(incoming, mapping, exo_daily, h_days=horizon_days)
    ml_fc = pd.concat(list(lgb_dict.values()), ignore_index=True) if lgb_dict else pd.DataFrame(
        columns=['Date', 'department_id', 'language', 'vertical', 'forecast_daily_lgb']
    )

    # 4) Historical panel for noise
    panel_hist = prepare_daily_panel(incoming, mapping)

    # 5) Blend
    blended = blend_daily_predictions(daily_fc_lang, ml_fc, panel_hist)
    blended = apply_mapping(blended, mapping)
    blended = blended.merge(prod_dept, on='department_id', how='left')
    blended['avg_tickets_per_agent_day'] = pd.to_numeric(blended['avg_tickets_per_agent_day'], errors='coerce')
    blended['FTE_per_day'] = np.where(
        blended['avg_tickets_per_agent_day'] > 0,
        blended['forecast_daily_language_blend'] / blended['avg_tickets_per_agent_day'],
        np.nan
    )

    cols = ['Date', 'department_id', 'department_name', 'vertical', 'language',
            'forecast_daily_language', 'forecast_daily_lgb', 'forecast_daily_language_blend', 'FTE_per_day']
    for c in ['department_name', 'vertical']:
        if c not in blended.columns:
            blended[c] = None
    return blended[cols].sort_values(['Date', 'vertical', 'department_id', 'language'])
    

# ==================== MAIN ====================

def main():
    # 1) Load
    incoming = load_incoming(INCOMING_SOURCE_PATH, sheet_name=INCOMING_SHEET)
    mapping = load_dept_map(DEPT_MAP_PATH, DEPT_MAP_SHEET)
    prod = load_productivity(PRODUCTIVITY_PATH)
    incoming = apply_mapping(incoming, mapping)

    # 2) Exogenous (daily + monthly)
    exo_daily, exo_monthly = build_exogenous_calendar(incoming, DAILY_HORIZON_DAYS)

    # 3) Monthly forecasts per (dept, lang)
    fc_monthly = forecast_per_dept_lang_monthly(incoming, exo_monthly)

    # 4) capacity_error-like table
    monthly = build_monthly_series(incoming)
    cap_err = compute_monthly_accuracy_with_history(monthly, fc_monthly, REPORT_START_MONTH)
    cap_err = apply_mapping(cap_err, mapping)
    cap_err = cap_err.merge(prod, on='department_id', how='left')
    cap_err['workdays_in_month'] = [business_days_in_month(m.start_time.year, m.start_time.month) for m in cap_err['month']]
    cap_err['Capacity_FTE_per_day'] = np.where(
        (pd.to_numeric(cap_err['avg_tickets_per_agent_day'], errors='coerce') > 0)
        & (cap_err['workdays_in_month'] > 0)
        & (cap_err['Forecast'].notna()),
        cap_err['Forecast'] / (cap_err['avg_tickets_per_agent_day'] * cap_err['workdays_in_month']),
        np.nan
    )

    # 5) Daily plan (stat + ML blend)
    daily_capacity_plan = build_daily_capacity_plan(incoming, mapping, prod, fc_monthly, exo_daily, DAILY_HORIZON_DAYS)

    # 6) CV table
    cv_table = build_cv_table(fc_monthly, mapping)

    # 7) Write Excel
    for df_out in [cap_err, daily_capacity_plan, cv_table]:
        df_out.replace([np.inf, -np.inf], np.nan, inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as w:
        (cap_err[['vertical', 'department_id', 'department_name', 'language', 'month',
                  'Actual_Volume', 'Forecast', 'Forecast_Accuracy',
                  'Capacity_FTE_per_day',
                  'winner_model', 'cv_prophet_smape', 'cv_arima_smape', 'cv_tbats_ets_smape', 'cv_weekly_smape',
                  'w_prophet', 'w_arima', 'w_tbats_ets', 'w_weekly']]
         .sort_values(['vertical', 'department_id', 'language', 'month'])
         .to_excel(w, "capacity_error", index=False))

        daily_capacity_plan.to_excel(w, "daily_capacity_plan", index=False)
        cv_table.to_excel(w, "mape_table_cv", index=False)

    print("Excel written:", OUTPUT_XLSX)



# Load manually before timing blocks
incoming = load_incoming(INCOMING_SOURCE_PATH, sheet_name=INCOMING_SHEET)
mapping = load_dept_map(DEPT_MAP_PATH, DEPT_MAP_SHEET)
prod = load_productivity(PRODUCTIVITY_PATH)

incoming = apply_mapping(incoming, mapping)


t0 = time.time()
exo_daily, exo_monthly = build_exogenous_calendar(incoming, DAILY_HORIZON_DAYS)
print(f"[TIMING] exogenous: {time.time() - t0:.1f}s"); t1 = time.time()

fc_monthly = forecast_per_dept_lang_monthly(incoming, exo_monthly)
print(f"[TIMING] monthly models: {time.time() - t1:.1f}s"); t2 = time.time()

daily_capacity_plan = build_daily_capacity_plan(incoming, mapping, prod, fc_monthly, exo_daily, DAILY_HORIZON_DAYS)
print(f"[TIMING] daily (stat+LGBM): {time.time() - t2:.1f}s")

# Entry point
if __name__ == "__main__":
    main()

# %%
"fc_monthly" in globals(), "daily_capacity_plan" in globals()


