# -*- coding: utf-8 -*-
"""
corporate_hybrid_forecast_v7_fixed.py

- Rutas portables (CAPACITY_BASE_DIR o carpeta del script)
- Exógenas por idioma con festivos reales (holidays) a nivel (Date, language)
- Fix de memoria: sin arrays dtype=object gigantes
- LGB global por vertical + forecast futuro vectorizado y exógenas futuras en matrices
- Salida Excel: outputs/capacity_forecast_hybrid.xlsx
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DEBUG = os.getenv("CAPACITY_DEBUG", "0") == "1"


def log(msg: str):
    if DEBUG:
        print(msg)


# =========================
# Optional dependencies
# =========================
try:
    import holidays as _holidays_lib
except Exception:
    _holidays_lib = None

try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet  # legacy
    except Exception:
        Prophet = None

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
except Exception:
    lgb = None
    TimeSeriesSplit = None

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = Path(os.getenv("CAPACITY_BASE_DIR", str(PROJECT_ROOT)))
INPUT_DIR = BASE_DIR / "input_model"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INCOMING_SOURCE_PATH = INPUT_DIR / "Incoming_new.xlsx"
INCOMING_SHEET = "Main"

DEPT_MAP_PATH = INPUT_DIR / "department.xlsx"
DEPT_MAP_SHEET = "map"

PRODUCTIVITY_PATH = INPUT_DIR / "productivity_agents.xlsx"

OUTPUT_XLSX = OUTPUT_DIR / "capacity_forecast_hybrid.xlsx"

H_MONTHS = 12
DAILY_HORIZON_DAYS = 90
REPORT_START_MONTH = "2025-01"

# Daily baseline -> split language
LANG_SPLIT_MODE = "from_column"  # "fixed_shares" | "from_column"
LANG_FIXED_SHARES = {"English": 0.6, "Spanish": 0.4}

# Outliers
OUTLIER_STL_PERIOD = 7
OUTLIER_Z_THRESH = 4.0

# LightGBM
LGB_N_ESTIMATORS = 250
LGB_LEARNING_RATE = 0.05
LGB_NUM_LEAVES = 31
LGB_MAX_DEPTH = -1

# Para controlar memoria en training (recomendado)
# 2023-01 a hoy está bien; si te pesa, baja a 540–730 días
LGB_TRAIN_LOOKBACK_DAYS = 900  # set 0 para usar todo

# Mapeo idioma -> país(es) (ajusta a tu realidad)
LANG_TO_COUNTRY = {
    "English": ["UK", "IE"],
    "Spanish": "ES",
    "French": "FR",
    "German": "DE",
    "Italian": "IT",
    "Scandinavian": ["SE", "DK", "NO"],
}

def pbar(iterable, **kwargs):
    # disable barra si no quieres ruido o si CAPACITY_DEBUG=0
    return tqdm(iterable, **kwargs)

# =========================
# Metrics
# =========================
def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def business_days_in_month(year: int, month: int) -> int:
    d0 = pd.Timestamp(year=year, month=month, day=1)
    d1 = d0 + pd.offsets.MonthBegin(1)
    return int(np.busday_count(d0.date(), d1.date()))


# =========================
# Loaders
# =========================
def load_incoming(path: Path, sheet_name: str = "Main") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)

    if "Date" not in df.columns:
        raise ValueError("Incoming must contain 'Date'.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()

    if "department_id" not in df.columns:
        raise ValueError("Incoming must contain 'department_id'.")

    if "ticket_total" not in df.columns:
        for alt in ["total_incoming", "tickets", "ticket", "incoming_tickets_new"]:
            if alt in df.columns:
                df["ticket_total"] = df[alt]
                break

    if "ticket_total" not in df.columns:
        raise ValueError("Incoming must contain 'ticket_total' (or a known alternative).")

    df["ticket_total"] = pd.to_numeric(df["ticket_total"], errors="coerce").fillna(0.0).astype(np.float32)

    if "language" not in df.columns:
        df["language"] = "Unknown"
    df["language"] = df["language"].fillna("Unknown").astype(str)

    return df


def load_dept_map(path: Path, sheet_name: str = "map") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    if "department_id" not in df.columns:
        raise ValueError("department.xlsx must contain 'department_id'.")
    return df


def load_productivity(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "department_id" not in df.columns:
        raise ValueError("productivity file must contain 'department_id'.")

    if "avg_tickets_per_agent_day" not in df.columns:
        for alt in ["tickets_per_agent_day", "avg_prod", "productivity"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "avg_tickets_per_agent_day"})
                break

    if "avg_tickets_per_agent_day" not in df.columns:
        df["avg_tickets_per_agent_day"] = np.nan

    df["avg_tickets_per_agent_day"] = pd.to_numeric(df["avg_tickets_per_agent_day"], errors="coerce").astype(np.float32)
    return df[["department_id", "avg_tickets_per_agent_day"]].copy()


def apply_mapping(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(mapping, on="department_id", how="left", suffixes=("", "_map"))
    if "department_name" not in out.columns and "department_name_map" in out.columns:
        out["department_name"] = out["department_name_map"]
    if "vertical" not in out.columns and "vertical_map" in out.columns:
        out["vertical"] = out["vertical_map"]
    if "vertical" not in out.columns:
        out["vertical"] = "Unknown"
    out["vertical"] = out["vertical"].fillna("Unknown").astype(str)
    return out


# =========================
# Outliers
# =========================
def correct_outliers_stl_daily(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("Date").copy()
    s = g.set_index("Date")["ticket_total"].astype(np.float64)

    if len(s) < 2 * OUTLIER_STL_PERIOD + 7:
        return g

    try:
        stl = STL(s, period=OUTLIER_STL_PERIOD, robust=True)
        res = stl.fit()
        resid = res.resid
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) + 1e-9
        z = (resid - med) / (1.4826 * mad)

        mask = np.abs(z) > OUTLIER_Z_THRESH
        if mask.any():
            corrected = (res.trend + res.seasonal + med).clip(lower=0)
            s2 = s.copy()
            s2[mask] = corrected[mask]
            g["ticket_total"] = s2.values.astype(np.float32)
    except Exception:
        pass

    return g


def correct_outliers(incoming: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (dept, lang), g in incoming.groupby(["department_id", "language"], as_index=False):
        out.append(correct_outliers_stl_daily(g))
    return pd.concat(out, ignore_index=True) if out else incoming


# =========================
# Holidays / Exogenous (FIX)
# =========================
def _holiday_flags_for_language(dates: pd.DatetimeIndex, language: str) -> np.ndarray:
    """
    Returns int8 flags (0/1) for holidays for the country(ies) mapped to language.
    If multiple countries: OR logic.
    """
    if _holidays_lib is None:
        return np.zeros(len(dates), dtype=np.int8)

    countries = LANG_TO_COUNTRY.get(language)
    if countries is None:
        return np.zeros(len(dates), dtype=np.int8)

    if isinstance(countries, str):
        countries = [countries]

    years = sorted({int(d.year) for d in dates})
    cals = []
    for c in countries:
        try:
            cals.append(_holidays_lib.country_holidays(c, years=years))
        except Exception:
            continue

    if not cals:
        return np.zeros(len(dates), dtype=np.int8)

    out = np.zeros(len(dates), dtype=np.int8)
    for i, d in enumerate(dates):
        dd = d.date()
        out[i] = 1 if any(dd in cal for cal in cals) else 0
    return out


def build_exogenous_calendar_v2(incoming: pd.DataFrame, horizon_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inc = incoming.copy()
    inc["Date"] = pd.to_datetime(inc["Date"], errors="coerce")
    inc = inc[inc["Date"].notna()].copy()

    if "language" not in inc.columns:
        inc["language"] = "Unknown"
    inc["language"] = inc["language"].fillna("Unknown").astype(str)

    last_date = inc["Date"].max()
    start = (last_date + pd.Timedelta(days=1)).normalize()
    idx = pd.date_range(start=start, periods=horizon_days, freq="D")

    base = pd.DataFrame({"Date": idx})
    base["dayofweek"] = base["Date"].dt.dayofweek.astype(np.int16)
    base["weekofyear"] = base["Date"].dt.isocalendar().week.astype(np.int16)
    base["month"] = base["Date"].dt.to_period("M").astype(str)

    langs = sorted(inc["language"].unique().tolist()) or ["Unknown"]

    frames = []
    for lang in langs:
        tmp = base.copy()
        tmp["language"] = lang

        if _holidays_lib is None:
            tmp["is_holiday_lang"] = 0
        else:
            tmp["is_holiday_lang"] = _holiday_flags_for_language(pd.DatetimeIndex(tmp["Date"]), lang).astype(np.int8)

        tmp["is_event"] = 0
        frames.append(tmp)

    exo_daily = pd.concat(frames, ignore_index=True)

    # Tipado fuerte para evitar dtype object
    exo_daily["language"] = exo_daily["language"].astype(str)
    exo_daily["is_holiday_lang"] = pd.to_numeric(exo_daily["is_holiday_lang"], errors="coerce").fillna(0).astype(np.float32)
    exo_daily["is_event"] = pd.to_numeric(exo_daily["is_event"], errors="coerce").fillna(0).astype(np.float32)
    exo_daily["dayofweek"] = exo_daily["dayofweek"].astype(np.int16)
    exo_daily["weekofyear"] = exo_daily["weekofyear"].astype(np.int16)

    exo_monthly = (
        exo_daily.groupby(["month", "language"], as_index=False)[["is_holiday_lang", "is_event"]]
        .sum()
        .rename(columns={"is_holiday_lang": "holiday_days_in_month", "is_event": "event_days_in_month"})
    )
    exo_monthly["holiday_days_in_month"] = exo_monthly["holiday_days_in_month"].astype(np.float32)
    exo_monthly["event_days_in_month"] = exo_monthly["event_days_in_month"].astype(np.float32)

    return exo_daily, exo_monthly



# =========================
# Monthly series + models
# =========================
def build_monthly_series(incoming: pd.DataFrame) -> pd.DataFrame:
    d = incoming.copy()
    d["month"] = d["Date"].dt.to_period("M").dt.to_timestamp()
    m = (
        d.groupby(["department_id", "language", "month"], as_index=False)["ticket_total"]
        .sum()
        .rename(columns={"ticket_total": "incoming_monthly"})
    )
    m["incoming_monthly"] = m["incoming_monthly"].astype(np.float32)
    return m


def fit_prophet_monthly_log(ts_m: pd.Series, exo_m: Optional[pd.DataFrame] = None):
    if Prophet is None or len(ts_m) < 18:
        return None, lambda h: np.full(h, np.nan, dtype=np.float32)

    dfp = pd.DataFrame({"ds": ts_m.index, "y": np.log1p(ts_m.values.astype(np.float64))})

    if exo_m is not None and len(exo_m) == len(dfp):
        for c in exo_m.columns:
            dfp[c] = pd.to_numeric(exo_m[c], errors="coerce").fillna(0.0).to_numpy()


    m = Prophet(weekly_seasonality=False, yearly_seasonality=True, daily_seasonality=False)
    if exo_m is not None:
        for c in exo_m.columns:
            m.add_regressor(c)

    m.fit(dfp)

    def predict(h: int):
        future = pd.DataFrame({"ds": pd.date_range(ts_m.index.max() + pd.offsets.MonthBegin(1), periods=h, freq="MS")})
        if exo_m is not None:
            for c in exo_m.columns:
                future[c] = 0.0
        fc = m.predict(future)
        return np.clip(np.expm1(fc["yhat"].values.astype(np.float64)), 0, None).astype(np.float32)

    return m, predict


def fit_arima_monthly_log(ts_m: pd.Series):
    if len(ts_m) < 12:
        return None, lambda h: np.full(h, np.nan, dtype=np.float32)

    y = np.log1p(ts_m.values.astype(np.float64))

    # bounded search
    candidates = [(p, d, q) for p in [0, 1, 2] for d in [0, 1] for q in [0, 1, 2]]
    best_aic = np.inf
    best_res = None

    for order in candidates:
        try:
            mod = SARIMAX(
                y,
                order=order,
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = mod.fit(disp=False)
            if res.aic < best_aic:
                best_aic = res.aic
                best_res = res
        except Exception:
            continue

    if best_res is None:
        return None, lambda h: np.full(h, np.nan, dtype=np.float32)

    def predict(h: int):
        fc = best_res.get_forecast(steps=h).predicted_mean
        return np.clip(np.expm1(np.asarray(fc, dtype=np.float64)), 0, None).astype(np.float32)

    return best_res, predict


def fit_ets_monthly_log(ts_m: pd.Series):
    if len(ts_m) < 6:
        return None, lambda h: np.full(h, np.nan, dtype=np.float32)

    y = np.log1p(ts_m.values.astype(np.float64))
    try:
        mod = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add" if len(ts_m) >= 24 else None,
            seasonal_periods=12 if len(ts_m) >= 24 else None,
        )
        res = mod.fit(optimized=True)

        def predict(h: int):
            return np.clip(np.expm1(np.asarray(res.forecast(h), dtype=np.float64)), 0, None).astype(np.float32)

        return res, predict
    except Exception:
        return None, lambda h: np.full(h, np.nan, dtype=np.float32)


def monthly_cv_smape(ts: pd.Series, exo_m: Optional[pd.DataFrame] = None, splits: int = 3, horizon: int = 2) -> Dict[str, float]:
    ts = ts.dropna()
    if len(ts) < 8:
        return {"Prophet": 200.0, "ARIMA": 200.0, "ETS": 200.0}

    metrics = {"Prophet": [], "ARIMA": [], "ETS": []}
    max_train_end = len(ts) - horizon
    if max_train_end <= 4:
        return {"Prophet": 200.0, "ARIMA": 200.0, "ETS": 200.0}

    split_points = np.unique(np.linspace(4, max_train_end, num=splits, dtype=int))
    for end in split_points:
        train = ts.iloc[:end]
        test = ts.iloc[end:end + horizon]
        if len(test) < horizon:
            continue

        ex_train = exo_m.iloc[:end] if exo_m is not None and len(exo_m) >= end else None

        try:
            _, fp = fit_prophet_monthly_log(train, ex_train)
            metrics["Prophet"].append(smape(test.values, np.nan_to_num(fp(horizon), nan=0.0)))
        except Exception:
            metrics["Prophet"].append(200.0)

        try:
            _, fa = fit_arima_monthly_log(train)
            metrics["ARIMA"].append(smape(test.values, np.nan_to_num(fa(horizon), nan=0.0)))
        except Exception:
            metrics["ARIMA"].append(200.0)

        try:
            _, fe = fit_ets_monthly_log(train)
            metrics["ETS"].append(smape(test.values, np.nan_to_num(fe(horizon), nan=0.0)))
        except Exception:
            metrics["ETS"].append(200.0)

    return {k: float(np.nanmean(v)) if len(v) else 200.0 for k, v in metrics.items()}


def blend_weights_from_cv(cv: Dict[str, float]) -> Dict[str, float]:
    inv = {k: 1.0 / max(1e-6, v) for k, v in cv.items()}
    s = float(sum(inv.values()))
    if s <= 0:
        return {k: 1.0 / len(inv) for k in inv}
    return {k: inv[k] / s for k in inv}


def forecast_one_monthly_series(ts: pd.Series, exo_m: Optional[pd.DataFrame] = None, h_months: int = H_MONTHS):
    cv = monthly_cv_smape(ts, exo_m=exo_m, splits=3, horizon=2)
    w = blend_weights_from_cv(cv)

    _, fp = fit_prophet_monthly_log(ts, exo_m)
    _, fa = fit_arima_monthly_log(ts)
    _, fe = fit_ets_monthly_log(ts)

    fc_p = fp(h_months)
    fc_a = fa(h_months)
    fc_e = fe(h_months)

    blend = np.zeros(h_months, dtype=np.float32)
    blend += np.nan_to_num(fc_p, nan=0.0).astype(np.float32) * np.float32(w.get("Prophet", 0.0))
    blend += np.nan_to_num(fc_a, nan=0.0).astype(np.float32) * np.float32(w.get("ARIMA", 0.0))
    blend += np.nan_to_num(fc_e, nan=0.0).astype(np.float32) * np.float32(w.get("ETS", 0.0))

    winner = min(cv, key=cv.get) if cv else "NA"
    meta = {"winner": winner, "cv": cv, "weights": w}
    return blend, meta


def forecast_per_dept_lang_monthly(incoming: pd.DataFrame, exo_monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = build_monthly_series(incoming)
    out_rows = []

    groups = list(monthly.groupby(["department_id", "language"]))
    for (dept, lang), g in pbar(groups, desc="Monthly forecast (dept+lang)", total=len(groups)):
        ts = g.sort_values("month").set_index("month")["incoming_monthly"].astype(np.float32)

        ex_m = None
        if exo_monthly is not None and len(exo_monthly) > 0:
            ex_m_all = exo_monthly[exo_monthly["language"] == lang].copy()
            if not ex_m_all.empty:
                idx_str = ts.index.to_period("M").astype(str)
                ex_m_all = ex_m_all.set_index("month").reindex(idx_str).fillna(0.0)
                ex_m = ex_m_all[["holiday_days_in_month", "event_days_in_month"]].astype(np.float32)

        fc, meta = forecast_one_monthly_series(ts, exo_m=ex_m, h_months=H_MONTHS)
        future_idx = pd.date_range(ts.index.max() + pd.offsets.MonthBegin(1), periods=H_MONTHS, freq="MS")

        for i, mth in enumerate(future_idx):
            out_rows.append({
                "department_id": dept,
                "language": lang,
                "month": mth,
                "Forecast": float(fc[i]),
                "winner_model": meta.get("winner"),
                "cv_prophet_smape": meta["cv"].get("Prophet", np.nan),
                "cv_arima_smape": meta["cv"].get("ARIMA", np.nan),
                "cv_ets_smape": meta["cv"].get("ETS", np.nan),
                "w_prophet": meta["weights"].get("Prophet", np.nan),
                "w_arima": meta["weights"].get("ARIMA", np.nan),
                "w_ets": meta["weights"].get("ETS", np.nan),
            })

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out["Forecast"] = pd.to_numeric(out["Forecast"], errors="coerce").fillna(0.0).astype(np.float32)
    return out


# =========================
# Monthly accuracy / CV table
# =========================
def compute_monthly_accuracy_with_history(monthly_actual: pd.DataFrame, fc_monthly: pd.DataFrame, start_month: str) -> pd.DataFrame:
    m = monthly_actual.rename(columns={"incoming_monthly": "Actual_Volume"}).copy()
    m = m.merge(fc_monthly, on=["department_id", "language", "month"], how="outer")
    m = m[m["month"] >= pd.to_datetime(start_month)].copy()

    m["Forecast_Accuracy"] = np.where(
        m["Actual_Volume"].notna() & m["Forecast"].notna() & (m["Actual_Volume"] != 0),
        1.0 - (np.abs(m["Forecast"] - m["Actual_Volume"]) / m["Actual_Volume"]),
        np.nan
    )
    return m


def build_cv_table(fc_monthly: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "department_id", "language", "winner_model",
        "cv_prophet_smape", "cv_arima_smape", "cv_ets_smape",
        "w_prophet", "w_arima", "w_ets"
    ]
    t = fc_monthly[cols].drop_duplicates(subset=["department_id", "language"]).copy()
    t = apply_mapping(t, mapping)
    return t


# =========================
# Daily panel + split
# =========================
def prepare_daily_panel(incoming: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    df = apply_mapping(incoming.copy(), mapping)
    df["Date"] = pd.to_datetime(df["Date"])
    df["ticket_total"] = pd.to_numeric(df["ticket_total"], errors="coerce").fillna(0.0).astype(np.float32)

    panel = (
        df.groupby(["Date", "department_id", "language", "vertical"], as_index=False)["ticket_total"]
        .sum()
        .rename(columns={"ticket_total": "y"})
    )
    panel["y"] = panel["y"].astype(np.float32)
    panel["dow"] = panel["Date"].dt.dayofweek.astype(np.int16)
    panel["weekofyear"] = panel["Date"].dt.isocalendar().week.astype(np.int16)
    return panel


def split_daily_forecast_by_language_dynamic(daily_fc: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if LANG_SPLIT_MODE == "fixed_shares":
        rows = []
        for lang, w in LANG_FIXED_SHARES.items():
            tmp = daily_fc.copy()
            tmp["language"] = lang
            tmp["forecast_daily_language"] = (tmp["forecast_daily"].astype(np.float32) * np.float32(w)).astype(np.float32)
            rows.append(tmp[["department_id", "language", "Date", "forecast_daily_language"]])
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["department_id", "language", "Date", "forecast_daily_language"]
        )

    hist = incoming.copy()
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    hist = hist[hist["Date"].notna()].copy()
    hist = hist[hist["Date"] >= (hist["Date"].max() - pd.Timedelta(days=56))].copy()

    if hist.empty:
        out = daily_fc.copy()
        out["forecast_daily_language"] = out["forecast_daily"].astype(np.float32)
        return out[["department_id", "language", "Date", "forecast_daily_language"]]

    lang_shares = (
        hist.groupby(["department_id", "language"], as_index=False)["ticket_total"].sum()
        .rename(columns={"ticket_total": "sum_lang"})
    )
    dept_tot = (
        hist.groupby(["department_id"], as_index=False)["ticket_total"].sum()
        .rename(columns={"ticket_total": "sum_dept"})
    )
    lang_shares = lang_shares.merge(dept_tot, on="department_id", how="left")
    lang_shares["share"] = np.where(lang_shares["sum_dept"] > 0, lang_shares["sum_lang"] / lang_shares["sum_dept"], 0.0).astype(np.float32)

    rows = []
    for dept, g in lang_shares.groupby("department_id"):
        base = daily_fc[daily_fc["department_id"] == dept].copy()
        if base.empty:
            continue
        for _, r in g.iterrows():
            tmp = base.copy()
            tmp["language"] = str(r["language"])
            tmp["forecast_daily_language"] = (tmp["forecast_daily"].astype(np.float32) * np.float32(r["share"])).astype(np.float32)
            rows.append(tmp[["department_id", "language", "Date", "forecast_daily_language"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["department_id", "language", "Date", "forecast_daily_language"]
    )


def blend_daily_predictions(daily_stat: pd.DataFrame, daily_ml: pd.DataFrame, hist_panel: pd.DataFrame) -> pd.DataFrame:
    scores = []
    for (dept, lang), g in hist_panel.groupby(["department_id", "language"]):
        s = g.sort_values("Date").set_index("Date")["y"].astype(np.float64)
        if len(s) < 21:
            continue
        s_last = s.iloc[-28:]
        y_true = s_last.values[7:]
        y_pred = s_last.shift(7).values[7:]
        ns = smape(y_true, y_pred)
        scores.append({"department_id": dept, "language": lang, "noise_smape": ns})

    scores = pd.DataFrame(scores)
    out = daily_stat.merge(daily_ml, on=["Date", "department_id", "language"], how="left")
    out = out.merge(scores, on=["department_id", "language"], how="left")

    if "noise_smape" not in out.columns:
        out["noise_smape"] = 50.0
    out["noise_smape"] = out["noise_smape"].fillna(out["noise_smape"].median() if not out["noise_smape"].isna().all() else 50.0)

    w_ml = np.clip(1.0 - (out["noise_smape"] / 200.0), 0.15, 0.85).astype(np.float32)

    out["forecast_daily_lgb"] = out["forecast_daily_lgb"].fillna(out["forecast_daily_language"])
    out["forecast_daily_language_blend"] = (
        (1.0 - w_ml) * out["forecast_daily_language"].astype(np.float32)
        + w_ml * out["forecast_daily_lgb"].astype(np.float32)
    ).astype(np.float32)

    return out


# =========================
# Global LGB per vertical — MEMORY SAFE
# =========================
def train_global_lgb_per_vertical(incoming: pd.DataFrame, mapping: pd.DataFrame, exo_daily: pd.DataFrame, h_days: int = 90) -> Dict[str, pd.DataFrame]:
    if lgb is None or TimeSeriesSplit is None:
        log("LightGBM not available; skipping ML.")
        return {}

    panel = prepare_daily_panel(incoming, mapping)

    # Merge exo (Date, language) — keep dtypes numeric
    exo = exo_daily.copy()
    exo["Date"] = pd.to_datetime(exo["Date"])
    exo["language"] = exo["language"].astype(str)

    # remove duplicates that can create weird object structures
    exo = exo.drop_duplicates(subset=["Date", "language"], keep="last")

    for c in ["dayofweek", "weekofyear", "is_holiday_lang", "is_event"]:
        if c in exo.columns:
            exo[c] = pd.to_numeric(exo[c], errors="coerce").fillna(0)
    exo["dayofweek"] = exo["dayofweek"].astype(np.int16)
    exo["weekofyear"] = exo["weekofyear"].astype(np.int16)
    exo["is_holiday_lang"] = exo["is_holiday_lang"].astype(np.float32)
    exo["is_event"] = exo["is_event"].astype(np.float32)

    panel["language"] = panel["language"].astype(str)
    panel = panel.merge(exo[["Date", "language", "is_holiday_lang", "is_event"]], on=["Date", "language"], how="left")
    panel["is_holiday_lang"] = panel["is_holiday_lang"].fillna(0).astype(np.float32)
    panel["is_event"] = panel["is_event"].fillna(0).astype(np.float32)

    # lookback (optional)
    if LGB_TRAIN_LOOKBACK_DAYS and LGB_TRAIN_LOOKBACK_DAYS > 0:
        cut = panel["Date"].max() - pd.Timedelta(days=int(LGB_TRAIN_LOOKBACK_DAYS))
        panel = panel[panel["Date"] >= cut].copy()

    feats = ["dow", "weekofyear", "is_holiday_lang", "is_event", "lag_1", "lag_7", "lag_14", "roll_7", "roll_14"]

    res: Dict[str, pd.DataFrame] = {}

    # Precompute future date index
    last_date = panel["Date"].max()
    idx_future = pd.date_range(start=(last_date + pd.Timedelta(days=1)).normalize(), periods=h_days, freq="D")
    dow_vec = np.array([d.dayofweek for d in idx_future], dtype=np.int16)
    woy_vec = np.array([int(pd.Timestamp(d).isocalendar().week) for d in idx_future], dtype=np.int16)

    for vert, g0 in panel.groupby("vertical"):
        g0 = g0.sort_values(["department_id", "language", "Date"]).copy()

        # Build lags/rolls WITHOUT apply (más eficiente)
        key = ["department_id", "language"]
        g0["lag_1"] = g0.groupby(key)["y"].shift(1)
        g0["lag_7"] = g0.groupby(key)["y"].shift(7)
        g0["lag_14"] = g0.groupby(key)["y"].shift(14)

        # Rolling means: transform (es más pesado, pero estable)
        g0["roll_7"] = g0.groupby(key)["y"].transform(lambda s: s.rolling(7).mean())
        g0["roll_14"] = g0.groupby(key)["y"].transform(lambda s: s.rolling(14).mean())

        g = g0.dropna(subset=["lag_1", "lag_7", "roll_7"]).copy()

        # downcast
        for c in ["lag_1", "lag_7", "lag_14", "roll_7", "roll_14"]:
            g[c] = pd.to_numeric(g[c], errors="coerce").astype(np.float32)

        X = g[feats].astype(np.float32)
        y = g["y"].astype(np.float32)

        params = dict(
            n_estimators=LGB_N_ESTIMATORS,
            learning_rate=LGB_LEARNING_RATE,
            num_leaves=LGB_NUM_LEAVES,
            max_depth=LGB_MAX_DEPTH,
            min_data_in_leaf=30,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            random_state=42,
        )

        model = lgb.LGBMRegressor(**params)

        # CV rápido (sanity)
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        for tr, te in tscv.split(X):
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            cv_scores.append(smape(y.iloc[te].values, pred))
        lgb_cv_smape = float(np.mean(cv_scores)) if cv_scores else np.nan

        model.fit(X, y)

        # ===== Future exogenous matrices (Date x Language) — FIX de object arrays =====
        langs = sorted(g0["language"].astype(str).unique().tolist())
        lang_to_idx = {l: j for j, l in enumerate(langs)}

        exo_future = exo[exo["Date"].isin(idx_future)].copy()
        exo_future = exo_future[exo_future["language"].isin(langs)].copy()
        exo_future = exo_future.drop_duplicates(subset=["Date", "language"], keep="last")

        # Build matrices (small: 90 x up to 6)
        hol_mat = (
            exo_future.pivot(index="Date", columns="language", values="is_holiday_lang")
            .reindex(idx_future)
            .reindex(columns=langs)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        evt_mat = (
            exo_future.pivot(index="Date", columns="language", values="is_event")
            .reindex(idx_future)
            .reindex(columns=langs)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        hol_mat = np.nan_to_num(hol_mat, nan=0.0).astype(np.float32)
        evt_mat = np.nan_to_num(evt_mat, nan=0.0).astype(np.float32)
        

        # ===== Initial state per series (KEEP SMALL, no huge object arrays) =====
        # We'll store arrays in Python lists (one per series), not in a giant dataframe column
        series_dept = []
        series_lang = []
        series_lang_idx = []
        tails: list[np.ndarray] = []

        for (dept, lang), hst in g0.groupby(["department_id", "language"]):
            hst = hst.sort_values("Date")
            y_hist = hst["y"].astype(np.float32).values
            if y_hist.size == 0:
                continue
            tail = y_hist[-14:] if y_hist.size >= 14 else y_hist
            tails.append(tail.copy())
            series_dept.append(dept)
            series_lang.append(str(lang))
            series_lang_idx.append(int(lang_to_idx.get(str(lang), 0)))

        if not tails:
            res[vert] = pd.DataFrame(columns=["Date", "department_id", "language", "vertical", "forecast_daily_lgb", "lgb_cv_smape"])
            continue

        series_dept = np.array(series_dept)
        series_lang = np.array(series_lang, dtype=object)
        series_lang_idx = np.array(series_lang_idx, dtype=np.int16)

        n_series = len(tails)
        out_frames = []

        # Helper functions on tails
        def lag(arr: np.ndarray, k: int) -> float:
            return float(arr[-k]) if arr.size >= k else float(arr[-1])

        def roll(arr: np.ndarray, k: int) -> float:
            if arr.size >= k:
                return float(arr[-k:].mean())
            return float(arr[-1])

        for i, d in enumerate(idx_future):
            # Build lag/roll arrays
            lag_1 = np.empty(n_series, dtype=np.float32)
            lag_7 = np.empty(n_series, dtype=np.float32)
            lag_14 = np.empty(n_series, dtype=np.float32)
            roll_7 = np.empty(n_series, dtype=np.float32)
            roll_14 = np.empty(n_series, dtype=np.float32)

            for s in range(n_series):
                a = tails[s]
                lag_1[s] = lag(a, 1)
                lag_7[s] = lag(a, 7)
                lag_14[s] = lag(a, 14)
                roll_7[s] = roll(a, 7)
                roll_14[s] = roll(a, 14)

            # exo per series from matrices
            is_hol = hol_mat[i, series_lang_idx].astype(np.float32)
            is_evt = evt_mat[i, series_lang_idx].astype(np.float32)

            Xf = pd.DataFrame({
                "dow": np.full(n_series, dow_vec[i], dtype=np.int16),
                "weekofyear": np.full(n_series, woy_vec[i], dtype=np.int16),
                "is_holiday_lang": is_hol,
                "is_event": is_evt,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "roll_7": roll_7,
                "roll_14": roll_14,
            })

            Xf = Xf[feats].astype(np.float32)
            preds = model.predict(Xf).astype(np.float32)
            preds = np.clip(preds, 0.0, None)

            tmp = pd.DataFrame({
                "Date": np.full(n_series, d, dtype="datetime64[ns]"),
                "department_id": series_dept,
                "language": series_lang,
                "vertical": vert,
                "forecast_daily_lgb": preds,
                "lgb_cv_smape": np.full(n_series, lgb_cv_smape, dtype=np.float32),
            })
            out_frames.append(tmp)

            # update tails (append pred, keep last 14)
            for s in range(n_series):
                a = tails[s]
                if a.size < 14:
                    tails[s] = np.append(a, preds[s]).astype(np.float32)
                else:
                    # shift left and set last
                    a = a.copy()
                    a[:-1] = a[1:]
                    a[-1] = preds[s]
                    tails[s] = a

        res[vert] = pd.concat(out_frames, ignore_index=True)

    return res


# =========================
# Daily plan builder
# =========================
def build_daily_capacity_plan(
    incoming: pd.DataFrame,
    mapping: pd.DataFrame,
    prod_dept: pd.DataFrame,
    exo_daily: pd.DataFrame,
    horizon_days: int
) -> pd.DataFrame:
    incoming = correct_outliers(incoming)

    last_date = pd.to_datetime(incoming["Date"]).max()
    idx_future = pd.date_range(start=(last_date + pd.Timedelta(days=1)).normalize(), periods=horizon_days, freq="D")

    # Base forecast per dept: DOW profile
    rows = []
    for dept, g in incoming.groupby("department_id"):
        g_daily = g.groupby("Date", as_index=False)["ticket_total"].sum().sort_values("Date")
        if g_daily.empty:
            continue

        prof = (
            g_daily.assign(dow=g_daily["Date"].dt.dayofweek)
            .groupby("dow")["ticket_total"].mean()
        )
        base = float(g_daily["ticket_total"].tail(28).mean()) if len(g_daily) >= 14 else float(g_daily["ticket_total"].mean())

        vals = [max(0.0, base * float(prof.get(d.dayofweek, 1.0))) for d in idx_future]
        rows.append(pd.DataFrame({"department_id": dept, "language": "Unknown", "Date": idx_future, "forecast_daily": np.array(vals, dtype=np.float32)}))

    daily_fc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["department_id", "language", "Date", "forecast_daily"])

    # Split by language
    daily_fc_lang = split_daily_forecast_by_language_dynamic(daily_fc, incoming)
    daily_fc_lang = apply_mapping(daily_fc_lang, mapping)

    # ML forecasts
    lgb_dict = train_global_lgb_per_vertical(incoming, mapping, exo_daily, h_days=horizon_days)
    ml_fc = pd.concat(list(lgb_dict.values()), ignore_index=True) if lgb_dict else pd.DataFrame(
        columns=["Date", "department_id", "language", "vertical", "forecast_daily_lgb"]
    )

    # Blend
    panel_hist = prepare_daily_panel(incoming, mapping)
    blended = blend_daily_predictions(daily_fc_lang, ml_fc, panel_hist)
    blended = apply_mapping(blended, mapping)

    blended = blended.merge(prod_dept, on="department_id", how="left")
    blended["avg_tickets_per_agent_day"] = pd.to_numeric(blended["avg_tickets_per_agent_day"], errors="coerce").astype(np.float32)

    blended["FTE_per_day"] = np.where(
        blended["avg_tickets_per_agent_day"] > 0,
        blended["forecast_daily_language_blend"] / blended["avg_tickets_per_agent_day"],
        np.nan
    ).astype(np.float32)

    cols = [
        "Date", "vertical", "department_id", "department_name", "language",
        "forecast_daily_language", "forecast_daily_lgb", "forecast_daily_language_blend",
        "avg_tickets_per_agent_day", "FTE_per_day"
    ]

    for c in ["department_name", "vertical"]:
        if c not in blended.columns:
            blended[c] = None

    return blended[cols].sort_values(["Date", "vertical", "department_id", "language"])


# =========================
# MAIN
# =========================
def main():
    # Load
    incoming = load_incoming(INCOMING_SOURCE_PATH, sheet_name=INCOMING_SHEET)
    mapping = load_dept_map(DEPT_MAP_PATH, sheet_name=DEPT_MAP_SHEET)
    prod = load_productivity(PRODUCTIVITY_PATH)

    incoming = apply_mapping(incoming, mapping)

    # Exogenous (single call, single source of truth)
    exo_daily, exo_monthly = build_exogenous_calendar_v2(incoming, DAILY_HORIZON_DAYS)
    log(f"exo_daily={exo_daily.shape} exo_monthly={exo_monthly.shape}")

    # Monthly forecasts
    fc_monthly = forecast_per_dept_lang_monthly(incoming, exo_monthly)

    # Monthly accuracy
    monthly_actual = build_monthly_series(incoming)
    cap_err = compute_monthly_accuracy_with_history(monthly_actual, fc_monthly, REPORT_START_MONTH)
    print("cap_err shape:", cap_err.shape)
    cap_err = apply_mapping(cap_err, mapping)
    print("cap_err after mapping:", cap_err.shape)
    cap_err = cap_err.merge(prod, on="department_id", how="left")
    print("cap_err after prod merge:", cap_err.shape)
    
    print("mapping rows:", mapping.shape)
    print("mapping unique department_id:", mapping["department_id"].nunique())
    print("mapping duplicated department_id:", mapping.duplicated(["department_id"]).sum())

    print("prod rows:", prod.shape)
    print("prod unique department_id:", prod["department_id"].nunique())
    print("prod duplicated department_id:", prod.duplicated(["department_id"]).sum())


    # claves típicas (ajusta si tus nombres son distintos)
    keys = [c for c in ["month", "department_id", "language"] if c in cap_err.columns]
    print("keys present:", keys)

    if keys:
        dup = cap_err.duplicated(keys).sum()
        print("duplicated rows on keys:", dup)
        if dup > 0:
            print("top duplicate counts:")
            print(
                cap_err.groupby(keys).size()
                .sort_values(ascending=False)
                .head(20)
            )

    # stop here while debugging
    return


    cap_err["workdays_in_month"] = [
        business_days_in_month(pd.Timestamp(m).year, pd.Timestamp(m).month) if pd.notna(m) else np.nan
        for m in cap_err["month"]
    ]

    cap_err["avg_tickets_per_agent_day"] = pd.to_numeric(
        cap_err["avg_tickets_per_agent_day"], errors="coerce"
    ).astype(np.float32)

    cap_err["Capacity_FTE_per_day"] = np.where(
        (cap_err["avg_tickets_per_agent_day"] > 0)
        & (cap_err["workdays_in_month"] > 0)
        & (cap_err["Forecast"].notna()),
        cap_err["Forecast"].astype(np.float32)
        / (cap_err["avg_tickets_per_agent_day"] * cap_err["workdays_in_month"].astype(np.float32)),
        np.nan,
    ).astype(np.float32)

    print("cap_err shape:", cap_err.shape)

    # Daily plan
    daily_capacity_plan = build_daily_capacity_plan(
        incoming, mapping, prod, exo_daily, DAILY_HORIZON_DAYS
    )

    # --- Export full daily dataset to Parquet (analytical layer) ---
    daily_parquet_path = OUTPUT_DIR / "daily_capacity_plan_full.parquet"
    if daily_capacity_plan is not None and not daily_capacity_plan.empty:
        daily_capacity_plan.to_parquet(
            daily_parquet_path,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )
        print(f"[OK] Parquet exported: {daily_parquet_path} ({len(daily_capacity_plan):,} rows)")
    else:
        print("[WARN] daily_capacity_plan vacío → no Parquet exportado")

    # CV table
    cv_table = build_cv_table(fc_monthly, mapping)

    # Run log (siempre)
    run_log = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(),
        "incoming_rows": int(len(incoming)),
        "fc_monthly_rows": int(len(fc_monthly)) if fc_monthly is not None else -1,
        "daily_capacity_rows": int(len(daily_capacity_plan)) if daily_capacity_plan is not None else -1,
        "cap_err_rows": int(len(cap_err)) if cap_err is not None else -1,
        "cv_table_rows": int(len(cv_table)) if cv_table is not None else -1,
        "parquet_path": str(daily_parquet_path),
    }])

    # Excel: nunca exceder límite de filas
    MAX_EXCEL_ROWS = 1_000_000

    if daily_capacity_plan is not None and len(daily_capacity_plan) > MAX_EXCEL_ROWS:
        daily_capacity_plan_excel = (
            daily_capacity_plan
            .groupby(["Date", "vertical", "department_id"], as_index=False)[
                ["forecast_daily_language_blend", "FTE_per_day"]
            ]
            .sum()
            .sort_values(["Date", "vertical", "department_id"])
        )
        excel_note = f"Daily full > {MAX_EXCEL_ROWS} rows. Excel has aggregated view; full detail in Parquet."
    else:
        daily_capacity_plan_excel = daily_capacity_plan
        excel_note = "Daily included at full granularity (<= Excel row limit)."

    # Write Excel (siempre escribe al menos RUN_LOG)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as w:
        run_log.to_excel(w, "RUN_LOG", index=False)
        pd.DataFrame([{"note": excel_note}]).to_excel(w, "README", index=False)

        if cap_err is not None and not cap_err.empty:
            cap_err.to_excel(w, "capacity_error", index=False)
        else:
            pd.DataFrame([{"msg": "capacity_error vacío"}]).to_excel(w, "capacity_error", index=False)

        if daily_capacity_plan_excel is not None and not daily_capacity_plan_excel.empty:
            daily_capacity_plan_excel.to_excel(w, "daily_capacity_plan", index=False)
        else:
            pd.DataFrame([{"msg": "daily_capacity_plan vacío"}]).to_excel(w, "daily_capacity_plan", index=False)

        if cv_table is not None and not cv_table.empty:
            cv_table.to_excel(w, "mape_table_cv", index=False)
        else:
            pd.DataFrame([{"msg": "mape_table_cv vacío"}]).to_excel(w, "mape_table_cv", index=False)

    print(f"Excel written: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
