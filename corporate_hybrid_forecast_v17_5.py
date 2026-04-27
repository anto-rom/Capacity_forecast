# coding: utf-8
# corporate_hybrid_forecast_v17_5 (updated)
#
# Key updates:
# 1) Add CALL_NOT_INDEXED input (calls_not_indexed.xlsx) with fallback to calls_not_indexed.xlsm
# 2) Add "Actual Volume with calls not indexed" = Actual+CallsNI (history) or Forecast+PredCallsNI (future)
# 3) Ensure Einstein solved reduces "human" workload via explicit "Human cases forecast" metric
#
# Pipeline preserved:
# recommended model mapping → daily forecast → monthly (sum) → Einstein deduction → bias calibration → export

from pathlib import Path
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------
# 1) Configuration
# -----------------------------
BASE_DIR = r'C:\Users\pt3canro\Desktop\CAPACITY'
BASE_DIR = str(Path(BASE_DIR).expanduser().resolve())
INPUT_DIR = str(Path(BASE_DIR) / 'input_model')
OUTPUT_DIR = str(Path(BASE_DIR) / 'outputs')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

INCOMING_SOURCE_PATH = os.path.join(INPUT_DIR, 'Incoming_new.xlsx')
INCOMING_SHEET = 'Main'
DEPT_MAP_PATH = os.path.join(INPUT_DIR, 'department.xlsx')
DEPT_MAP_SHEET = 'map'
AGENT_CAPACITY_PATH = os.path.join(INPUT_DIR, 'agent_language_n_target.xlsx')
EINSTEIN_PATH = os.path.join(INPUT_DIR, 'einstein.xlsx')
INVENTORY_PATH = os.path.join(INPUT_DIR, 'inventory_month.xlsx')
PRODUCTIVITY_AGENTS_PATH = os.path.join(INPUT_DIR, 'productivity_agents.xlsx')

# NEW: Calls not indexed input
CALL_NOT_INDEXED_PATH = os.path.join(INPUT_DIR, 'calls_not_indexed.xlsx')
CALL_NOT_INDEXED_FALLBACK_XLSM = os.path.join(INPUT_DIR, 'calls_not_indexed.xlsm')

REPEATS_PATH = os.path.join(INPUT_DIR, 'repeats.xlsx')
REPEATS_FALLBACK_XLSM = os.path.join(INPUT_DIR, 'repeats.xlsm')

# If True: add repeats to history "actual_volume" (only if you want the training signal uplifted)
APPLY_REPEATS_TO_TRAINING = False  # recommended: keep

OUTPUT_XLSX = os.path.join(OUTPUT_DIR, 'capacity_forecast_v17_5.xlsx')

HORIZON_MONTHS = 12
VERTICALS_TARGET = ['Payments', 'Partners', 'Hospitality']
TARGET_TPH = 6.0
EXCLUDE_DEPARTMENT_NAME_TOKENS = ['PROJ', 'DIST', 'KEY', 'PROXIMIS']

# Optional switch (suggested improvement): uplift training by Calls Not Indexed
# If True, incoming tickets are adjusted upward in history (cases + calls_not_indexed).
APPLY_CALLS_NOT_INDEXED_TO_TRAINING = True

print('OUTPUT_XLSX →', OUTPUT_XLSX)

# -----------------------------
# 2) Helpers
# -----------------------------
def pick_col(df, candidates):
    """Pick the first present column from candidates (case-insensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def std_cols(df):
    df.columns = [c.strip() for c in df.columns]
    return df

def to_month_start(dt_series):
    s = pd.to_datetime(dt_series, errors='coerce')
    return s.dt.to_period('M').dt.to_timestamp(how='start')

def validate_quantiles(dfm):
    viol = dfm[
        (dfm['forecast_p05_dept'] > dfm['forecast_monthly_dept'])
        | (dfm['forecast_monthly_dept'] > dfm['forecast_p95_dept'])
    ]
    if not viol.empty:
        raise ValueError('Quantile order violation in monthly aggregation.')

def safe_int_series(s):
    """Convert to integer-like safely (keeps NaN)."""
    return pd.to_numeric(s, errors='coerce')

# -----------------------------
# 3) Forecast engines (Baseline/STL + SARIMAX-7)
# -----------------------------
def forecast_daily_baseline(y: pd.Series, horizon_days: int = 365):
    """Baseline forecast: STL on log1p with 7-day seasonality; fallback to seasonal naive."""
    y = y.asfreq('D').fillna(0)

    # If too short, seasonal naive
    if len(y) < 56:
        idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq='D')
        last_week = y[-7:].to_numpy()
        p50 = np.resize(last_week, horizon_days)
        resid = (y[7:] - y.shift(7)[7:]).dropna()
        std = resid.std() if len(resid) > 0 else max(1.0, np.sqrt(max(y.mean(), 1)))
        p05 = np.clip(p50 - 1.645 * std, 0, None)
        p95 = p50 + 1.645 * std
        return pd.DataFrame({'date': idx, 'p50': p50, 'p05': p05, 'p95': p95})

    try:
        from statsmodels.tsa.seasonal import STL
        y_box = np.log1p(y)
        stl = STL(y_box, period=7, robust=True)
        res = stl.fit()
        trend, seas, resid = res.trend, res.seasonal, res.resid
        last_trend = trend.iloc[-1]
        std = resid.std() if resid.std() > 0 else 0.5

        idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq='D')
        seas_fut = np.resize(seas[-7:].to_numpy(), horizon_days)
        mu_log = last_trend + seas_fut

        p50 = np.expm1(mu_log); p50 = np.clip(p50, 0, None)
        p05 = np.expm1(mu_log - 1.645 * std); p05 = np.clip(p05, 0, None)
        p95 = np.expm1(mu_log + 1.645 * std)
        return pd.DataFrame({'date': idx, 'p50': p50, 'p05': p05, 'p95': p95})
    except Exception:
        idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq='D')
        last_week = y[-7:].to_numpy()
        p50 = np.resize(last_week, horizon_days)
        resid = (y[7:] - y.shift(7)[7:]).dropna()
        std = resid.std() if len(resid) > 0 else max(1.0, np.sqrt(max(y.mean(), 1)))
        p05 = np.clip(p50 - 1.645 * std, 0, None)
        p95 = p50 + 1.645 * std
        return pd.DataFrame({'date': idx, 'p50': p50, 'p05': p05, 'p95': p95})

def forecast_daily_sarimax(y: pd.Series, horizon_days: int = 365):
    """SARIMAX with weekly seasonality."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        return None

    y = y.asfreq('D').fillna(0)
    y_log = np.log1p(y)

    try:
        model = SARIMAX(
            y_log,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=horizon_days)
        mean = np.expm1(pred.predicted_mean).clip(lower=0)

        resid = (y[7:] - y.shift(7)[7:]).dropna()
        std = resid.std() if len(resid) > 0 else 1.0
        p05 = np.clip(mean - 1.645 * std, 0, None)
        p95 = mean + 1.645 * std

        idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq='D')
        return pd.DataFrame({'date': idx, 'p50': mean.values, 'p05': p05.values, 'p95': p95.values})
    except Exception:
        return None

# -----------------------------
# 4) Load inputs (Incoming + Dept map)
# -----------------------------
incoming_raw = pd.read_excel(INCOMING_SOURCE_PATH, sheet_name=INCOMING_SHEET, engine='openpyxl')
std_cols(incoming_raw)

c_date = pick_col(incoming_raw, ['Date', 'date'])
c_dept = pick_col(incoming_raw, ['department_id', 'dept_id', 'Department_ID'])
c_cnt  = pick_col(incoming_raw, ['ticket_total', 'Ticket_Total', 'count', 'tickets', 'qty'])

if c_date is None or c_dept is None:
    raise KeyError('Incoming_new.xlsx must include Date and department_id columns.')

incoming = incoming_raw[[c_date, c_dept] + ([c_cnt] if c_cnt else [])].copy()
incoming.columns = ['date', 'department_id'] + (['ticket_total'] if c_cnt else [])

if 'ticket_total' not in incoming.columns:
    incoming['ticket_total'] = 1

incoming['date'] = pd.to_datetime(incoming['date'], errors='coerce')
incoming['month'] = to_month_start(incoming['date'])
incoming['department_id'] = pd.to_numeric(incoming['department_id'], errors='coerce').astype('Int64')

dept_map = pd.read_excel(DEPT_MAP_PATH, sheet_name=DEPT_MAP_SHEET, engine='openpyxl')
std_cols(dept_map)

dm_id   = pick_col(dept_map, ['department_id', 'dept_id', 'Department_ID'])
dm_name = pick_col(dept_map, ['department_name', 'Department', 'dept_name', 'Department_Name'])
dm_vert = pick_col(dept_map, ['vertical', 'Vertical'])

if dm_id is None or dm_name is None or dm_vert is None:
    raise KeyError('department.xlsx must contain department_id, department_name and vertical (sheet map).')

dept_map = dept_map[[dm_id, dm_name, dm_vert]].drop_duplicates()
dept_map.columns = ['department_id', 'department_name', 'vertical']
dept_map['department_id'] = pd.to_numeric(dept_map['department_id'], errors='coerce').astype('Int64')

incoming = incoming.merge(dept_map, on='department_id', how='left')

# Scope & exclusions
incoming = incoming[incoming['vertical'].isin(VERTICALS_TARGET)].copy()
mask_excl = pd.Series(False, index=incoming.index)
for tok in EXCLUDE_DEPARTMENT_NAME_TOKENS:
    mask_excl = mask_excl | incoming['department_name'].astype(str).str.upper().str.contains(tok.upper(), na=False)
incoming = incoming.loc[~mask_excl].copy()

# -----------------------------
# 4b) Load Calls Not Indexed and build monthly series
# -----------------------------
calls_ni_month = pd.DataFrame(columns=['department_id', 'month', 'calls_not_indexed'])

cni_path = None
if Path(CALL_NOT_INDEXED_PATH).exists():
    cni_path = CALL_NOT_INDEXED_PATH
elif Path(CALL_NOT_INDEXED_FALLBACK_XLSM).exists():
    cni_path = CALL_NOT_INDEXED_FALLBACK_XLSM

if cni_path is not None:
    cni = pd.read_excel(cni_path, engine='openpyxl')
    std_cols(cni)

    # Typical exports have Created Date + User Team; sometimes also department_id or a pre-aggregated volume column.
    cn_date = pick_col(cni, ['Created Date', 'CreatedDate', 'Date', 'date'])
    cn_dept = pick_col(cni, ['department_id', 'dept_id', 'Department_ID'])
    cn_team = pick_col(cni, ['User Team', 'Team', 'department_name', 'Department', 'Department_name'])
    cn_qty  = pick_col(cni, ['calls_not_indexed', 'call_not_indexed', 'Volume', 'Qty', 'count', 'tickets'])

    tmp = cni.copy()

    if cn_date is None:
        raise KeyError('calls_not_indexed file must contain a date column (e.g., Created Date / Date).')

    tmp['date'] = pd.to_datetime(tmp[cn_date], errors='coerce')
    tmp['month'] = to_month_start(tmp['date'])

    # Department mapping: prefer department_id, else map using team/name
    if cn_dept is not None:
        tmp['department_id'] = pd.to_numeric(tmp[cn_dept], errors='coerce').astype('Int64')
    elif cn_team is not None:
        tmp['department_name'] = tmp[cn_team].astype(str).str.strip()
        tmp = tmp.merge(dept_map[['department_id', 'department_name']], on='department_name', how='left')
    else:
        raise KeyError('calls_not_indexed file must contain department_id or a team/name column (e.g., User Team).')

    # Quantity: if a numeric column exists use it; otherwise count rows as calls
    if cn_qty is not None:
        tmp['calls_not_indexed'] = pd.to_numeric(tmp[cn_qty], errors='coerce').fillna(0)
        agg = tmp.groupby(['department_id', 'month'], as_index=False).agg(calls_not_indexed=('calls_not_indexed', 'sum'))
    else:
        agg = tmp.dropna(subset=['department_id', 'month']).groupby(['department_id', 'month'], as_index=False).size()
        agg = agg.rename(columns={'size': 'calls_not_indexed'})

    calls_ni_month = agg.copy()
# -----------------------------
# 4c) Load Repeats and build monthly series
# -----------------------------

repeats_month = pd.DataFrame(columns=['department_id', 'month', 'repeats_workload'])

rpt_path = None
if Path(REPEATS_PATH).exists():
    rpt_path = REPEATS_PATH
elif Path(REPEATS_FALLBACK_XLSM).exists():
    rpt_path = REPEATS_FALLBACK_XLSM

if rpt_path is not None:
    rr = pd.read_excel(rpt_path, engine='openpyxl')
    std_cols(rr)

    rr_dept = pick_col(rr, ['department_id', 'dept_id', 'Department_ID', 'department'])
    rr_date = pick_col(rr, ['Date', 'date', 'Month', 'month', 'MonthDate'])
    rr_val  = pick_col(rr, [
        'repeat_touchpoints', 'repeats', 'repeat_items', 'RR', 'rr_items', 'total_touchpoints_7d'
    ])

    if rr_dept is not None and rr_date is not None and rr_val is not None:
        tmp = rr[[rr_dept, rr_date, rr_val]].copy()
        tmp.columns = ['department_id', 'date', 'repeats_workload']
        tmp['department_id'] = pd.to_numeric(tmp['department_id'], errors='coerce').astype('Int64')
        tmp['date'] = pd.to_datetime(tmp['date'], errors='coerce')
        tmp['month'] = to_month_start(tmp['date'])
        tmp['repeats_workload'] = pd.to_numeric(tmp['repeats_workload'], errors='coerce').fillna(0)

        repeats_month = (
            tmp.groupby(['department_id', 'month'], as_index=False)
               .agg(repeats_workload=('repeats_workload', 'sum'))
        )

# -----------------------------
# 4d) Monthly actuals (optionally uplift by Calls Not Indexed)
# -----------------------------
monthly_actuals = (
    incoming
    .groupby(['vertical', 'department_id', 'department_name', 'month'], as_index=False)
    .agg(actual_volume=('ticket_total', 'sum'))
)

# If enabled, uplift training signal (cases + calls not indexed)
if APPLY_CALLS_NOT_INDEXED_TO_TRAINING and not calls_ni_month.empty:
    monthly_actuals = monthly_actuals.merge(calls_ni_month, on=['department_id', 'month'], how='left')
    monthly_actuals['calls_not_indexed'] = pd.to_numeric(monthly_actuals['calls_not_indexed'], errors='coerce').fillna(0)
    monthly_actuals['actual_volume'] = monthly_actuals['actual_volume'] + monthly_actuals['calls_not_indexed']
    monthly_actuals = monthly_actuals.drop(columns=['calls_not_indexed'])

print('Loaded incoming rows =', len(incoming))

# -----------------------------
# 5) Recommended model mapping
# -----------------------------
recommended = {}  # department_id -> engine label

# Option 1: read from previous remediation report
try:
    prev_path = os.path.join(OUTPUT_DIR, 'capacity_forecast_v17.xlsx')
    if Path(prev_path).exists():
        prev = pd.read_excel(prev_path, sheet_name='Remediation_Report', engine='openpyxl')
        if {'department_id', 'Recommended_Model'}.issubset(prev.columns):
            for _, r in prev.iterrows():
                if pd.notna(r['department_id']) and pd.notna(r['Recommended_Model']):
                    recommended[int(r['department_id'])] = str(r['Recommended_Model'])
except Exception:
    pass

# Option 2: read manual CSV (department_id,model)
try:
    rm_csv = os.path.join(INPUT_DIR, 'recommended_models.csv')
    if Path(rm_csv).exists():
        dfm = pd.read_csv(rm_csv)
        if {'department_id', 'model'}.issubset(dfm.columns):
            for _, r in dfm.iterrows():
                recommended[int(r['department_id'])] = str(r['model'])
except Exception:
    pass

print('Recommended models loaded for', len(recommended), 'departments (fallback Baseline for others).')

# -----------------------------
# 6) Daily forecast using per-department engine
# -----------------------------
incoming_daily = (
    incoming.groupby(['department_id', 'date'], as_index=False)
    .agg(tickets=('ticket_total', 'sum'))
)

engine_funcs = {
    'Baseline(STL)': forecast_daily_baseline,
    'STL': forecast_daily_baseline,
    'SARIMAX-7': forecast_daily_sarimax
}

daily_forecasts = []
HORIZON_DAYS = 365

for dpt_id, grp in incoming_daily.groupby('department_id'):
    y = grp.set_index('date')['tickets'].sort_index().asfreq('D').fillna(0)
    engine = recommended.get(int(dpt_id), 'Baseline(STL)')
    func = engine_funcs.get(engine, forecast_daily_baseline)
    try:
        fc = func(y, horizon_days=HORIZON_DAYS)
        if fc is None:
            raise RuntimeError('Engine returned None')
    except Exception:
        fc = forecast_daily_baseline(y, horizon_days=HORIZON_DAYS)
        engine = 'Baseline(STL)'

    fc.insert(0, 'department_id', dpt_id)
    fc['engine_used'] = engine
    daily_forecasts.append(fc)

fc_daily_built = pd.concat(daily_forecasts, ignore_index=True) if daily_forecasts else pd.DataFrame()
fc_daily_built = fc_daily_built.merge(dept_map, on='department_id', how='left')
fc_daily_built = fc_daily_built[fc_daily_built['vertical'].isin(VERTICALS_TARGET)]

for tok in EXCLUDE_DEPARTMENT_NAME_TOKENS:
    fc_daily_built = fc_daily_built[
        ~fc_daily_built['department_name'].astype(str).str.upper().str.contains(tok.upper(), na=False)
    ]

fc_daily_built.rename(
    columns={'p50': 'forecast_daily_dept', 'p05': 'p05_daily_dept', 'p95': 'p95_daily_dept'},
    inplace=True
)

print('Daily forecast rows:', len(fc_daily_built))

# -----------------------------
# 7) Monthly aggregation
# -----------------------------
monthly_fc_raw = (
    fc_daily_built.assign(month=to_month_start(fc_daily_built['date']))
    .groupby(['vertical', 'department_id', 'department_name', 'month'], as_index=False)
    .agg(
        forecast_monthly_dept=('forecast_daily_dept', 'sum'),
        forecast_p05_dept=('p05_daily_dept', 'sum'),
        forecast_p95_dept=('p95_daily_dept', 'sum')
    )
)

validate_quantiles(monthly_fc_raw)

# -----------------------------
# 8) Einstein: build monthly + recent rate per department
# -----------------------------
ein_month = pd.DataFrame(columns=['department_id', 'month', 'einstein_solved'])

if Path(EINSTEIN_PATH).exists():
    ein = pd.read_excel(EINSTEIN_PATH, engine='openpyxl')
    std_cols(ein)

    nr_col = pick_col(ein, ['Not_Reopened', 'not_reopened', 'Not Reopened', 'NotReopened'])
    e_year = pick_col(ein, ['Week_Year', 'Year', 'year'])
    e_mnum = pick_col(ein, ['Month_number', 'month_number', 'Month', 'month'])
    e_dept_id = pick_col(ein, ['department_id', 'dept_id', 'Department_ID'])
    e_dept_name = pick_col(ein, ['department', 'department_name', 'Department', 'Department_name'])
    e_date = pick_col(ein, ['Date', 'date'])

    if nr_col is not None and e_year is not None and e_mnum is not None and (e_dept_id is not None or e_dept_name is not None):
        use_cols = [e_year, e_mnum, nr_col] + ([e_dept_id] if e_dept_id is not None else [e_dept_name])
        ein_use = ein[use_cols].copy()
        ein_use = ein_use.rename(columns={e_year: 'year', e_mnum: 'month_number', nr_col: 'einstein_solved'})
        if e_dept_id is not None:
            ein_use = ein_use.rename(columns={e_dept_id: 'department_id'})
        else:
            ein_use = ein_use.rename(columns={e_dept_name: 'department_name'})

        ein_use['year'] = safe_int_series(ein_use['year'])
        ein_use['month_number'] = safe_int_series(ein_use['month_number'])
        ein_use['einstein_solved'] = pd.to_numeric(ein_use['einstein_solved'], errors='coerce').fillna(0)

        ein_use['month'] = pd.to_datetime(
            ein_use['year'].astype('Int64').astype(str) + '-' + ein_use['month_number'].astype('Int64').astype(str) + '-01',
            errors='coerce'
        ).dt.to_period('M').dt.to_timestamp(how='start')

        if 'department_id' not in ein_use.columns:
            ein_use['department_name'] = ein_use['department_name'].astype(str).str.strip()
            ein_use = ein_use.merge(dept_map[['department_id', 'department_name']], on='department_name', how='left')

        ein_month = (
            ein_use.dropna(subset=['department_id', 'month'])
            .groupby(['department_id', 'month'], as_index=False)
            .agg(einstein_solved=('einstein_solved', 'sum'))
        )

    elif e_date is not None and e_dept_id is not None:
        # Legacy fallback: count rows as solved events
        ein_use = ein[[e_date, e_dept_id]].copy()
        ein_use.columns = ['date', 'department_id']
        ein_use['date'] = pd.to_datetime(ein_use['date'], errors='coerce')
        ein_use['month'] = to_month_start(ein_use['date'])
        ein_use['department_id'] = pd.to_numeric(ein_use['department_id'], errors='coerce').astype('Int64')
        ein_month = (
            ein_use.dropna(subset=['department_id', 'month'])
            .groupby(['department_id', 'month'], as_index=False)
            .size().rename(columns={'size': 'einstein_solved'})
        )

# Recent Einstein rate based on history (last 3 available months)
recent = pd.DataFrame(columns=['department_id', 'einstein_rate_recent'])

if not ein_month.empty:
    hist_rates = monthly_actuals.merge(ein_month, on=['department_id', 'month'], how='left')
    hist_rates['einstein_solved'] = pd.to_numeric(hist_rates['einstein_solved'], errors='coerce').fillna(0)
    hist_rates['actual_volume'] = pd.to_numeric(hist_rates['actual_volume'], errors='coerce').fillna(0)

    # Rate relative to actual indexed cases (guard division by zero)
    hist_rates['einstein_rate'] = np.where(
        hist_rates['actual_volume'] > 0,
        hist_rates['einstein_solved'] / hist_rates['actual_volume'],
        np.nan
    )

    hist_rates = hist_rates.sort_values('month')
    recent = (
        hist_rates.groupby('department_id', as_index=False)
        .apply(lambda g: pd.Series({'einstein_rate_recent': np.nanmean(g.tail(3)['einstein_rate'])}))
        .reset_index(drop=True)
    )

recent['einstein_rate_recent'] = pd.to_numeric(recent['einstein_rate_recent'], errors='coerce').fillna(0).clip(0, 0.9)

# Apply Einstein deduction
monthly_adj = monthly_fc_raw.merge(recent, on='department_id', how='left')
monthly_adj['einstein_rate_recent'] = pd.to_numeric(monthly_adj['einstein_rate_recent'], errors='coerce').fillna(0).clip(0, 0.9)

for c in ['forecast_monthly_dept', 'forecast_p05_dept', 'forecast_p95_dept']:
    monthly_adj[c + '_post_einstein'] = monthly_adj[c] * (1 - monthly_adj['einstein_rate_recent'])

print('Monthly + Einstein done → rows:', len(monthly_adj))

# -----------------------------
# 9) Calls Not Indexed: compute recent ratio and forecast additional workload
# -----------------------------
calls_recent = pd.DataFrame(columns=['department_id', 'calls_not_indexed_rate_recent'])

if not calls_ni_month.empty:
    # Merge with monthly actuals to estimate ratio calls_not_indexed / actual_volume
    hist_calls = monthly_actuals.merge(calls_ni_month, on=['department_id', 'month'], how='left')
    hist_calls['calls_not_indexed'] = pd.to_numeric(hist_calls['calls_not_indexed'], errors='coerce').fillna(0)
    hist_calls['actual_volume'] = pd.to_numeric(hist_calls['actual_volume'], errors='coerce').fillna(0)

    hist_calls['calls_not_indexed_rate'] = np.where(
        hist_calls['actual_volume'] > 0,
        hist_calls['calls_not_indexed'] / hist_calls['actual_volume'],
        np.nan
    )

    hist_calls = hist_calls[
        (hist_calls['calls_not_indexed_rate'].notna()) &
        (hist_calls['calls_not_indexed_rate'] < 3.0)
    ]

    calls_debug = hist_calls[
    ['department_id', 'month', 'actual_volume', 'calls_not_indexed', 'calls_not_indexed_rate']
    ].copy()

    calls_recent = (
        hist_calls.groupby('department_id', as_index=False)
        .apply(lambda g: pd.Series({'calls_not_indexed_rate_recent': np.nanmean(g.tail(3)['calls_not_indexed_rate'])}))
        .reset_index(drop=True)
    )

calls_recent['calls_not_indexed_rate_recent'] = pd.to_numeric(
    calls_recent['calls_not_indexed_rate_recent'], errors='coerce'
).fillna(0).clip(0, 5.0)  # allow >100% vs cases (e.g., 60% not indexed => ratio 1.5 vs indexed cases)

monthly_adj = monthly_adj.merge(calls_recent, on='department_id', how='left')
monthly_adj['calls_not_indexed_rate_recent'] = pd.to_numeric(
    monthly_adj['calls_not_indexed_rate_recent'], errors='coerce'
).fillna(0).clip(0, 5.0)

# Forecast additional calls not indexed (future) based on base cases forecast
monthly_adj['calls_not_indexed_forecast'] = monthly_adj['forecast_monthly_dept'] * monthly_adj['calls_not_indexed_rate_recent']

# -----------------------------
# 9b) Repeats: compute recent ratio and forecast additional workload
# -----------------------------

repeats_recent = pd.DataFrame(columns=['department_id', 'repeats_rate_recent'])

if not repeats_month.empty:
    hist_rr = monthly_actuals.merge(repeats_month, on=['department_id', 'month'], how='left')
    hist_rr['repeats_workload'] = pd.to_numeric(hist_rr['repeats_workload'], errors='coerce').fillna(0)
    hist_rr['actual_volume'] = pd.to_numeric(hist_rr['actual_volume'], errors='coerce').fillna(0)

    # Avoid division by zero, cap to keep sanity
    hist_rr['repeats_rate'] = np.where(
        hist_rr['actual_volume'] > 0,
        hist_rr['repeats_workload'] / hist_rr['actual_volume'],
        0.0
    )

    # Recent rate: last 3 available months per department (same idea as Einstein/CNI)
    hist_rr = hist_rr.sort_values(['department_id', 'month'])
    repeats_recent = (
        hist_rr.groupby('department_id', as_index=False)
               .tail(3)
               .groupby('department_id', as_index=False)
               .agg(repeats_rate_recent=('repeats_rate', 'mean'))
    )
    repeats_recent['repeats_rate_recent'] = pd.to_numeric(
        repeats_recent['repeats_rate_recent'], errors='coerce'
    ).fillna(0).clip(0, 5.0)

# Attach to monthly_adj and forecast repeats using base forecast
monthly_adj = monthly_adj.merge(repeats_recent, on='department_id', how='left')
monthly_adj['repeats_rate_recent'] = pd.to_numeric(
    monthly_adj.get('repeats_rate_recent', 0), errors='coerce'
).fillna(0).clip(0, 5.0)

monthly_adj['repeats_forecast'] = monthly_adj['forecast_monthly_dept'] * monthly_adj['repeats_rate_recent']

# -----------------------------
# 10) Bias-based calibration (same approach as before)
# -----------------------------
model_used_error_df = pd.DataFrame([
    {'vertical': 'Hospitality', 'department_id': 7,  'department_name': 'CS_PMSH_L1',                 'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 29.35, 'wape_pct': 26.00, 'bias_pct': -0.40},
    {'vertical': 'Hospitality', 'department_id': 8,  'department_name': 'CS_PMSP_CLOUD_L1',           'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 43.80, 'wape_pct': 39.12, 'bias_pct': 12.64},
    {'vertical': 'Hospitality', 'department_id': 11, 'department_name': 'CS_PMSP_CLOUD_L2',           'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 27.76, 'wape_pct': 22.27, 'bias_pct': -4.06},
    {'vertical': 'Hospitality', 'department_id': 23, 'department_name': 'CS_PMSP_FRANCE',             'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 31.44, 'wape_pct': 25.79, 'bias_pct': -1.79},
    {'vertical': 'Hospitality', 'department_id': 5,  'department_name': 'CS_PMSP_INTEG',              'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 20.88, 'wape_pct': 17.68, 'bias_pct': -2.70},
    {'vertical': 'Hospitality', 'department_id': 9,  'department_name': 'CS_PMSP_PREM_L1',            'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 17.29, 'wape_pct': 14.44, 'bias_pct': -3.07},
    {'vertical': 'Hospitality', 'department_id': 10, 'department_name': 'CS_PMSP_PREM_L2',            'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 28.77, 'wape_pct': 23.23, 'bias_pct': -1.12},
    {'vertical': 'Partners',    'department_id': 12, 'department_name': 'CS_PART_APAC',               'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 20.88, 'wape_pct': 20.99, 'bias_pct': -5.14},
    {'vertical': 'Partners',    'department_id': 13, 'department_name': 'CS_PART_EMEA',               'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 27.76, 'wape_pct': 25.91, 'bias_pct': 7.36},
    {'vertical': 'Partners',    'department_id': 14, 'department_name': 'CS_PART_LATAM',              'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 26.60, 'wape_pct': 21.38, 'bias_pct': 5.29},
    {'vertical': 'Partners',    'department_id': 15, 'department_name': 'CS_PART_US',                 'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 40.03, 'wape_pct': 30.97, 'bias_pct': -9.47},
    {'vertical': 'Payments',    'department_id': 3,  'department_name': 'CA_PYAC',                    'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 18.63, 'wape_pct': 18.70, 'bias_pct': 1.28},
    {'vertical': 'Payments',    'department_id': 1,  'department_name': 'CS_GT3C_EU',                 'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 24.80, 'wape_pct': 22.19, 'bias_pct': 0.58},
    {'vertical': 'Payments',    'department_id': 18, 'department_name': 'Datatrans L2 Customer Support','model_used': 'STL','backtest_months': 12,'mape_pct': 54.89, 'wape_pct': 38.55, 'bias_pct': -7.28},
    {'vertical': 'Payments',    'department_id': 2,  'department_name': 'L2 Customer Support',        'model_used': 'STL', 'backtest_months': 12, 'mape_pct': 28.56, 'wape_pct': 24.46, 'bias_pct': -1.36},
    {'vertical': 'Payments',    'department_id': 21, 'department_name': 'Specialist - L2 Customer Support','model_used': 'STL','backtest_months': 12,'mape_pct': 42.25, 'wape_pct': 41.25, 'bias_pct': 17.26},
])

calib_from_bias = model_used_error_df[['department_id', 'bias_pct']].copy()
calib_from_bias['department_id'] = pd.to_numeric(calib_from_bias['department_id'], errors='coerce')
calib_from_bias['calib_factor'] = (1 - calib_from_bias['bias_pct'] / 100.0).clip(0.70, 1.30)

monthly_adj = monthly_adj.merge(calib_from_bias[['department_id', 'calib_factor']], on='department_id', how='left')
monthly_adj['calib_factor'] = monthly_adj['calib_factor'].fillna(1.0)

for c in ['forecast_monthly_dept_post_einstein', 'forecast_p05_dept_post_einstein', 'forecast_p95_dept_post_einstein']:
    monthly_adj[c + '_cal'] = monthly_adj[c] * monthly_adj['calib_factor']

print('Calibration applied. Share of rows with calib_factor != 1:', (monthly_adj['calib_factor'] != 1.0).mean().round(2))

# -----------------------------
# 11) Capacity & Productivity from productivity_agents.xlsx (kept as-is skeleton)
# -----------------------------
cap_prod_month = pd.DataFrame(columns=['department_id', 'month', 'capacity_agents', 'productivity_agents'])

if Path(PRODUCTIVITY_AGENTS_PATH).exists():
    pa = pd.read_excel(PRODUCTIVITY_AGENTS_PATH, engine='openpyxl')
    std_cols(pa)
    # NOTE: Keep your existing logic here to build cap_prod_month
else:
    print('WARNING: productivity_agents.xlsx not found → capacity/productivity from agents will be empty.')

print('Agents-based capacity/productivity rows:', len(cap_prod_month))

# -----------------------------
# 12) Build long_dept and final export frame
# -----------------------------
fc_board = monthly_adj.copy()

# Forecast (Cases) - total cases demand
fc_board['forecast_base'] = pd.to_numeric(fc_board['forecast_monthly_dept'], errors='coerce')

# Forecast after Einstein (Human cases) - calibrated
fc_board['forecast_human_cases_cal'] = pd.to_numeric(fc_board['forecast_monthly_dept_post_einstein_cal'], errors='coerce')

# Einstein solved forecast (keep decimals)
fc_board['einstein_rate_recent'] = pd.to_numeric(fc_board['einstein_rate_recent'], errors='coerce').fillna(0).clip(0, 0.9)
fc_board['einstein_solved_forecast'] = fc_board['forecast_base'] * fc_board['einstein_rate_recent']

# Calls not indexed forecast (future)
fc_board['calls_not_indexed_forecast'] = pd.to_numeric(fc_board['calls_not_indexed_forecast'], errors='coerce').fillna(0)

# Attach actuals
long_dept = (
    fc_board[[
        'vertical', 'department_id', 'department_name', 'month',
        'forecast_base',
        'forecast_human_cases_cal',
        'einstein_rate_recent', 'einstein_solved_forecast',
        'calls_not_indexed_rate_recent', 'calls_not_indexed_forecast'
    ]]
    .merge(
        monthly_actuals[['vertical', 'department_id', 'department_name', 'month', 'actual_volume']],
        on=['vertical', 'department_id', 'department_name', 'month'],
        how='left'
    )
)

# Attach calls not indexed actual (history)
if not calls_ni_month.empty:
    long_dept = long_dept.merge(
        calls_ni_month.rename(columns={'calls_not_indexed': 'calls_not_indexed_actual'}),
        on=['department_id', 'month'],
        how='left'
    )
else:
    long_dept['calls_not_indexed_actual'] = np.nan

# Attach capacity/productivity (agents)
cap_prod_month['month'] = pd.to_datetime(cap_prod_month['month'], errors='coerce').dt.to_period('M').dt.to_timestamp(how='start')
long_dept = long_dept.merge(cap_prod_month, on=['department_id', 'month'], how='left')

# Ensure numeric
for c in [
    'forecast_base', 'forecast_human_cases_cal',
    'einstein_rate_recent', 'einstein_solved_forecast',
    'actual_volume', 'calls_not_indexed_actual',
    'calls_not_indexed_forecast'
]:
    if c in long_dept.columns:
        long_dept[c] = pd.to_numeric(long_dept[c], errors='coerce')
# -----------------------------
# 12b) Add past months (testimony) to long_dept
# -----------------------------

# Keys that identify a unique dept-month
KEYS = ['vertical', 'department_id', 'department_name', 'month']

# ------------------------------------------
# FILTER: keep historical actuals only for current year
# ------------------------------------------
current_year = datetime.now().year

monthly_actuals['month'] = pd.to_datetime(monthly_actuals['month'], errors='coerce')

monthly_actuals = monthly_actuals[
    monthly_actuals['month'].dt.year == current_year
]

hist_keys = monthly_actuals[KEYS].drop_duplicates()
fc_keys = long_dept[KEYS].drop_duplicates()

# Find dept-months that exist in history but not in forecast frame
hist_only_keys = (
    hist_keys.merge(fc_keys, on=KEYS, how='left', indicator=True)
             .query("_merge == 'left_only'")
             .drop(columns=['_merge'])
)

if not hist_only_keys.empty:
    hist_only = hist_only_keys.merge(
        monthly_actuals[KEYS + ['actual_volume']],
        on=KEYS,
        how='left'
    )

    # Add the forecast-side columns as empty, so schema matches long_dept
    for c in [
        'forecast_base',
        'forecast_human_cases_cal',
        'einstein_rate_recent', 'einstein_solved_forecast',
        'calls_not_indexed_rate_recent', 'calls_not_indexed_forecast',
        'calls_not_indexed_actual',
        'capacity_agents', 'productivity_agents'
    ]:
        if c not in hist_only.columns:
            hist_only[c] = np.nan

    # Concatenate: past months + existing long_dept (future)
    long_dept = pd.concat([hist_only, long_dept], ignore_index=True, sort=False)

# Keep ordering clean
long_dept['month'] = pd.to_datetime(long_dept['month'], errors='coerce')
long_dept = long_dept.sort_values(['vertical', 'department_name', 'month']).reset_index(drop=True)

# -----------------------------
# 13) Build export table (capacity_forecast)
# -----------------------------

# Base frame
cap_wide = long_dept.copy()

# =============================
# Rename core columns
# =============================
cap_wide = cap_wide.rename(columns={
    'month': 'Month',
    'vertical': 'Vertical',
    'department_name': 'Department_name',
    'actual_volume': 'Actual Volume',
    'forecast_base': 'Forecast (Cases)',
    'forecast_human_cases_cal': 'Forecast after Einstein (Human cases)',
    'einstein_solved_forecast': 'Einstein solved forecast',
    'calls_not_indexed_actual': 'Calls not indexed (actual)',
    'calls_not_indexed_forecast': 'Calls not indexed (forecast)',
    'repeats_actual': 'Repeats (actual)',
    'repeats_forecast': 'Repeats (forecast)',
})

# =============================
# Null safety
# =============================
for c in [
    'Calls not indexed (actual)',
    'Calls not indexed (forecast)',
    'Repeats (actual)',
    'Repeats (forecast)',
]:
    if c not in cap_wide.columns:
        cap_wide[c] = 0
    cap_wide[c] = pd.to_numeric(cap_wide[c], errors='coerce').fillna(0)

cap_wide['Actual Volume'] = pd.to_numeric(
    cap_wide.get('Actual Volume'), errors='coerce'
)

cap_wide['Forecast (Cases)'] = pd.to_numeric(
    cap_wide.get('Forecast (Cases)'), errors='coerce'
)

# =============================
# Actual workload (history)
# =============================
cap_wide['Actual Workload (Cases + calls not indexed + repeats)'] = (
    cap_wide['Actual Volume'].fillna(0)
    + cap_wide['Calls not indexed (actual)']
    + cap_wide['Repeats (actual)']
)

# =============================
# Forecast workload for staffing
# =============================
cap_wide['Workload Forecast (Humans + calls not indexed + repeats)'] = (
    cap_wide['Forecast after Einstein (Human cases)'].fillna(0)
    + cap_wide['Calls not indexed (forecast)']
    + cap_wide['Repeats (forecast)']
)

# =============================
# Capacity & productivity (placeholders-safe)
# =============================
if 'productivity_agents' in long_dept.columns:
    cap_wide['Productivity'] = pd.to_numeric(
        long_dept['productivity_agents'], errors='coerce'
    )
else:
    cap_wide['Productivity'] = np.nan

if 'capacity_agents' in long_dept.columns:
    cap_wide['Capacity'] = pd.to_numeric(
        long_dept['capacity_agents'], errors='coerce'
    )
else:
    cap_wide['Capacity'] = np.nan

cap_wide['Expected Workload vs Capacity'] = (
    cap_wide['Workload Forecast (Humans + calls not indexed + repeats)']
    - cap_wide['Capacity']
)

# =============================
# Final export view
# =============================
capacity_forecast_display = cap_wide[[
    'Month', 'Vertical', 'Department_name',

    # History
    'Actual Volume',
    'Calls not indexed (actual)',
    'Repeats (actual)',
    'Actual Workload (Cases + calls not indexed + repeats)',

    # Forecast
    'Forecast (Cases)',
    'Calls not indexed (forecast)',
    'Repeats (forecast)',
    'Einstein solved forecast',
    'Forecast after Einstein (Human cases)',
    'Workload Forecast (Humans + calls not indexed + repeats)',

    # Staffing
    'Capacity',
    'Productivity',
    'Expected Workload vs Capacity'
]].copy()

# =============================
# Export
# =============================
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl', mode='w') as writer:
    monthly_adj.to_excel(writer, sheet_name='Monthly_Forecast_CAL', index=False)
    model_used_error_df.to_excel(writer, sheet_name='Model_Used_and_Error', index=False)
    capacity_forecast_display.to_excel(writer, sheet_name='capacity_forecast', index=False)

print('Export complete →', OUTPUT_XLSX)
