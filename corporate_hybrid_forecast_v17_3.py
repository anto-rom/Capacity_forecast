
# coding: utf-8

# 
# # corporate_hybrid_forecast_v17_3
# 
# This notebook extends v17_2 to:
# 
# 1. **Capacity / Productivity from agents** (per month & department) using `input_model/productivity_agents.xlsx`:
#    - **Capacity** (per month) = *mean daily number of agents with* `item_target_day > 1` *×* *mean* `item_target_day` *(over those agents)*.
#    - **Productivity** (per month) = *sum of* `prod_total_model` *(all agents, with or without target)*.
#    - These values feed the **Board_[department]_2627** sheets.
# 
# 2. **New sheet `capacity_forecast`**: a long-format consolidation of all departments and months
#    (Jan-2026 → Feb-2027) with columns: *Month, department_name, KPI, Total* (no "Comments").
# 
# It preserves the v17_2 pipeline (recommended model mapping → daily forecast → monthly (sum) → Einstein deduction → **bias-based calibration** → export boards).
# 

# ## 1) Configuration

# In[16]:


from pathlib import Path
import os
import pandas as pd

BASE_DIR = r'C:\Users\pt3canro\Desktop\CAPACITY'
BASE_DIR = str(Path(BASE_DIR).expanduser().resolve())

INPUT_DIR  = str(Path(BASE_DIR) / 'input_model')
OUTPUT_DIR = str(Path(BASE_DIR) / 'outputs')
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

INCOMING_SOURCE_PATH = os.path.join(INPUT_DIR, 'Incoming_new.xlsx')
INCOMING_SHEET       = 'Main'
DEPT_MAP_PATH  = os.path.join(INPUT_DIR, 'department.xlsx')
DEPT_MAP_SHEET = 'map'
AGENT_CAPACITY_PATH = os.path.join(INPUT_DIR, 'agent_language_n_target.xlsx')
EINSTEIN_PATH       = os.path.join(INPUT_DIR, 'einstein.xlsx')
INVENTORY_PATH      = os.path.join(INPUT_DIR, 'inventory_month.xlsx')
PRODUCTIVITY_AGENTS_PATH = os.path.join(INPUT_DIR, 'productivity_agents.xlsx')

OUTPUT_XLSX = os.path.join(OUTPUT_DIR, 'capacity_forecast_v17_3.xlsx')

HORIZON_MONTHS = 12
VERTICALS_TARGET = ['Payments','Partners','Hospitality']
TARGET_TPH = 6.0
EXCLUDE_DEPARTMENT_NAME_TOKENS = ['PROJ','DIST','KEY','PROXIMIS']

print('OUTPUT_XLSX →', OUTPUT_XLSX)


# ## 2) Helpers

# In[17]:


import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from calendar import monthrange

# Pick first present column name

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

# Normalize headers

def std_cols(df):
    df.columns = [c.strip() for c in df.columns]
    return df

# Month start vectorized

def to_month_start(dt_series):
    s = pd.to_datetime(dt_series, errors='coerce')
    return s.dt.to_period('M').dt.to_timestamp(how='start')

# Working days in a month

def working_days_in_month(year:int, month:int):
    start = datetime(year, month, 1)
    end = datetime(year+1,1,1)-timedelta(days=1) if month==12 else datetime(year,month+1,1)-timedelta(days=1)
    days = pd.date_range(start, end, freq='D')
    return int(np.sum(days.dayofweek < 5))

# Quantile monotonicity guard

def validate_quantiles(dfm):
    viol = dfm[(dfm['forecast_p05_dept']>dfm['forecast_monthly_dept']) | (dfm['forecast_monthly_dept']>dfm['forecast_p95_dept'])]
    if not viol.empty:
        raise ValueError('Quantile order violation in monthly aggregation.')


# ## 3) Forecast engines (Baseline/STL + SARIMAX-7)

# In[18]:


# Baseline (STL / SeasonalNaive fallback)

def forecast_daily_baseline(y: pd.Series, horizon_days: int = 365):
    y = y.asfreq('D').fillna(0)
    if len(y) < 56:
        idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq='D')
        last_week = y[-7:].to_numpy()
        p50 = np.resize(last_week, horizon_days)
        resid = (y[7:] - y.shift(7)[7:]).dropna()
        std = resid.std() if len(resid)>0 else max(1.0, np.sqrt(max(y.mean(),1)))
        p05 = np.clip(p50 - 1.645*std, 0, None)
        p95 = p50 + 1.645*std
        return pd.DataFrame({'date': idx, 'p50': p50, 'p05': p05, 'p95': p95})
    try:
        from statsmodels.tsa.seasonal import STL
        y_box = np.log1p(y)
        stl = STL(y_box, period=7, robust=True)
        res = stl.fit()
        trend, seas, resid = res.trend, res.seasonal, res.resid
        last_trend = trend.iloc[-1]
        std = resid.std() if resid.std()>0 else 0.5
        idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq='D')
        seas_fut = np.resize(seas[-7:].to_numpy(), horizon_days)
        mu_log = last_trend + seas_fut
        p50 = np.expm1(mu_log); p50 = np.clip(p50, 0, None)
        p05 = np.expm1(mu_log - 1.645*std); p05 = np.clip(p05, 0, None)
        p95 = np.expm1(mu_log + 1.645*std)
        return pd.DataFrame({'date': idx, 'p50': p50, 'p05': p05, 'p95': p95})
    except Exception:
        idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq='D')
        last_week = y[-7:].to_numpy()
        p50 = np.resize(last_week, horizon_days)
        resid = (y[7:] - y.shift(7)[7:]).dropna()
        std = resid.std() if len(resid)>0 else max(1.0, np.sqrt(max(y.mean(),1)))
        p05 = np.clip(p50 - 1.645*std, 0, None)
        p95 = p50 + 1.645*std
        return pd.DataFrame({'date': idx, 'p50': p50, 'p05': p05, 'p95': p95})

# SARIMAX-7 (weekly)

def forecast_daily_sarimax(y: pd.Series, horizon_days: int = 365):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        return None
    y = y.asfreq('D').fillna(0)
    y_log = np.log1p(y)
    try:
        model = SARIMAX(y_log, order=(1,0,1), seasonal_order=(1,0,1,7), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=horizon_days)
        mean = np.expm1(pred.predicted_mean).clip(lower=0)
        resid = (y[7:] - y.shift(7)[7:]).dropna()
        std = resid.std() if len(resid)>0 else 1.0
        p05 = np.clip(mean - 1.645*std, 0, None)
        p95 = mean + 1.645*std
        idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq='D')
        return pd.DataFrame({'date': idx, 'p50': mean.values, 'p05': p05.values, 'p95': p95.values})
    except Exception:
        return None


# ## 4) Load inputs

# In[19]:


# Incoming
incoming_raw = pd.read_excel(INCOMING_SOURCE_PATH, sheet_name=INCOMING_SHEET, engine='openpyxl')
std_cols(incoming_raw)

c_date = pick_col(incoming_raw, ['Date','date'])
c_dept = pick_col(incoming_raw, ['department_id','dept_id','Department_ID'])
c_cnt  = pick_col(incoming_raw, ['ticket_total','Ticket_Total','count','tickets','qty'])
if c_date is None or c_dept is None:
    raise KeyError('Incoming_new.xlsx must include Date and department_id columns.')

incoming = incoming_raw[[c_date, c_dept] + ([c_cnt] if c_cnt else [])].copy()
incoming.columns = ['date','department_id'] + (['ticket_total'] if c_cnt else [])
if 'ticket_total' not in incoming.columns:
    incoming['ticket_total'] = 1

incoming['date']  = pd.to_datetime(incoming['date'], errors='coerce')
incoming['month'] = to_month_start(incoming['date'])

# Dept map
dept_map = pd.read_excel(DEPT_MAP_PATH, sheet_name=DEPT_MAP_SHEET, engine='openpyxl')
std_cols(dept_map)
dm_id   = pick_col(dept_map, ['department_id','dept_id','Department_ID'])
dm_name = pick_col(dept_map, ['department_name','Department','dept_name','Department_Name'])
dm_vert = pick_col(dept_map, ['vertical','Vertical'])
if dm_id is None or dm_name is None or dm_vert is None:
    raise KeyError('department.xlsx must contain department_id, department_name and vertical (sheet map).')

dept_map = dept_map[[dm_id, dm_name, dm_vert]].drop_duplicates()
dept_map.columns = ['department_id','department_name','vertical']

incoming['department_id'] = pd.to_numeric(incoming['department_id'], errors='coerce').astype('Int64')
dept_map['department_id'] = pd.to_numeric(dept_map['department_id'], errors='coerce').astype('Int64')

incoming = incoming.merge(dept_map, on='department_id', how='left')

# Scope & exclusions
incoming = incoming[incoming['vertical'].isin(VERTICALS_TARGET)].copy()
mask_excl = pd.Series(False, index=incoming.index)
for tok in EXCLUDE_DEPARTMENT_NAME_TOKENS:
    mask_excl |= incoming['department_name'].astype(str).str.upper().str.contains(tok.upper(), na=False)

incoming = incoming.loc[~mask_excl].copy()

# Monthly actuals
monthly_actuals = (incoming
                   .groupby(['vertical','department_id','department_name','month'], as_index=False)
                   .agg(actual_volume=('ticket_total','sum')))

print('Loaded incoming rows =', len(incoming))


# In[20]:


from pathlib import Path

# Einstein
if Path(EINSTEIN_PATH).exists():
    ein = pd.read_excel(EINSTEIN_PATH, engine='openpyxl'); std_cols(ein)
    e_date = pick_col(ein, ['Date','date']); e_dept = pick_col(ein, ['department_id','dept_id','Department_ID'])
    if e_date and e_dept:
        ein_use = ein[[e_date, e_dept]].copy(); ein_use.columns=['date','department_id']
        ein_use['date'] = pd.to_datetime(ein_use['date'], errors='coerce'); ein_use['month'] = to_month_start(ein_use['date'])
        ein_month = ein_use.groupby(['department_id','month'], as_index=False).size().rename(columns={'size':'einstein_solved'})
    else:
        ein_month = pd.DataFrame(columns=['department_id','month','einstein_solved'])
else:
    ein_month = pd.DataFrame(columns=['department_id','month','einstein_solved'])

# Capacity (historical optional)
if Path(AGENT_CAPACITY_PATH).exists():
    cap = pd.read_excel(AGENT_CAPACITY_PATH, engine='openpyxl'); std_cols(cap)
    c_year  = pick_col(cap, ['Year','year'])
    c_mnum  = pick_col(cap, ['Month_number','month_number','month'])
    c_mstart= pick_col(cap, ['MonthStartDate','monthstartdate'])
    c_deptc = pick_col(cap, ['department_id','dept_id','Department_ID'])
    c_prod_total = pick_col(cap, ['productivity_total','prod_total_model'])
    c_avgpd      = pick_col(cap, ['avg_per_day'])
    c_hours      = pick_col(cap, ['productive'])
    if c_mstart is None and (c_year is not None and c_mnum is not None):
        cap['MonthStartDate'] = pd.to_datetime(cap[c_year].astype(str)+'-'+cap[c_mnum].astype(str)+'-01'); c_mstart='MonthStartDate'
    if c_mstart is None or c_deptc is None:
        monthly_capacity_hist = pd.DataFrame(columns=['department_id','month','capacity'])
    else:
        cols=[c_mstart,c_deptc]; ren={c_mstart:'month', c_deptc:'department_id'}
        if c_prod_total: cols.append(c_prod_total); ren[c_prod_total]='tickets_capacity'
        if c_avgpd:      cols.append(c_avgpd);      ren[c_avgpd]='avg_per_day'
        if c_hours:      cols.append(c_hours);      ren[c_hours]='productive_hours'
        cap_use = cap[cols].rename(columns=ren)
        cap_use['month'] = pd.to_datetime(cap_use['month'], errors='coerce').dt.to_period('M').dt.to_timestamp(how='start')
        if 'tickets_capacity' in cap_use.columns:
            monthly_capacity_hist = cap_use.groupby(['department_id','month'], as_index=False).agg(capacity=('tickets_capacity','sum'))
        elif 'avg_per_day' in cap_use.columns:
            cap_use['wdays'] = cap_use['month'].apply(lambda d: working_days_in_month(d.year, d.month))
            cap_use['capacity'] = cap_use['avg_per_day'] * cap_use['wdays']
            monthly_capacity_hist = cap_use.groupby(['department_id','month'], as_index=False).agg(capacity=('capacity','sum'))
        elif 'productive_hours' in cap_use.columns:
            cap_use['capacity'] = cap_use['productive_hours'] * TARGET_TPH
            monthly_capacity_hist = cap_use.groupby(['department_id','month'], as_index=False).agg(capacity=('capacity','sum'))
        else:
            monthly_capacity_hist = pd.DataFrame(columns=['department_id','month','capacity'])
else:
    monthly_capacity_hist = pd.DataFrame(columns=['department_id','month','capacity'])

# Inventory (optional)
if Path(INVENTORY_PATH).exists():
    inv = pd.read_excel(INVENTORY_PATH, engine='openpyxl'); std_cols(inv)
    i_date = pick_col(inv, ['Date','date']); i_dept = pick_col(inv, ['department_id','dept_id','Department_ID'])
    if i_date and i_dept:
        inv_use = inv[[i_date,i_dept]].copy(); inv_use.columns=['date','department_id']
        inv_use['date']=pd.to_datetime(inv_use['date'], errors='coerce'); inv_use['month']=to_month_start(inv_use['date'])
        inv_daily = inv_use.groupby(['department_id','date'], as_index=False).size().rename(columns={'size':'open_count'})
        inv_daily['month']=to_month_start(inv_daily['date'])
        monthly_inventory = inv_daily.groupby(['department_id','month'], as_index=False).agg(inventory_mean=('open_count','mean'))
    else:
        monthly_inventory = pd.DataFrame(columns=['department_id','month','inventory_mean'])
else:
    monthly_inventory = pd.DataFrame(columns=['department_id','month','inventory_mean'])


# ## 5) Recommended model mapping

# In[21]:


recommended = {}  # department_id -> engine label
# Option 1: read from previous remediation report (v17)
try:
    prev = pd.read_excel(os.path.join(OUTPUT_DIR, 'capacity_forecast_v17.xlsx'), sheet_name='Remediation_Report')
    if {'department_id','Recommended_Model'}.issubset(prev.columns):
        for _,r in prev.iterrows():
            if pd.notna(r['department_id']) and pd.notna(r['Recommended_Model']):
                recommended[int(r['department_id'])] = str(r['Recommended_Model'])
except Exception:
    pass
# Option 2: read manual CSV (department_id,model)
try:
    rm_csv = os.path.join(INPUT_DIR,'recommended_models.csv')
    if Path(rm_csv).exists():
        dfm = pd.read_csv(rm_csv)
        if {'department_id','model'}.issubset(dfm.columns):
            for _,r in dfm.iterrows():
                recommended[int(r['department_id'])] = str(r['model'])
except Exception:
    pass
print('Recommended models loaded for', len(recommended), 'departments (fallback Baseline for others).')


# ## 6) Daily forecast using per-department engine

# In[22]:


incoming_daily = (incoming.groupby(['department_id','date'], as_index=False)
                  .agg(tickets=('ticket_total','sum')))

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
    fc_daily_built = fc_daily_built[~fc_daily_built['department_name'].astype(str).str.upper().str.contains(tok.upper(), na=False)]

fc_daily_built.rename(columns={'p50':'forecast_daily_dept','p05':'p05_daily_dept','p95':'p95_daily_dept'}, inplace=True)
print('Daily forecast rows:', len(fc_daily_built))


# ## 7) Monthly aggregation + Einstein deduction

# In[23]:


# Monthly sum
monthly_fc_raw = (fc_daily_built.assign(month=to_month_start(fc_daily_built['date']))
                  .groupby(['vertical','department_id','department_name','month'], as_index=False)
                  .agg(forecast_monthly_dept=('forecast_daily_dept','sum'),
                       forecast_p05_dept=('p05_daily_dept','sum'),
                       forecast_p95_dept=('p95_daily_dept','sum')))
validate_quantiles(monthly_fc_raw)

# Einstein deduction (using 3-month recent per department)
if not ein_month.empty:
    hist_rates = monthly_actuals.merge(ein_month, on=['department_id','month'], how='left')
    hist_rates['einstein_solved'] = hist_rates['einstein_solved'].fillna(0)
    hist_rates['einstein_rate'] = (hist_rates['einstein_solved'] / hist_rates['actual_volume']).replace([np.inf,-np.inf], 0).fillna(0)
    recent = (hist_rates.sort_values('month')
              .groupby('department_id')
              .apply(lambda g: g.set_index('month')['einstein_rate'].tail(3).mean())
              .rename('einstein_rate_recent').reset_index())
else:
    recent = pd.DataFrame(columns=['department_id','einstein_rate_recent'])

monthly_adj = monthly_fc_raw.merge(recent, on='department_id', how='left')
monthly_adj['einstein_rate_recent'] = monthly_adj['einstein_rate_recent'].fillna(0).clip(0, 0.9)
for c in ['forecast_monthly_dept','forecast_p05_dept','forecast_p95_dept']:
    monthly_adj[c + '_post_einstein'] = monthly_adj[c] * (1 - monthly_adj['einstein_rate_recent'])

print('Monthly + Einstein done → rows:', len(monthly_adj))


# ## 8) Bias-based calibration (from Model_Used_and_Error table)

# In[24]:


# Build Model_Used_and_Error from user's table (can be replaced by a dynamic backtest later)
model_used_error_df = pd.DataFrame([
    {'vertical':'Hospitality','department_id':7,'department_name':'CS_PMSH_L1','model_used':'STL','backtest_months':12,'mape_pct':29.35,'wape_pct':26.00,'bias_pct':-0.40},
    {'vertical':'Hospitality','department_id':8,'department_name':'CS_PMSP_CLOUD_L1','model_used':'STL','backtest_months':12,'mape_pct':43.80,'wape_pct':39.12,'bias_pct':12.64},
    {'vertical':'Hospitality','department_id':11,'department_name':'CS_PMSP_CLOUD_L2','model_used':'STL','backtest_months':12,'mape_pct':27.76,'wape_pct':22.27,'bias_pct':-4.06},
    {'vertical':'Hospitality','department_id':23,'department_name':'CS_PMSP_FRANCE','model_used':'STL','backtest_months':12,'mape_pct':31.44,'wape_pct':25.79,'bias_pct':-1.79},
    {'vertical':'Hospitality','department_id':5,'department_name':'CS_PMSP_INTEG','model_used':'STL','backtest_months':12,'mape_pct':20.88,'wape_pct':17.68,'bias_pct':-2.70},
    {'vertical':'Hospitality','department_id':9,'department_name':'CS_PMSP_PREM_L1','model_used':'STL','backtest_months':12,'mape_pct':17.29,'wape_pct':14.44,'bias_pct':-3.07},
    {'vertical':'Hospitality','department_id':10,'department_name':'CS_PMSP_PREM_L2','model_used':'STL','backtest_months':12,'mape_pct':28.77,'wape_pct':23.23,'bias_pct':-1.12},
    {'vertical':'Partners','department_id':12,'department_name':'CS_PART_APAC','model_used':'STL','backtest_months':12,'mape_pct':20.88,'wape_pct':20.99,'bias_pct':-5.14},
    {'vertical':'Partners','department_id':13,'department_name':'CS_PART_EMEA','model_used':'STL','backtest_months':12,'mape_pct':27.76,'wape_pct':25.91,'bias_pct':7.36},
    {'vertical':'Partners','department_id':14,'department_name':'CS_PART_LATAM','model_used':'STL','backtest_months':12,'mape_pct':26.60,'wape_pct':21.38,'bias_pct':5.29},
    {'vertical':'Partners','department_id':15,'department_name':'CS_PART_US','model_used':'STL','backtest_months':12,'mape_pct':40.03,'wape_pct':30.97,'bias_pct':-9.47},
    {'vertical':'Payments','department_id':3,'department_name':'CA_PYAC','model_used':'STL','backtest_months':12,'mape_pct':18.63,'wape_pct':18.70,'bias_pct':1.28},
    {'vertical':'Payments','department_id':1,'department_name':'CS_GT3C_EU','model_used':'STL','backtest_months':12,'mape_pct':24.80,'wape_pct':22.19,'bias_pct':0.58},
    {'vertical':'Payments','department_id':18,'department_name':'Datatrans L2 Customer Support','model_used':'STL','backtest_months':12,'mape_pct':54.89,'wape_pct':38.55,'bias_pct':-7.28},
    {'vertical':'Payments','department_id':2,'department_name':'L2 Customer Support','model_used':'STL','backtest_months':12,'mape_pct':28.56,'wape_pct':24.46,'bias_pct':-1.36},
    {'vertical':'Payments','department_id':21,'department_name':'Specialist - L2 Customer Support','model_used':'STL','backtest_months':12,'mape_pct':42.25,'wape_pct':41.25,'bias_pct':17.26},
])

calib_from_bias = model_used_error_df[['department_id','bias_pct']].copy()
calib_from_bias['department_id'] = pd.to_numeric(calib_from_bias['department_id'], errors='coerce')
calib_from_bias['calib_factor'] = (1 - calib_from_bias['bias_pct']/100.0).clip(0.70, 1.30)

monthly_adj = monthly_adj.drop(columns=['calib_factor'], errors='ignore')
monthly_adj = monthly_adj.merge(calib_from_bias[['department_id','calib_factor']], on='department_id', how='left')
monthly_adj['calib_factor'] = monthly_adj['calib_factor'].fillna(1.0)

for c in ['forecast_monthly_dept_post_einstein','forecast_p05_dept_post_einstein','forecast_p95_dept_post_einstein']:
    monthly_adj[c + '_cal'] = monthly_adj[c] * monthly_adj['calib_factor']

print('Calibration applied. Share of rows with calib_factor != 1:', (monthly_adj['calib_factor']!=1.0).mean().round(2))


# ## 9) Capacity & Productivity from productivity_agents.xlsx (agents)

# In[25]:


# The file contains daily agent-level (or agent-day) information.
# We compute per department-month:
#   - agents_gt1_day: daily count of agents with item_target_day > 1 (unique agents if available, else row count)
#   - target_mean_gt1_day: daily mean of item_target_day over rows with item_target_day > 1
#   - prod_sum_day: daily sum of prod_total_model (all rows, no filter)
# Then:
#   - Capacity_month = mean(agents_gt1_day across the month) * mean(target_mean_gt1_day across the month)
#   - Productivity_month = sum(prod_sum_day across the month)

if Path(PRODUCTIVITY_AGENTS_PATH).exists():
    pa = pd.read_excel(PRODUCTIVITY_AGENTS_PATH, engine='openpyxl')
    std_cols(pa)
    c_date = pick_col(pa, ['date','Date','work_date'])
    c_dept = pick_col(pa, ['department_id','dept_id','Department_ID'])
    c_agent= pick_col(pa, ['agent_id','Agent_ID','worker_id'])
    c_target = pick_col(pa, ['item_target_day','target_day','item_target'])
    c_prod   = pick_col(pa, ['prod_total_model','productivity','prod_model'])

    if c_date is None or c_dept is None or c_target is None or c_prod is None:
        print('WARNING: productivity_agents.xlsx missing required columns. Skipping agents-based capacity/productivity.')
        cap_prod_month = pd.DataFrame(columns=['department_id','month','capacity_agents','productivity_agents'])
    else:
        pa = pa[[c_date, c_dept] + ([c_agent] if c_agent else []) + [c_target, c_prod]].copy()
        pa.columns = ['date','department_id'] + (['agent_id'] if c_agent else []) + ['item_target_day','prod_total_model']
        pa['date'] = pd.to_datetime(pa['date'], errors='coerce')
        pa['month'] = to_month_start(pa['date'])
        pa['department_id'] = pd.to_numeric(pa['department_id'], errors='coerce')
        pa['item_target_day'] = pd.to_numeric(pa['item_target_day'], errors='coerce')
        pa['prod_total_model'] = pd.to_numeric(pa['prod_total_model'], errors='coerce')

        # Daily aggregates by department
        if 'agent_id' in pa.columns:
            # unique agents per day with target>1
            daily_gt1 = (pa[pa['item_target_day']>1]
                         .groupby(['department_id','date'])['agent_id'].nunique()
                         .rename('agents_gt1_day').reset_index())
        else:
            # fallback: count rows as agent-day records
            daily_gt1 = (pa[pa['item_target_day']>1]
                         .groupby(['department_id','date'], as_index=False)
                         .size().rename(columns={'size':'agents_gt1_day'}))

        daily_target = (pa[pa['item_target_day']>1]
                        .groupby(['department_id','date'], as_index=False)
                        .agg(target_mean_gt1_day=('item_target_day','mean')))

        daily_prod = (pa.groupby(['department_id','date'], as_index=False)
                      .agg(prod_sum_day=('prod_total_model','sum')))

        # Merge daily pieces
        daily_all = (daily_gt1.merge(daily_target, on=['department_id','date'], how='outer')
                               .merge(daily_prod, on=['department_id','date'], how='outer'))
        daily_all['month'] = to_month_start(daily_all['date'])

        # Monthly metrics
        cap_prod_month = (daily_all.groupby(['department_id','month'], as_index=False)
                          .agg(agents_gt1_month_mean = ('agents_gt1_day','mean'),
                               target_gt1_month_mean = ('target_mean_gt1_day','mean'),
                               productivity_agents   = ('prod_sum_day','sum')))

        cap_prod_month['capacity_agents'] = cap_prod_month['agents_gt1_month_mean'] * cap_prod_month['target_gt1_month_mean']
        cap_prod_month = cap_prod_month[['department_id','month','capacity_agents','productivity_agents']]
else:
    print('WARNING: productivity_agents.xlsx not found → capacity/productivity from agents will be empty.')
    cap_prod_month = pd.DataFrame(columns=['department_id','month','capacity_agents','productivity_agents'])

print('Agents-based capacity/productivity rows:', len(cap_prod_month))


# ## 10) Capacity assembly (hist + projection) and injection of agents KPIs

# In[26]:


# Build monthly_capacity_all (hist + projection for HORIZON_MONTHS)
if 'monthly_capacity_hist' not in globals() or monthly_capacity_hist is None:
    monthly_capacity_hist = pd.DataFrame(columns=['department_id','month','capacity'])

if not monthly_capacity_hist.empty:
    monthly_capacity_hist = monthly_capacity_hist.copy()
    monthly_capacity_hist['department_id'] = pd.to_numeric(monthly_capacity_hist['department_id'], errors='coerce')
    monthly_capacity_hist['month'] = pd.to_datetime(monthly_capacity_hist['month'], errors='coerce').dt.to_period('M').dt.to_timestamp(how='start')
    monthly_capacity_hist['capacity'] = pd.to_numeric(monthly_capacity_hist['capacity'], errors='coerce')

last_hist_candidates = []
if not monthly_actuals.empty:
    last_hist_candidates.append(pd.to_datetime(monthly_actuals['month'], errors='coerce').max())
if not monthly_capacity_hist.empty:
    last_hist_candidates.append(monthly_capacity_hist['month'].max())

if len(last_hist_candidates)>0 and pd.notna(max(last_hist_candidates)):
    last_hist_month = max(last_hist_candidates)
else:
    last_hist_month = pd.Timestamp.today().to_period('M').to_timestamp(how='start')

first_future_month = last_hist_month + pd.offsets.MonthBegin(1)
future_months = pd.date_range(first_future_month, periods=HORIZON_MONTHS, freq='MS')

proj_rows = []
if not monthly_capacity_hist.empty and len(future_months)>0:
    for dpt_id, g in monthly_capacity_hist.groupby('department_id'):
        last3_mean = pd.to_numeric(g.sort_values('month').tail(3)['capacity'], errors='coerce').mean()
        for m in future_months:
            proj_rows.append({'department_id': dpt_id, 'month': m, 'capacity': last3_mean})
cap_proj = pd.DataFrame(proj_rows, columns=['department_id','month','capacity'])
monthly_capacity_all = pd.concat([monthly_capacity_hist, cap_proj], ignore_index=True)

# Inject agents KPIs to override capacity/productivity for boards
# Merge cap_prod_month to monthly_adj keys (department_id + month)
cap_prod_month['month'] = pd.to_datetime(cap_prod_month['month'], errors='coerce').dt.to_period('M').dt.to_timestamp(how='start')

# We'll assemble long_dept from calibrated forecast and then attach agents capacity/productivity


# ## 11) Build long_dept (calibrated forecast + actuals + agents KPIs + inventory)

# In[27]:


# Use calibrated forecast for boards
fc_board = monthly_adj.rename(columns={'forecast_monthly_dept_post_einstein_cal': 'forecast_adj_cal'})
fc_board['forecast_adj_cal'] = monthly_adj['forecast_monthly_dept_post_einstein_cal']

# Base long_dept (bring actuals)
long_dept = (fc_board[['vertical','department_id','department_name','month','forecast_adj_cal']]
             .merge(monthly_actuals[['vertical','department_id','department_name','month','actual_volume']],
                    on=['vertical','department_id','department_name','month'], how='left'))

# Attach inventory
long_dept = long_dept.merge(monthly_inventory, on=['department_id','month'], how='left')

# Attach capacity (from agents) and productivity (from agents)
long_dept = long_dept.merge(cap_prod_month, on=['department_id','month'], how='left')

# Fallback: if no agents data, try historical capacity
long_dept = long_dept.merge(monthly_capacity_all, on=['department_id','month'], how='left', suffixes=('', '_hist'))

# Decide final capacity/productivity columns for the boards
# - Capacity: prefer agents-based capacity when available; else use historical
long_dept['capacity'] = np.where(long_dept['capacity_agents'].notna(), long_dept['capacity_agents'], long_dept['capacity'])

# - Productivity: prefer agents-based productivity when available; else (if nothing) fallback to capacity
long_dept['productivity'] = np.where(long_dept['productivity_agents'].notna(), long_dept['productivity_agents'], long_dept['capacity'])

# Final forecast value
long_dept['forecast'] = long_dept['forecast_adj_cal']

# Clean numeric types
for c in ['forecast','actual_volume','capacity','productivity','inventory_mean']:
    if c in long_dept.columns:
        long_dept[c] = pd.to_numeric(long_dept[c], errors='coerce')

print('long_dept rows:', len(long_dept))


# ## 12) Build Board_[department]_2627 with agents-based KPIs

# In[28]:


# --- Month labels (numeric) ---
start_month = pd.Timestamp('2026-01-01')
end_month   = pd.Timestamp('2027-02-01')
month_range = pd.date_range(start_month, end_month, freq='MS')

# Example: ['01-26', '02-26', ..., '01-27', '02-27']
labels = pd.date_range(start_month, end_month, freq='MS').strftime('%m-%y').tolist()


# --- Alignment helper ---
def to_row_range(values: pd.Series, months: pd.Series, label_list: list) -> pd.Series:
    """
    Align a series of values to the given month label list using '%m-%y' format.
    - values: numeric/any values to align
    - months: datetime-like Series (must be convertible to datetime)
    - label_list: list of '%m-%y' strings used as the target index
    """
    # Ensure months is datetime
    m = pd.to_datetime(months, errors='coerce')

    # Build labels in '%m-%y' directly (avoids month_map)
    lab = m.dt.strftime('%m-%y')

    # Create a Series indexed by those labels
    s = pd.Series(values.to_numpy(), index=lab.to_numpy())

    # Reindex to the provided label list to preserve order and include missing as NaN
    return s.reindex(label_list)


# --- Build one board matrix ---
def build_dept_matrix_range(long_dept_in: pd.DataFrame, department_name: str) -> pd.DataFrame:
    base = long_dept_in.copy()
    base = base[
        (base['department_name'] == department_name) &
        (base['month'] >= start_month) &
        (base['month'] <= end_month)
    ]

    agg_cols = {
        'forecast': 'sum',
        'actual_volume': 'sum',
        'capacity': 'sum',
        'productivity': 'sum',
        'inventory_mean': 'mean'
    }

    base = (
        base.groupby('month', as_index=False)
            .agg(**{k: pd.NamedAgg(column=k, aggfunc=v) for k, v in agg_cols.items()})
    )

    # Reindex to ensure we have a row for every month in the range
    base = base.set_index('month').reindex(month_range).reset_index().rename(columns={'index': 'month'})

    # Coerce numeric fields
    for c in ['forecast', 'actual_volume', 'capacity', 'productivity', 'inventory_mean']:
        base[c] = pd.to_numeric(base[c], errors='coerce')

    # Rows aligned to the numeric month labels
    row_forecast = to_row_range(base['forecast'].round(0),      base['month'], labels)
    row_actual   = to_row_range(base['actual_volume'].round(0), base['month'], labels)

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_vals = 100.0 - ((base['forecast'] - base['actual_volume']).abs() / base['actual_volume'] * 100.0)
    row_acc = to_row_range(acc_vals.round(1), base['month'], labels)

    row_capacity = to_row_range(base['capacity'],     base['month'], labels)
    row_prod     = to_row_range(base['productivity'], base['month'], labels)

    # % difference capacity vs productivity (positive if capacity>productivity)
    row_capacity_num = pd.to_numeric(row_capacity, errors='coerce')
    row_prod_num     = pd.to_numeric(row_prod, errors='coerce')
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_cap_prod = ((row_capacity_num - row_prod_num) / row_capacity_num * 100.0)
    row_diff_cap_prod = diff_cap_prod.round(0).fillna(0).astype(int).astype(str) + '%'

    row_exp_vs_cap = to_row_range((base['forecast'] - base['capacity']).round(0), base['month'], labels)

    # % diff between Forecast and Actual Volume
    with np.errstate(divide='ignore', invalid='ignore'):
        avp_vals = ((base['forecast'] - base['actual_volume']) / base['actual_volume'] * 100.0)
    row_act_vs_prod = to_row_range(avp_vals.round(1), base['month'], labels)

    row_inventory = to_row_range(base['inventory_mean'].round(2), base['month'], labels)
    row_comments  = pd.Series([''] * len(labels), index=labels)

    mat = pd.DataFrame({
        'Comments': row_comments,
        'Forecast': row_forecast,
        'Actual Volume': row_actual,
        'Forecast Accuracy': row_acc,
        'Capacity': row_capacity,
        'Productivity': row_prod,
        'Difference Capacity vs Productivity': row_diff_cap_prod,
        'Expected Forecast vs Capacity': row_exp_vs_cap,
        'Actual Volume vs Productivity': row_act_vs_prod,
        'Inventory': row_inventory
    }).T

    return mat


# ## 13) Build consolidated sheet `capacity_forecast` (no Comments)

# In[29]:


# Build a long-format dataset by stacking each department's matrix rows (excluding Comments)

def flatten_board_to_long(dept_name: str, mat: pd.DataFrame, labels: list) -> pd.DataFrame:
    # mat is row-indexed by KPI names, columns are month labels
    mat2 = mat.loc[[k for k in mat.index if k != 'Comments']].copy()
    records = []
    for kpi in mat2.index:
        for m in labels:
            val = mat2.loc[kpi, m] if m in mat2.columns else np.nan
            records.append({'Month': m, 'department_name': dept_name, 'KPI': kpi, 'Total': val})
    return pd.DataFrame(records)

capacity_forecast_long = []
for dname in sorted(long_dept['department_name'].dropna().unique().tolist()):
    mat = build_dept_matrix_range(long_dept, dname)
    df_long = flatten_board_to_long(dname, mat, labels)
    capacity_forecast_long.append(df_long)

capacity_forecast = pd.concat(capacity_forecast_long, ignore_index=True) if capacity_forecast_long else pd.DataFrame(columns=['Month','department_name','KPI','Total'])
print('capacity_forecast rows:', len(capacity_forecast))


# ## 14) Export to Excel

# In[30]:


# -------------------------------------------------------------------
# 1) Correct dataframe aliases (prevents NameError)
# -------------------------------------------------------------------
# In v17_3, the sheet "Monthly_Forecast_CAL" is exported from `monthly_adj`.
monthly_forecast_cal_df = monthly_adj.copy()

# In v17_3, the sheet "Model_Used_and_Error" is exported from `model_used_error_df`.
model_used_and_error_df = model_used_error_df.copy()

# -------------------------------------------------------------------
# 2) Build the WIDE "capacity_forecast" sheet with the requested layout
#    Source of truth: `long_dept` (month + dept + vertical + forecast + actual + capacity + productivity + inventory)
# -------------------------------------------------------------------
cap_wide = long_dept.copy()

# Rename columns to match the requested display names
cap_wide = cap_wide.rename(columns={
    "month": "Month",
    "vertical": "Vertical",
    "department_name": "Department_name",
    "inventory_mean": "Inventory",
    "actual_volume": "Actual Volume",
    "productivity": "Productivity",
    "capacity": "Capacity",
    # IMPORTANT: Use the monthly forecast value already used in the boards (`long_dept['forecast']`)
    "forecast": "Forecast",
})

# Ensure numeric types (avoids string arithmetic issues)
for c in ["Inventory", "Actual Volume", "Productivity", "Capacity", "Forecast"]:
    if c in cap_wide.columns:
        cap_wide[c] = pd.to_numeric(cap_wide[c], errors="coerce")

# -------------------------------------------------------------------
# 3) Derived metrics 
# -------------------------------------------------------------------
# Actual Volume vs Productivity = Actual Volume - Productivity
cap_wide["Actual Volume vs Productivity"] = cap_wide["Actual Volume"] - cap_wide["Productivity"]

# Forecast accuracy = % difference between Forecast and Actual Volume
# Implemented as signed % difference: (Forecast - Actual) / Actual
# If you prefer absolute % error, replace with abs(Forecast - Actual) / Actual
cap_wide["Forecast accuracy"] = np.where(
    cap_wide["Actual Volume"].notna() & (cap_wide["Actual Volume"] != 0),
    (cap_wide["Forecast"] - cap_wide["Actual Volume"]) / cap_wide["Actual Volume"],
    np.nan
)

# Diference Capacity vs Productivity = Productivity - Capacity
cap_wide["Diference Capacity vs Productivity"] = cap_wide["Productivity"] - cap_wide["Capacity"]

# Expected Forecast vs Capacity = Forecast - Capacity
cap_wide["Expected Forecast vs Capacity"] = cap_wide["Forecast"] - cap_wide["Capacity"]

# -------------------------------------------------------------------
# 4) Final column order
# -------------------------------------------------------------------
capacity_forecast_display = cap_wide[[
    "Month",
    "Vertical",
    "Department_name",
    "Inventory",
    "Actual Volume",
    "Productivity",
    "Actual Volume vs Productivity",
    "Forecast",
    "Forecast accuracy",
    "Diference Capacity vs Productivity",
    "Capacity",
    "Expected Forecast vs Capacity"
]].copy()

# -------------------------------------------------------------------
# 5) Export ONLY the three required worksheets
# -------------------------------------------------------------------
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as writer:
    monthly_forecast_cal_df.to_excel(writer, sheet_name="Monthly_Forecast_CAL", index=False)
    model_used_and_error_df.to_excel(writer, sheet_name="Model_Used_and_Error", index=False)
    capacity_forecast_display.to_excel(writer, sheet_name="capacity_forecast", index=False)

print("Export complete →", OUTPUT_XLSX)

