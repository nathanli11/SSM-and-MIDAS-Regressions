import pandas as pd
import numpy as np


file_path = 'data_qg.xlsx'

sheet_names = [
    'GDP', 'T10', 'T1', 'SP', 'IP', 'Emply',
    'Exptn', 'PI', 'LEI',
    'Manu', 'Manu1', 'Oil'
]

def import_data(file_path, sheet_names):
    dfs = {
    name: pd.read_excel(file_path, sheet_name=name)
    for name in sheet_names
}

    for name, df in dfs.items():
        df['Date'] = pd.to_datetime(df['Date'])
        value_col = df.columns[1]
        df = df.rename(columns={value_col: name})
        df = df.set_index('Date')
        dfs[name] = df[[name]]

    data = pd.concat(dfs.values(), axis=1)
    data = data.sort_index()

    data['Manu_final'] = data['Manu'].combine_first(data['Manu1'])

    data = data.drop(columns=['Manu', 'Manu1'])
    data = data.rename(columns={'Manu_final': 'Manu'})

    rates = data[['T10', 'T1']].copy()
    rates.index = pd.to_datetime(rates.index)
    rates_m = rates.resample('M').mean()
    rates_m['TERM'] = rates_m['T10'] - rates_m['T1']
    rates_m = rates_m[['TERM']]
    monthly_vars = ['SP', 'IP', 'Emply', 'Exptn', 'PI', 'LEI', 'Manu', 'Oil']
    X_m = data[monthly_vars].copy()
    X_m.index = pd.to_datetime(X_m.index)
    X_m = X_m.resample('M').last()
    X_m = X_m.join(rates_m[['TERM']], how='inner')
    X_m = X_m[X_m.index >= '1959-01-01']

    return data, X_m

def log(x):
    return np.log(x)

def dlog(x):
    return np.log(x).diff()

def quarterly_to_monthly_sparse_from_period(y_q: pd.Series, monthly_index: pd.DatetimeIndex) -> pd.Series:
    if isinstance(y_q.index, pd.PeriodIndex):
        q_end = y_q.index.to_timestamp(how="end").to_period("M").to_timestamp("M")
        y_map = pd.Series(y_q.values, index=pd.DatetimeIndex(q_end))
    else:
        q_end = pd.to_datetime(y_q.index).to_period("M").to_timestamp("M")
        y_map = pd.Series(y_q.values, index=pd.DatetimeIndex(q_end))

    y_sparse = pd.Series(np.nan, index=monthly_index, name="y")
    common = monthly_index.intersection(y_map.index)
    y_sparse.loc[common] = y_map.loc[common].values
    return y_sparse

def extract_quarterly_gdp(data: pd.DataFrame, col="GDP") -> pd.Series:
    # index datetime propre
    df = data.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    # garder les dates où GDP existe
    gdp_obs = df[col].dropna().astype(float).sort_index()

    # convertir en trimestriel : chaque observation est assignée à son trimestre
    # (si tu as une observation par trimestre, c'est parfait)
    gdp_q = gdp_obs.groupby(gdp_obs.index.to_period("Q")).last()
    gdp_q.name = col
    return gdp_q


# Transformations
data, X_m = import_data(file_path, sheet_names)
print(data)
levels_var = ['TERM', 'LEI']
log_vars = ['Exptn']
dlog_vars = ['SP', 'IP', 'Emply', 'PI', 'Manu', 'Oil']

for v in log_vars:
    X_m[v] = log(X_m[v])

for v in dlog_vars:
    X_m[v] = dlog(X_m[v])

X_stationnary = X_m.copy()
X_stationnary["Exptn"] = X_stationnary["Exptn"].where(X_stationnary.index >= "1978-01-31")
X_stationnary["Oil"]  = X_stationnary["Oil"].where(X_stationnary.index >= "1982-01-01")
X_stationnary = X_stationnary[X_stationnary.index <= "2024-01-01"]
X_stationnary.to_excel('stationnary_data.xlsx')

gdp_q = extract_quarterly_gdp(data, col="GDP")

final_data = X_stationnary.copy()
final_data.index = pd.to_datetime(final_data.index).to_period("M").to_timestamp("M")
y_sparse = quarterly_to_monthly_sparse_from_period(gdp_q, final_data.index)
y_sparse = y_sparse[y_sparse.index <= "2024-01-01"]
y_sparse.to_excel('y_sparse.xlsx')