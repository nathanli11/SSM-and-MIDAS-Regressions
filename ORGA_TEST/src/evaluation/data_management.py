import pandas as pd
import numpy as np

# Path du fichier de data brute
file_path = 'data_qg.xlsx'

# Noms des feuille du fichier excel
sheet_names = [
    'GDP', 'T10', 'T1', 'SP', 'IP', 'Emply',
    'Exptn', 'PI', 'LEI',
    'Manu', 'Manu1', 'Oil'
]

def import_data(file_path, sheet_names):
    # Fonction qui importe et traite toute la data
    dfs = {
    name: pd.read_excel(file_path, sheet_name=name)
    for name in sheet_names
}

    # Gestion des dates
    for name, df in dfs.items():
        df['Date'] = pd.to_datetime(df['Date'])
        value_col = df.columns[1]
        df = df.rename(columns={value_col: name})
        df = df.set_index('Date')
        dfs[name] = df[[name]]

    data = pd.concat(dfs.values(), axis=1)
    data = data.sort_index()

    # gestion de la serie manufacturing
    data['Manu_final'] = data['Manu'].combine_first(data['Manu1'])

    data = data.drop(columns=['Manu', 'Manu1'])
    data = data.rename(columns={'Manu_final': 'Manu'})

    # Creation de TERM
    rates = data[['T10', 'T1']].copy()
    rates.index = pd.to_datetime(rates.index)
    rates_m = rates.resample('M').mean()
    rates_m['TERM'] = rates_m['T10'] - rates_m['T1']
    rates_m = rates_m[['TERM']]

    # Variables mensuelles
    monthly_vars = ['SP', 'IP', 'Emply', 'Exptn', 'PI', 'LEI', 'Manu', 'Oil']
    X_m = data[monthly_vars].copy()
    X_m.index = pd.to_datetime(X_m.index)
    X_m = X_m.resample('M').last()
    X_m = X_m.join(rates_m[['TERM']], how='inner')
    X_m = X_m[X_m.index >= '1959-01-01']

    return data, X_m

def log(x):
    # log transformation
    return np.log(x)

def dlog(x):
    # log diff
    return np.log(x).diff()

def quarterly_to_monthly_sparse_from_period(y_q: pd.Series, monthly_index: pd.DatetimeIndex) -> pd.Series:

    # Converti des series trimestrielles en series mensuelles (fin de mois)
    # avec NaN pour les mois sans observation trimestrielle)
    if isinstance(y_q.index, pd.PeriodIndex):
        q_end = y_q.index.to_timestamp(how="end").to_period("M").to_timestamp("M")
        y_map = pd.Series(y_q.values, index=pd.DatetimeIndex(q_end))
    else:
        q_end = pd.to_datetime(y_q.index).to_period("M").to_timestamp("M")
        y_map = pd.Series(y_q.values, index=pd.DatetimeIndex(q_end))

    # créer une série mensuelle avec NaN pour les mois sans observation
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

    # garder les dates ou GDP existe
    gdp_obs = df[col].dropna().astype(float).sort_index()

    # convertir en trimestriel, chaque observation est assignée à son trimestre
    gdp_q = gdp_obs.groupby(gdp_obs.index.to_period("Q")).last()
    gdp_q.name = col
    return gdp_q

def normalize_full_sample(df: pd.DataFrame):
    """Normalise une série (DataFrame 1 colonne) par sa moyenne/écart-type full sample."""
    mu = df[df.columns[0]].mean()
    sd = df[df.columns[0]].std(ddof=0)
    sd = 1.0 if sd == 0 else sd
    return (df[df.columns[0]] - mu) / sd, mu, sd

# Fonctions utilitaires pour les périodes trimestrielles
def quarter_end_months(q_period: pd.Period) -> pd.Timestamp:
    """Dernier mois (fin de trimestre) en fin de mois."""
    return q_period.asfreq("Q").end_time.to_period("M").to_timestamp("M")

def quarter_second_month(q_period: pd.Period) -> pd.Timestamp:
    """2e mois du trimestre en fin de mois: Q1->Feb, Q2->May, Q3->Aug, Q4->Nov."""
    start = q_period.start_time.to_period("M").to_timestamp("M")
    return (start.to_period("M") + 1).to_timestamp("M")  # +1 mois

def to_quarter_period_index(idx):
    """Index -> PeriodIndex trimestriel."""
    return pd.PeriodIndex(idx, freq="Q")

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

# Variables mensuelles stationnaires
X_stationnary = X_m.copy()
# Gestion dates
X_stationnary["Exptn"] = X_stationnary["Exptn"].where(X_stationnary.index >= "1978-01-31")
X_stationnary["Oil"]  = X_stationnary["Oil"].where(X_stationnary.index >= "1982-01-01")
X_stationnary = X_stationnary[X_stationnary.index <= "2024-01-01"]
# Export excel
X_stationnary.to_excel('stationnary_data.xlsx')

# Creation de y_sparse
gdp_q = extract_quarterly_gdp(data, col="GDP")

final_data = X_stationnary.copy()
final_data.index = pd.to_datetime(final_data.index).to_period("M").to_timestamp("M")
y_sparse = quarterly_to_monthly_sparse_from_period(gdp_q, final_data.index)
# Gestion dates
y_sparse = y_sparse[y_sparse.index <= "2024-01-01"]
# Export excel
y_sparse.to_excel('y_sparse.xlsx')