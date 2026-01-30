import pandas as pd
from src.evaluation.data_management import (
    import_data, log, dlog, extract_quarterly_gdp, quarterly_to_monthly_sparse_from_period
)


# Path du fichier de data brute
file_path = r'data/data.xlsx'

# Noms des feuille du fichier excel
sheet_names = [
    'GDP', 'T10', 'T1', 'SP', 'IP', 'Emply',
    'Exptn', 'PI', 'LEI',
    'Manu', 'Manu1', 'Oil'
]

# Transformations
data, X_m = import_data(file_path, sheet_names)
print(data)
#levels_var = ['TERM', 'LEI']
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
X_stationnary.to_excel(r'data/stationnary_data.xlsx')

# Creation de y_sparse
gdp_q = extract_quarterly_gdp(data, col="GDP")

final_data = X_stationnary.copy()
final_data.index = pd.to_datetime(final_data.index).to_period("M").to_timestamp("M")
y_sparse = quarterly_to_monthly_sparse_from_period(gdp_q, final_data.index)
# Gestion dates
y_sparse = y_sparse[y_sparse.index <= "2024-01-01"]
# Export excel
y_sparse.to_excel(r'data/y_sparse.xlsx')