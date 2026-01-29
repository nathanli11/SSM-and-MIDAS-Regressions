import pandas as pd
import numpy as np

from ORGA_TEST.src.ssm.kalman_two_series import (
    build_ssm_two_series_ar1_ml, kalman_filter_minimal, fit_ssm_ml
)
from ORGA_TEST.src.evaluation.recursive_table7 import recursive_forecast_exercise

# Data importation
stat_xl = pd.read_excel('stationnary_data.xlsx', index_col=0, parse_dates=True)
y_parse = pd.read_excel('y_sparse.xlsx', index_col=0, parse_dates=True)

series_names = stat_xl.columns.tolist()
gdp_name = y_parse.columns[0]

gdp_diff_df = y_parse.dropna().diff()
gdp_diff_df.columns = [gdp_name]

final_data = (
    stat_xl
    .merge(gdp_diff_df, left_index=True, right_index=True, how="left")
)

# Boucle à run pour sortir les resultats de forecast et RMSE
for i in range(len(stat_xl.columns)):
  names = [gdp_name, series_names[i]]
  params = fit_ssm_ml(final_data, names)

  G, Q, a, Z, R = build_ssm_two_series_ar1_ml(params)

  x = series_names[i]

  kf = kalman_filter_minimal(
      df_x=final_data,
      G=G, Q=Q, a=a, Z=Z, R=R,
      names=[gdp_name,x],
      scaler=None  
  )

  f_filt = kf["a_filt"][:, 0]
  f_pred = kf["a_pred"][:, 0]

  first_date = [0,1,2,4,5,8]
  second_date = [3] #EXPTN
  third_date = [6]
  fourth_date = [7] #oil

  if not (i in first_date or i in third_date or i in fourth_date):
    continue
  
  elif i in first_date:
    df_forecasts = recursive_forecast_exercise(
      gdp_q_col=final_data[[gdp_name]],
      x_m_col=final_data[[series_names[i]]],
      start_est_q="1958Q1",
      end_est_q="1990Q4",
      names = [gdp_name, series_names[i]],
      eval_start_q="1991Q1",
      eval_end_q="2023Q4",
      horizons=range(1,9),
      gdp_lags=1,
      x_month_lags=6
    )
  elif i in second_date:
    df_forecasts = recursive_forecast_exercise(
      gdp_q_col=final_data[[gdp_name]],
      x_m_col=final_data[[series_names[i]]],
      start_est_q="1978Q1",
      end_est_q="1990Q4",
      names = [gdp_name, series_names[i]],
      eval_start_q="1991Q1",
      eval_end_q="2023Q4",
      horizons=range(1,9),
      gdp_lags=1,
      x_month_lags=6
    )
  elif i in third_date:
    df_forecasts = recursive_forecast_exercise(
      gdp_q_col=final_data[[gdp_name]],
      x_m_col=final_data[[series_names[i]]],
      start_est_q="1967Q1",
      end_est_q="1990Q4",
      names = [gdp_name, series_names[i]],
      eval_start_q="1991Q1",
      eval_end_q="2023Q4",
      horizons=range(1,9),
      gdp_lags=1,
      x_month_lags=6
    )
  elif i in fourth_date:
    df_forecasts = recursive_forecast_exercise(
      gdp_q_col=final_data[[gdp_name]],
      x_m_col=final_data[[series_names[i]]],
      start_est_q="1982Q1",
      end_est_q="1992Q4",
      names = [gdp_name, series_names[i]],
      eval_start_q="1993Q1",
      eval_end_q="2023Q4",
      horizons=range(1,9),
      gdp_lags=1,
      x_month_lags=6
    )

  # 1) Garder uniquement les observations valides pour le SSM
  df_eval = df_forecasts.dropna(subset=["ssm_hat", "gdp_real"]).copy()
  df_eval["sq_err"] = (df_eval["ssm_hat"] - df_eval["gdp_real"]) ** 2

  # 2) RMSE par horizon selon la formule (1/N_h * somme)
  rmse_ssm_by_h = (
      df_eval.groupby("h")
      .agg(
          N_h=("sq_err", "size"),
          RMSE_SSM=("sq_err", lambda s: np.sqrt(s.sum() / len(s)))
      )
      .reset_index()
      .sort_values("h")
  )
  df_forecasts.to_excel(f"forecast_{i}.xlsx")
  rmse_ssm_by_h.to_excel(f"rmse_{i}.xlsx")