import pandas as pd
import numpy as np
from pathlib import Path

from src.ssm.kalman_two_series import build_ssm_two_series_ar1_ml, fit_ssm_ml
from src.evaluation.recursive_table7 import recursive_forecast_exercise, table7_rmse_grid

# Output dir
OUTDIR_RMSE_KALMAN = Path("scripts/results/table_7_8/results_rmse_kalman_2024")
OUTDIR_RMSE_KALMAN.mkdir(parents=True, exist_ok=True)
OUTDIR_RMSE_MIDAS = Path("scripts/results/table_7_8/results_rmse_midas_2024")
OUTDIR_RMSE_MIDAS.mkdir(parents=True, exist_ok=True)
OUTDIR_FORECAST = Path("scripts/results/table_7_8")
OUTDIR_FORECAST.mkdir(parents=True, exist_ok=True)

# Data importation
stat_xl = pd.read_excel(r"data/stationnary_data.xlsx", index_col=0, parse_dates=True)
y_parse = pd.read_excel(r"data/y_sparse.xlsx", index_col=0, parse_dates=True)

series_names = stat_xl.columns.tolist()
gdp_name = y_parse.columns[0]

# GDP en croissance trimestrielle (différences premières des observations de fin de trimestre)
gdp_diff_df = y_parse.dropna().diff()
gdp_diff_df.columns = [gdp_name]

final_data = stat_xl.merge(gdp_diff_df, left_index=True, right_index=True, how="left")

# GDP en croissance trimestrielle pour MIDAS
y_q = gdp_diff_df[gdp_name].copy()
y_q.index = pd.PeriodIndex(pd.to_datetime(y_q.index), freq="Q")
y_q = y_q.sort_index().asfreq("Q")

# indices de séries gardées
first_date = [0, 1, 2, 4, 5, 8]
second_date = [3]   # EXPTN
third_date = [6]
fourth_date = [7]   # oil

for i in range(len(series_names)):
    if not (i in first_date or i in second_date or i in third_date or i in fourth_date):
        continue

    x_name = series_names[i]
    names = [gdp_name, x_name]

    # -----------------------
    # Choix fenêtres
    # -----------------------
    if i in first_date:
        start_est_q, end_est_q = "1958Q1", "1990Q4"
        eval_start_q, eval_end_q = "1991Q1", "2023Q4"
    elif i in second_date:
        start_est_q, end_est_q = "1978Q1", "1990Q4"
        eval_start_q, eval_end_q = "1991Q1", "2023Q4"
    elif i in third_date:
        start_est_q, end_est_q = "1967Q1", "1990Q4"
        eval_start_q, eval_end_q = "1991Q1", "2023Q4"
    else:
        start_est_q, end_est_q = "1982Q1", "1992Q4"
        eval_start_q, eval_end_q = "1993Q1", "2023Q4"

    print(f"\n=== {i} | {x_name} ===")
    print("SSM window:", start_est_q, "->", end_est_q, "| eval:", eval_start_q, "->", eval_end_q)

    # -----------------------
    # 1) SSM
    # -----------------------
    params = fit_ssm_ml(final_data, names)
    G, Q, a, Z, R = build_ssm_two_series_ar1_ml(params)

    df_forecasts = recursive_forecast_exercise(
        gdp_q_col=final_data[[gdp_name]],
        x_m_col=final_data[[x_name]],
        start_est_q=start_est_q,
        end_est_q=end_est_q,
        names=names,
        eval_start_q=eval_start_q,
        eval_end_q=eval_end_q,
        horizons=range(1, 9),
        gdp_lags=1,
        x_month_lags=6
    )

    df_eval = df_forecasts.dropna(subset=["ssm_hat", "gdp_real"]).copy()
    df_eval["sq_err"] = (df_eval["ssm_hat"] - df_eval["gdp_real"]) ** 2
    rmse_ssm_by_h = (
        df_eval.groupby("h")
        .agg(
            N_h=("sq_err", "size"),
            RMSE_SSM=("sq_err", lambda s: float(np.sqrt(s.sum() / len(s))))
        )
        .reset_index()
        .sort_values("h")
    )

    df_forecasts.to_excel(OUTDIR_FORECAST / f"forecast_ssm_{i}_{x_name}.xlsx", index=False)
    rmse_ssm_by_h.to_excel(OUTDIR_RMSE_KALMAN / f"rmse_ssm_{i}_{x_name}.xlsx", index=False)

    # -----------------------
    # 2) MIDAS
    # -----------------------
    # aligner les fenêtres MIDAS sur celles du SSM
    df_midas = table7_rmse_grid(
        y_q=y_q,
        X=stat_xl[[x_name]],             # mensuel stationnaire
        horizons=range(1, 9),
        regressors=[x_name],
        Kx_LF=6,
        m=3,
        j_obs=2,
        train_end0=end_est_q,
        eval_start=eval_start_q,
        eval_end=eval_end_q,
        warm_start=True,
        maxiter=3000,
    ).reset_index()

    df_midas.to_csv(OUTDIR_RMSE_MIDAS / f"rmse_midas_{i}_{x_name}.csv", index=False)

print("\nDone. Results in:", OUTDIR_RMSE_MIDAS)