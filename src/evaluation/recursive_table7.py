from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

from midas.adl_midas import ADLRegularMIDAS, ADLMultiplicativeMIDAS
from src.evaluation.data_management import (
    quarter_end_months, quarter_second_month, normalize_full_sample)
from src.ssm.kalman_two_series import kalman_filter_states, fit_ssm_ml
from src.ssm.forecast import forecast_gdp_quarter_ssm

def recursive_rmse(
    y_q: pd.Series,
    x_m: pd.Series,
    h: int,
    model: str,
    Kx_LF: int = 6,
    m: int = 3,
    j_obs: int = 2,
    train_end0: str = "1978Q4",
    eval_start: str = "1979Q1",
    eval_end: str = "2009Q1",
    warm_start: bool = True,
    maxiter: int = 3000,
) -> Tuple[float, int]:
    """Calcule le RMSE en fenêtre croissante (expanding window)
    pour un seul régresseur et un horizon donné.

    Paramétrage du papier (Tableau 7) :
      - l'échantillon initial d'estimation se termine en 1978Q4
      - période d'évaluation : 1979Q1-2009Q1
      - horizons h = 1..8
      - ensemble d'information : données mensuelles disponibles
        jusqu'au 2e mois du trimestre (j_obs = 2)
    """
    y_q = y_q.copy()
    if not isinstance(y_q.index, pd.PeriodIndex):
        y_q.index = pd.PeriodIndex(pd.to_datetime(y_q.index), freq="Q")
    y_q = y_q.sort_index().asfreq("Q")

    train_end0 = pd.Period(train_end0, freq="Q")
    eval_start = pd.Period(eval_start, freq="Q")
    eval_end = pd.Period(eval_end, freq="Q")

    if model == "regular":
        M = ADLRegularMIDAS(Kx_LF=Kx_LF, m=m, j_obs=j_obs, include_intercept=True)
        theta_init = (0.0, 0.0)
    elif model == "multiplicative":
        M = ADLMultiplicativeMIDAS(Kx_LF=Kx_LF, m=m, j_obs=j_obs, include_intercept=True)
        theta_init = (0.0, 0.0, 0.0, 0.0)
    else:
        raise ValueError("model must be 'regular' or 'multiplicative'")

    preds, actual = [], []

    for target in pd.period_range(eval_start, eval_end, freq="Q"):
        origin = target - int(h)
        if origin < train_end0:
            continue
        if target not in y_q.index or pd.isna(y_q.loc[target]):
            continue

        # fenêtre croissante : estimation avec les données disponibles jusqu'à origin
        # (y jusqu'à origin ; x est géré via j_obs à l'intérieur du modèle)
        if model == "regular":
            fit = M.fit(
                y_q=y_q[y_q.index <= origin],
                x_m=x_m,
                h=h,
                origin=origin,
                theta_init=theta_init,
                maxiter=maxiter,
            )
            yhat = M.predict_one(y_q=y_q, x_m=x_m, origin=origin)
            if warm_start:
                theta_init = (float(fit.theta[0]), float(fit.theta[1]))
        else:
            fit = M.fit(
                y_q=y_q[y_q.index <= origin],
                x_m=x_m,
                h=h,
                origin=origin,
                theta_out_init=(theta_init[0], theta_init[1]),
                theta_in_init=(theta_init[2], theta_init[3]),
            )
            yhat = M.predict_one(y_q=y_q, x_m=x_m, origin=origin)
            if warm_start:
                theta_init = (
                    float(fit.theta[0]),
                    float(fit.theta[1]),
                    float(fit.theta[2]),
                    float(fit.theta[3]),
                )

        preds.append(float(yhat))
        actual.append(float(y_q.loc[target]))

    preds = np.asarray(preds, float)
    actual = np.asarray(actual, float)
    rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
    return rmse, int(len(preds))


def table7_rmse_grid(
    y_q: pd.Series,
    X: pd.DataFrame,
    horizons=range(1, 9),
    regressors=None,
    **kwargs,
) -> pd.DataFrame:
    """Retourne un DataFrame avec MultiIndex (regressor, h) et RMSE en colonne"""
    if regressors is None:
        regressors = list(X.columns)

    rows = []
    for name in regressors:
        x_m = X[name].dropna()
        for h in horizons:
            rm1, n1 = recursive_rmse(y_q, x_m, h=h, model="regular", **kwargs)
            rm2, n2 = recursive_rmse(y_q, x_m, h=h, model="multiplicative", **kwargs)
            rows.append((name, h, rm1, rm2, n1))

    out = pd.DataFrame(rows, columns=["regressor", "h", "rmse_regular", "rmse_multiplicative", "n"])
    return out.set_index(["regressor", "h"]).sort_index()


# Previsions récursives SSM avec Kalman
def recursive_forecast_exercise(gdp_q_col : pd.DataFrame,
                                x_m_col : pd.DataFrame,
                                start_est_q: str,
                                end_est_q: str,
                                names,
                                eval_start_q: str,
                                eval_end_q: str,
                                horizons=range(1,9),
                                gdp_lags=1,
                                x_month_lags=6):
    """
    - Normalise (full sample) GDP et x.
    - Estime récursivement:
       * SSM (ML via Kalman)
    - Prévisions faites avec données mensuelles jusqu'au 2e mois du trimestre à prévoir.
    """

    # --- normalisation full sample (comme le papier) ---
    #gdp_q, gdp_mu, gdp_sd = normalize_full_sample(gdp_q_col.astype(float))
    #x_m, x_mu, x_sd = normalize_full_sample(x_m_col.astype(float))
    gdp_norm, gdp_mu, gdp_sd = normalize_full_sample(gdp_q_col.astype(float))
    x_norm, x_mu, x_sd = normalize_full_sample(x_m_col.astype(float))

    # --- indices trimestriels ---
    gdp_q = gdp_norm.dropna()
    gdp_q.index = pd.PeriodIndex(gdp_q.index, freq="Q")
    gdp_q = gdp_q.groupby(level=0).last().sort_index()


    start_est = pd.Period(start_est_q, freq="Q")
    end_est   = pd.Period(end_est_q,   freq="Q")
    eval_start= pd.Period(eval_start_q,freq="Q")
    eval_end  = pd.Period(eval_end_q,  freq="Q")

    # trimestres évalués
    qs_all = gdp_q.index
    eval_qs = [q for q in qs_all if (q >= eval_start and q <= eval_end)]

    # stockage
    out = []
    # pré-calendrier mensuel (index mensuel fin de mois)
    x_m = x_norm.astype(float).copy()
    x_m.index = pd.DatetimeIndex(x_m.index).to_period("M").to_timestamp("M")
    x_m = x_m.sort_index()

    for origin_q in eval_qs:
        print("ORIGIN:", origin_q)

        # fenêtre d'estimation expanding: de start_est à origin_q-1
        est_end_q = origin_q - 1
        if est_end_q < start_est:
            continue
        gdp_est = gdp_q.loc[start_est:est_end_q].dropna()

        # construire x disponible pour l'estimation:
        # on suppose qu'à chaque trimestre, on n'utilise x que jusqu'au 2e mois
        # => pour l'estimation, il suffit d'avoir x sur tout l'historique mensuel.
        x_est = x_m.copy()


        # --------- (B) SSM estimation (ML) ----------
        # construire df_hf mensuel avec gdp observé aux fins de trimestre de la fenêtre d'estimation
        # et NaN ailleurs. Le GDP du trimestre est placé au dernier mois du trimestre.
        # IMPORTANT: on n'inclut pas d'observation GDP pour le trimestre origin_q (à prévoir).
        start_m = quarter_end_months(start_est).to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(2)  # approx
        end_m   = quarter_second_month(origin_q)  # info set: jusqu'au 2e mois du trimestre origin_q (comme le papier)
        months = pd.date_range(start=start_m, end=end_m, freq="M")

        df_hf = pd.DataFrame(index=months, columns=names, dtype=float)
        df_hf[names[1]] = x_m.reindex(months).values

        # GDP observé seulement pour trimestres <= est_end_q, au dernier mois du trimestre
        for q in gdp_est.index:
            m3 = quarter_end_months(q)
            if m3 in df_hf.index:
                df_hf.loc[m3, names[0]] = gdp_est.loc[q]

        # (il y aura des NaN si x manque : on laisse, le Kalman gère)

        # Estimation ML
        params = fit_ssm_ml(df_hf, names)

        # Filtre jusqu'à end_m (2e mois du trimestre origin_q)
        kf_out = kalman_filter_states(df_hf, params, names)

        # --------- Forecasts h = 1..8 ----------
        for h in horizons:
            #target_q = origin_q + (h - 1)  # h=1 => trimestre origin_q
            target_q = origin_q + h #h=0 => trimestre origin_q

            # SSM: on prédit GDP du trimestre target_q à info jusqu'au 2e mois de origin_q
            # (pour h>1, origin_q est plus tôt, mais dans le papier ils font 2..8 quarters ahead pareil
            #  => tu peux remplacer origin_month par 2e mois de origin_q pour "as-of" constant)
            origin_month = quarter_second_month(origin_q)
            yhat_ssm = forecast_gdp_quarter_ssm(
                kf_out, params, origin_month=origin_month, target_q=target_q
            )

            print(origin_q, target_q)
            
            out.append({
                "origin_q": str(origin_q),
                "target_q": str(target_q),
                "h": h,
                "ssm_hat": yhat_ssm,
                "gdp_real": gdp_q.loc[target_q] if target_q in gdp_q.index else np.nan
            })

    return pd.DataFrame(out)
