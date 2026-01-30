import numpy as np
import pandas as pd

from src.ssm.params import OneFactorParams, TwoFactorParams
from src.ssm.likelihood import fit_kalman_mle
from src.ssm.periodic_kf import (periodic_steady_state_kf, run_periodic_kf_filter)
from src.evaluation.data_management import quarter_end_months

def forecast_y_from_state(p: OneFactorParams, state_at_t: np.ndarray, h: int) -> float:
    """
    Stock variable cas: y_{t+h} = y*_{t+h} = lam_y f_{t+h} + u_y,t+h.

    """
    rho_pow = p.rho ** (p.m * h)
    d_pow = p.d ** (p.m * h)
    f_t = state_at_t[0]
    uy_t = state_at_t[1]
    return p.lam_y * rho_pow * f_t + d_pow * uy_t

def kalman_filter_forecast(y, x, h=1, m=3):
    '''
    Calcule des prev à horizon h  via le filtre de Kalman
    y: (T_BF,) variable basse frequence
    x: (T_HF, n_x) avec T_HF = T_BF*m variables haute frequence

    '''
    # Estimation des parametres
    p_hat = fit_kalman_mle(y, x, m=m)
    # Filtre de Kalman
    kf = periodic_steady_state_kf(p_hat)

    _, states_low = run_periodic_kf_filter(kf, y, x)


    forecasts = []
    actuals = []
    # Prevision recursive
    for t in range(len(y) - h):
        forecasts.append(forecast_y_from_state(p_hat, states_low[t], h))
        actuals.append(y[t + h])

    return np.array(forecasts), np.array(actuals)

def forecast_gdp_quarter_ssm(kf_out, params, origin_month, target_q):
    """
    Prévision du GDP trimestriel target_q à partir du filtre Kalman
    """
    idx = pd.DatetimeIndex(kf_out["index"])
    if origin_month not in idx:
        return np.nan

    t0 = int(np.where(idx == origin_month)[0][0])
    a0 = kf_out["a_filt"][t0].copy()
    G = kf_out["G"]
    g1 = params["gamma1"]

    # dernier mois du trimestre target_q
    m3 = quarter_end_months(target_q)
    if m3 not in idx:
        step = (m3.to_period("M") - origin_month.to_period("M")).n
    else:
        step = int(np.where(idx == m3)[0][0]) - t0

    if step <= 0:
        return np.nan

    # projection d'état jusqu'à m3
    a = a0
    for _ in range(step):
        a = G @ a

    f_m3, u1_m3 = a[0], a[1]
    gdp_hat = g1 * f_m3 + u1_m3
    return float(gdp_hat)

def forecast_y_from_state_2f(p: TwoFactorParams, state, h):
    f1, f2, uy, _ = state
    return (p.rho1**(p.m*h))*f1 + (p.rho2**(p.m*h))*f2 + (p.d**(p.m*h))*uy

