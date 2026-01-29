import numpy as np
from params import OneFactorParams
from ORGA_TEST.src.ssm.likelihood import fit_kalman_mle
from ORGA_TEST.src.ssm.periodic_kf import (periodic_steady_state_kf, run_periodic_k_filter)

def forecast_y_from_state(p: OneFactorParams, state_at_t: np.ndarray, h: int) -> float:
    """
    Stock variable case: y_{t+h} = y*_{t+h} = lam_y f_{t+h} + u_y,t+h.
    Under AR(1) transitions: f_{t+h} = rho^(m*h) f_t in expectation, u_y similarly.
    """
    rho_pow = p.rho ** (p.m * h)
    d_pow = p.d ** (p.m * h)
    f_t = state_at_t[0]
    uy_t = state_at_t[1]
    return p.lam_y * rho_pow * f_t + d_pow * uy_t

def kalman_filter_forecast(y, x, h=1, m=3):
    p_hat = fit_kalman_mle(y, x, m=m)
    kf = periodic_steady_state_kf(p_hat)

    _, states_low = run_periodic_kf_filter(kf, y, x)

    forecasts = []
    actuals = []
    for t in range(len(y) - h):
        forecasts.append(forecast_y_from_state(p_hat, states_low[t], h))
        actuals.append(y[t + h])

    return np.array(forecasts), np.array(actuals)


