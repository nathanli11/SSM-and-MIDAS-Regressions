import numpy as np
from typing import Tuple

from src.ssm.params import OneFactorParams
from src.ssm.periodic_kf import periodic_steady_state_kf, run_periodic_kf_filter
from src.ssm.forecast import forecast_y_from_state

KF_WARMUP_PERIODS = 100 

# -----------------------------
# Exctraction des poids Kalman par impulsions
# -----------------------------

def kalman_weights_by_impulses(p: OneFactorParams, h: int, K_bar: int) -> Tuple[np.ndarray, np.ndarray]:
    kf = periodic_steady_state_kf(p)
    m = p.m
    n_x = p.n_x

    # longueur d'échantillon dans les périodes à basse fréquence
    T_low = KF_WARMUP_PERIODS + (K_bar + 5) + (h + 2)
    T_high = T_low * m

    # données de base (zéros)
    base_y = np.zeros(T_low)
    base_x = np.zeros((T_high, n_x))

    # exécution du filtre Kalman périodique sur les données de base
    _, base_states_low = run_periodic_kf_filter(kf, base_y, base_x)
    t0 = KF_WARMUP_PERIODS + K_bar + 1  # index du temps de référence pour les poids
    yhat0 = forecast_y_from_state(p, base_states_low[t0], h=h)

    # poids pour les y-lags à basse fréquence y_{t0 - j}
    w_y = np.zeros(K_bar + 1)
    for j in range(K_bar + 1):
        y = base_y.copy()
        # impulsion en y_{t0 - j}
        y[t0 - j] = 1.0
        _, states_low = run_periodic_kf_filter(kf, y, base_x)
        yhat = forecast_y_from_state(p, states_low[t0], h=h)
        w_y[j] = yhat - yhat0

    # poids pour les x-lags à haute fréquence x_{t0 - k/m}
    # (note: k=0 correspond à x_{t0}, k=1 à x_{t0 - 1/m}, ..., k=m*K_bar à x_{t0 - K_bar})
    end_high = t0 * m + (m - 1)
    w_x = np.zeros(m * K_bar + 1)

    for k in range(m * K_bar + 1):
        x = base_x.copy()
        # impulsion en x_{t0 - k/m}
        # correspond à l'index end_high - k en haute fréquence
        x[end_high - k, 0] = 1.0
        _, states_low = run_periodic_kf_filter(kf, base_y, x)
        yhat = forecast_y_from_state(p, states_low[t0], h=h)
        w_x[k] = yhat - yhat0

    return w_y, w_x


def kalman_weights_by_impulses_multi_x(p: OneFactorParams, h: int, K_bar: int) -> Tuple[np.ndarray, np.ndarray]:

    kf = periodic_steady_state_kf(p)
    m = p.m
    n_x = p.n_x

    T_low = KF_WARMUP_PERIODS + (K_bar + 5) + (h + 2)
    T_high = T_low * m

    base_y = np.zeros(T_low)
    base_x = np.zeros((T_high, n_x))

    _, base_states_low = run_periodic_kf_filter(kf, base_y, base_x)
    t0 = KF_WARMUP_PERIODS + K_bar + 1
    yhat0 = forecast_y_from_state(p, base_states_low[t0], h=h)

    w_y = np.zeros(K_bar + 1)
    for j in range(K_bar + 1):
        y = base_y.copy()
        y[t0 - j] = 1.0
        _, states_low = run_periodic_kf_filter(kf, y, base_x)
        w_y[j] = forecast_y_from_state(p, states_low[t0], h=h) - yhat0

    end_high = t0 * m + (m - 1)
    w_x = np.zeros((n_x, m * K_bar + 1))
    for i in range(n_x):
        for k in range(m * K_bar + 1):
            x = base_x.copy()
            x[end_high - k, i] = 1.0
            _, states_low = run_periodic_kf_filter(kf, base_y, x)
            w_x[i, k] = forecast_y_from_state(p, states_low[t0], h=h) - yhat0

    return w_y, w_x

