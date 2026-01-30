import numpy as np
from typing import Tuple
from scipy.optimize import minimize

from src.midas.midas_base import exp_almon_weights
from src.evaluation.model_selection import OPT_TOL

# -----------------------------
# Specifications MIDAS (regular et multiplicative)
# -----------------------------
def midas_regular_coeffs(K_bar: int, m: int, theta_y: Tuple[float, float], theta_x: Tuple[float, float],
                        beta_y: float, beta_x: float) -> Tuple[np.ndarray, np.ndarray]:

    wy = beta_y * exp_almon_weights(K_bar, theta_y[0], theta_y[1])
    wx = beta_x * exp_almon_weights(m * K_bar, theta_x[0], theta_x[1])
    return wy, wx


def midas_multiplicative_coeffs(K_bar: int, m: int,
                                theta_y: Tuple[float, float],
                                theta_outer_x: Tuple[float, float],
                                theta_inner_x: Tuple[float, float],
                                beta_y: float, beta_x: float) -> Tuple[np.ndarray, np.ndarray]:

    wy = beta_y * exp_almon_weights(K_bar, theta_y[0], theta_y[1])
    w_outer = exp_almon_weights(K_bar, theta_outer_x[0], theta_outer_x[1])
    w_inner = exp_almon_weights(m - 1, theta_inner_x[0], theta_inner_x[1])  

    wx = np.zeros(m * K_bar + 1)
    for j in range(K_bar + 1):
        for r in range(m):
            k = j * m + r
            if k <= m * K_bar:
                wx[k] += beta_x * w_outer[j] * w_inner[r]
    return wy, wx


# -----------------------------
# Objectives pour les Tables 1–2: L2 distance entre les poids KF et MIDAS (Eq. 3.8)
# -----------------------------
def l2_distance_weights(wy_kf: np.ndarray, wx_kf: np.ndarray,
                        wy_m: np.ndarray, wx_m: np.ndarray) -> float:
    return float(np.sum((wy_kf - wy_m) ** 2) + np.sum((wx_kf - wx_m) ** 2))


def fit_regular_midas_to_kf_by_l2(wy_kf: np.ndarray, wx_kf: np.ndarray, K_bar: int, m: int) -> float:

    wy_shape = lambda th: exp_almon_weights(K_bar, th[0], th[1])
    wx_shape = lambda th: exp_almon_weights(m * K_bar, th[0], th[1])

    def obj(u: np.ndarray) -> float:
        th_y = (u[0], u[1])
        th_x = (u[2], u[3])
        sy = wy_shape(th_y)
        sx = wx_shape(th_x)

        # betas optimaux
        by = float(np.dot(wy_kf, sy) / max(1e-15, np.dot(sy, sy)))
        bx = float(np.dot(wx_kf, sx) / max(1e-15, np.dot(sx, sx)))

        wy_m, wx_m = midas_regular_coeffs(K_bar, m, th_y, th_x, by, bx)
        return l2_distance_weights(wy_kf, wx_kf, wy_m, wx_m)

    x0 = np.array([-0.2, 0.0, -0.2, 0.0])
    res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": OPT_TOL, "fatol": OPT_TOL, "maxiter": 50_000})
    return float(res.fun)


def fit_multiplicative_midas_to_kf_by_l2(wy_kf: np.ndarray, wx_kf: np.ndarray, K_bar: int, m: int) -> float:

    wy_shape = lambda th: exp_almon_weights(K_bar, th[0], th[1])

    def wx_shape(th_outer: Tuple[float, float], th_inner: Tuple[float, float]) -> np.ndarray:
        w_outer = exp_almon_weights(K_bar, th_outer[0], th_outer[1])
        w_inner = exp_almon_weights(m - 1, th_inner[0], th_inner[1])
        sx = np.zeros(m * K_bar + 1)
        for j in range(K_bar + 1):
            for r in range(m):
                k = j * m + r
                if k <= m * K_bar:
                    sx[k] += w_outer[j] * w_inner[r]
        return sx

    def obj(u: np.ndarray) -> float:
        th_y = (u[0], u[1])
        th_ox = (u[2], u[3])
        th_ix = (u[4], u[5])

        sy = wy_shape(th_y)
        sx = wx_shape(th_ox, th_ix)

        by = float(np.dot(wy_kf, sy) / max(1e-15, np.dot(sy, sy)))
        bx = float(np.dot(wx_kf, sx) / max(1e-15, np.dot(sx, sx)))

        wy_m, wx_m = midas_multiplicative_coeffs(K_bar, m, th_y, th_ox, th_ix, by, bx)
        return l2_distance_weights(wy_kf, wx_kf, wy_m, wx_m)

    x0 = np.array([-0.2, 0.0, -0.2, 0.0, -0.2, 0.0])
    res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": OPT_TOL, "fatol": OPT_TOL, "maxiter": 80_000})
    return float(res.fun)

