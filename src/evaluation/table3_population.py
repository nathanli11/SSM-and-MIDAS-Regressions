import numpy as np
from typing import Tuple
import math
from scipy.optimize import minimize

from src.midas.midas_base import exp_almon_weights
from src.ssm.kalman_weights import kalman_weights_by_impulses
from src.ssm.params import OneFactorParams
from model_selection import OPT_TOL

# -----------------------------
# fonctions de covariance pour Table 3
# -----------------------------
def cov_one_or_two_factor_yx(
    m: int,
    rho: float,
    d: float,
    # pour un facteur
    V_factor: np.ndarray,
    sig2_uy: float,
    sig2_ux: float,
) -> Tuple[callable, callable, callable]:

    # Structure de covariance pour y et x_i (i=1,...,n_x) dans le modèle à un facteur:

    var_uy = sig2_uy / (1.0 - d * d) if abs(d) < 1 else 1e12
    var_ux = sig2_ux / (1.0 - d * d) if abs(d) < 1 else 1e12

    def cov_yy(i: int, j: int) -> float:
        L = abs(i - j)
        return float((rho ** L) * V_factor[0, 0] + (d ** L) * var_uy)

    def cov_xx(i: int, j: int) -> float:
        L = abs(i - j)
        return float((rho ** L) * V_factor[1, 1] + (d ** L) * var_ux)

    def cov_xy(i: int, j: int) -> float:
        L = abs(i - j)
        return float((rho ** L) * V_factor[1, 0]) 

    return cov_yy, cov_xx, cov_xy


def sigma_matrix_for_upsilon(m: int, h: int, K_bar: int,
                            cov_yy, cov_xx, cov_xy) -> np.ndarray:

    steps = list(range(-m * h, m * K_bar + 1))  # entiers de -m*h à m*K_bar
    dim = 2 * len(steps)
    S = np.zeros((dim, dim))
    for a, i in enumerate(steps):
        for b, j in enumerate(steps):
            # positions
            ax = 2 * a
            ay = 2 * a + 1
            bx = 2 * b
            by = 2 * b + 1

            S[ax, bx] = cov_xx(i, j)
            S[ay, by] = cov_yy(i, j)
            S[ax, by] = cov_xy(i, j)  # x,y
            S[ay, bx] = cov_xy(j, i)  # y,x
    return S


def w_vector_from_coeffs(m: int, h: int, K_bar: int, wy: np.ndarray, wx: np.ndarray) -> np.ndarray:
    steps = list(range(-m * h, m * K_bar + 1))
    dim = 2 * len(steps)
    w = np.zeros(dim)


    w[1] = 1.0

    # soustraire les coefficients x pour x_{t - k/m}: cela correspond à step = k
    for k in range(m * K_bar + 1):
        #  trouver l'index correspondant à step = k
        idx = steps.index(k)
        w[2 * idx] -= wx[k]

    # soustraire les coefficients y pour y_{t - j}: cela correspond à step = -m * j
    for j in range(K_bar + 1):
        idx = steps.index(m * j)
        w[2 * idx + 1] -= wy[j]

    return w


# -----------------------------
# Table 3: choisir les paramètres MIDAS en minimisant la variance PE (Eq. 3.6)
# -----------------------------
def fit_regular_midas_by_pe_variance(Sigma: np.ndarray, m: int, h: int, K_bar: int) -> Tuple[float, np.ndarray, np.ndarray]:

    steps = list(range(-m * h, m * K_bar + 1))

    def build_w_from_params(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        th_y = (u[0], u[1])
        th_x = (u[2], u[3])
        sy = exp_almon_weights(K_bar, th_y[0], th_y[1])
        sx = exp_almon_weights(m * K_bar, th_x[0], th_x[1])

        w0 = np.zeros(2 * len(steps))
        w0[1] = 1.0

        # vecteurs de base pour wy et wx
        wy_basis = w_vector_from_coeffs(m, h, K_bar, wy=sy, wx=np.zeros(m * K_bar + 1)) - w0
        wx_basis = w_vector_from_coeffs(m, h, K_bar, wy=np.zeros(K_bar + 1), wx=sx) - w0

        # w(beta) = w0 + beta_y * wy_basis + beta_x * wx_basis
        return w0, wy_basis, wx_basis

    def obj(u: np.ndarray) -> float:
        w0, wyb, wxb = build_w_from_params(u)
        # minimisation quadratique en beta
        B = np.column_stack([wyb, wxb])  # dim x 2
        A = B.T @ Sigma @ B
        c = B.T @ Sigma @ w0
        # fonction objectif = w0'Σw0 + 2 b'c + b'A b
        try:
            b = -np.linalg.solve(A, c)
        except np.linalg.LinAlgError:
            return 1e50
        w = w0 + B @ b
        return float(w.T @ Sigma @ w)

    x0 = np.array([-0.2, 0.0, -0.2, 0.0])
    res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": OPT_TOL, "fatol": OPT_TOL, "maxiter": 80_000})

    # reconstitution des poids MIDAS optimaux
    w0, wyb, wxb = build_w_from_params(res.x)
    B = np.column_stack([wyb, wxb])
    A = B.T @ Sigma @ B
    c = B.T @ Sigma @ w0
    b = -np.linalg.solve(A, c)

    th_y = (res.x[0], res.x[1])
    th_x = (res.x[2], res.x[3])
    sy = exp_almon_weights(K_bar, th_y[0], th_y[1])
    sx = exp_almon_weights(m * K_bar, th_x[0], th_x[1])
    beta_y, beta_x = float(b[0]), float(b[1])
    wy = beta_y * sy
    wx = beta_x * sx

    w = w_vector_from_coeffs(m, h, K_bar, wy, wx)
    pe_var = float(w.T @ Sigma @ w)
    return pe_var, wy, wx


# -----------------------------
# Table 3: choix des paramètres SS1 en minimisant la variance PE
# -----------------------------
def ss1_best_pe_variance_under_true_sigma(
    Sigma_true: np.ndarray,
    m: int,
    h: int,
    K_bar: int,
    d_fixed: float,
    rho_init: float,
) -> float:

    def unpack(u: np.ndarray) -> OneFactorParams:
        rho = np.tanh(u[0]) 
        lam_y = u[1]
        lam_x = np.array([u[2]])
        sig2_f = math.exp(u[3])
        sig2_uy = math.exp(u[4])
        sig2_ux = np.array([math.exp(u[5])])
        return OneFactorParams(
            m=m, rho=rho, d=d_fixed, lam_y=lam_y, lam_x=lam_x,
            sig2_f=sig2_f, sig2_uy=sig2_uy, sig2_ux=sig2_ux
        )

    def obj(u: np.ndarray) -> float:
        p = unpack(u)
        # récuopération des poids Kalman
        try:
            wy_kf, wx_kf = kalman_weights_by_impulses(p, h=h, K_bar=K_bar)
        except Exception:
            return 1e50
        w = w_vector_from_coeffs(m, h, K_bar, wy_kf, wx_kf)
        return float(w.T @ Sigma_true @ w)

    # initialisation
    x0 = np.array([np.arctanh(max(-0.99, min(0.99, rho_init))), 1.0, 1.0, 0.0, 0.0, 0.0])
    res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 5_000})
    return float(res.fun)
