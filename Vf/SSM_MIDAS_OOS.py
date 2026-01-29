from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from utils import * 
from typing import Dict, Tuple, List, Optional
from scipy.optimize import minimize

# -----------------------------
# variables globales
# -----------------------------
K_BAR = 40              # lag max pour les comparaisons MIDAS-KF
RICCATI_MAX_ITERS = 50_000
RICCATI_TOL = 1e-9
KF_WARMUP_PERIODS = 100    
OPT_TOL = 1e-10

# Tables 1 et 2: grilles de paramètres
D_GRID = [-0.9, -0.5, 0.0, 0.5, 0.95]
RHO_GRID = [-0.9, -0.5, 0.0, 0.5, 0.95]


# -----------------------------
# Def SSM
# -----------------------------
@dataclass(frozen=True)
class OneFactorParams:
    m: int
    rho: float        # facteur latent AR(1)
    d: float          # mesure d'erreur AR(1) 
    lam_y: float      # coeff pour y
    lam_x: np.ndarray # coeff pour tous les x (taille = (n_x,))
    sig2_f: float     # Var(eps_f)
    sig2_uy: float    # Var(eps_u_y)
    sig2_ux: np.ndarray # Var(eps_u_xi) (taille = (n_x,))

    @property
    def n_x(self) -> int:
        return int(self.lam_x.shape[0])

    @property
    def dim_state(self) -> int:
        # état = [f, u_y, u_x1, ..., u_xn]
        return 2 + self.n_x


@dataclass(frozen=True)
class TwoFactorDGPParams:
    m: int
    rho: float
    d: float
    a: np.ndarray      # taille (2,)
    b: np.ndarray      # taille (n_x, 2)
    sig2_f1: float = 1.0
    sig2_f2: float = 1.0
    sig2_uy: float = 1.0
    sig2_ux: Optional[np.ndarray] = None  # taille (n_x,)

    def __post_init__(self):
        if self.sig2_ux is None:
            object.__setattr__(self, "sig2_ux", np.ones(self.b.shape[0], dtype=float))

    @property
    def n_x(self) -> int:
        return int(self.b.shape[0])


# -----------------------------
# SSM-Filtre Kalman (Riccati)
# -----------------------------
@dataclass
class PeriodicKF:
    params: OneFactorParams
    P_pred: List[np.ndarray]  # P_{j|j-1}
    K_gain: List[np.ndarray]  # K_{j|j-1}
    Z_list: List[np.ndarray]  # Z_j
    H_list: List[np.ndarray]  # var_cov des bruits pour j

    def G(self) -> np.ndarray:
        p = self.params
        # diag(rho, d, d, ..., d)
        diag = np.array([p.rho] + [p.d] * (1 + p.n_x), dtype=float)
        return np.diag(diag)

    def Q(self) -> np.ndarray:
        p = self.params
        diag = np.array([p.sig2_f, p.sig2_uy] + list(p.sig2_ux), dtype=float)
        return np.diag(diag)


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


# -----------------------------
# MIDAS specifications (regular and multiplicative)
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
# Objectives for Tables 1–2: L2 distance between KF and MIDAS weights (Eq. 3.8)
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


# -----------------------------
# construction des tableaux
# -----------------------------
def table1() -> Dict[str, pd.DataFrame]:

    out: Dict[str, pd.DataFrame] = {}

    for m in [3, 13]:
        for h in [1, 4]:
            reg_vals = np.zeros((len(D_GRID), len(RHO_GRID)))
            mul_vals = np.zeros((len(D_GRID), len(RHO_GRID)))

            for i_d, d in enumerate(D_GRID):
                for i_r, rho in enumerate(RHO_GRID):
                    # modèle à un facteur
                    p = OneFactorParams(
                        m=m, rho=rho, d=d,
                        lam_y=1.0, lam_x=np.array([1.0]),
                        sig2_f=1.0, sig2_uy=1.0, sig2_ux=np.array([1.0])
                    )

                    wy_kf, wx_kf = kalman_weights_by_impulses(p, h=h, K_bar=K_BAR)

                    reg_vals[i_d, i_r] = fit_regular_midas_to_kf_by_l2(wy_kf, wx_kf, K_bar=K_BAR, m=m)
                    mul_vals[i_d, i_r] = fit_multiplicative_midas_to_kf_by_l2(wy_kf, wx_kf, K_bar=K_BAR, m=m)

            df_reg = pd.DataFrame(reg_vals, index=D_GRID, columns=RHO_GRID)
            df_mul = pd.DataFrame(mul_vals, index=D_GRID, columns=RHO_GRID)
            out[f"Table1_m={m}_h={h}_regular"] = df_reg
            out[f"Table1_m={m}_h={h}_multiplicative"] = df_mul

    return out


def table2() -> Dict[str, pd.DataFrame]:

    out: Dict[str, pd.DataFrame] = {}
    h = 1

    for m in [3, 13]:
        for unequal in [False, True]:
            reg_vals = np.zeros((len(D_GRID), len(RHO_GRID)))
            mul_vals = np.zeros((len(D_GRID), len(RHO_GRID)))

            for i_d, d in enumerate(D_GRID):
                for i_r, rho in enumerate(RHO_GRID):
                    sig2_ux = np.array([1.0, 1.0]) if not unequal else np.array([1.0, 10.0])  # variance égale ou inégale

                    p = OneFactorParams(
                        m=m, rho=rho, d=d,
                        lam_y=1.0, lam_x=np.array([1.0, 1.0]),
                        sig2_f=1.0, sig2_uy=1.0, sig2_ux=sig2_ux
                    )

                    wy_kf, wx_kf_all = kalman_weights_by_impulses_multi_x(p, h=h, K_bar=K_BAR)
                    # Pour MIDAS à deux x, nous devons ajuster les deux vecteurs de poids x séparément.
                    # On minimise la somme des distances L2 pour les deux x.

                    def fit_regular_two_x() -> float:
                        def obj(u: np.ndarray) -> float:
                            th_y = (u[0], u[1])
                            th_x1 = (u[2], u[3])
                            th_x2 = (u[4], u[5])

                            sy = exp_almon_weights(K_BAR, th_y[0], th_y[1])
                            sx1 = exp_almon_weights(m * K_BAR, th_x1[0], th_x1[1])
                            sx2 = exp_almon_weights(m * K_BAR, th_x2[0], th_x2[1])

                            by = float(np.dot(wy_kf, sy) / max(1e-15, np.dot(sy, sy)))
                            b1 = float(np.dot(wx_kf_all[0], sx1) / max(1e-15, np.dot(sx1, sx1)))
                            b2 = float(np.dot(wx_kf_all[1], sx2) / max(1e-15, np.dot(sx2, sx2)))

                            wy_m = by * sy
                            wx1_m = b1 * sx1
                            wx2_m = b2 * sx2

                            return float(np.sum((wy_kf - wy_m) ** 2) +
                                         np.sum((wx_kf_all[0] - wx1_m) ** 2) +
                                         np.sum((wx_kf_all[1] - wx2_m) ** 2))

                        x0 = np.array([-0.2, 0.0, -0.2, 0.0, -0.2, 0.0])
                        res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": OPT_TOL, "fatol": OPT_TOL, "maxiter": 80_000})
                        return float(res.fun)

                    def fit_multiplicative_two_x() -> float:
                        def obj(u: np.ndarray) -> float:
                            th_y = (u[0], u[1])

                            th_o1 = (u[2], u[3])
                            th_i1 = (u[4], u[5])

                            th_o2 = (u[6], u[7])
                            th_i2 = (u[8], u[9])

                            sy = exp_almon_weights(K_BAR, th_y[0], th_y[1])

                            # construction des formes sx pour les deux x
                            def sx_shape(th_o, th_i):
                                w_outer = exp_almon_weights(K_BAR, th_o[0], th_o[1])
                                w_inner = exp_almon_weights(m - 1, th_i[0], th_i[1])
                                sx = np.zeros(m * K_BAR + 1)
                                for jj in range(K_BAR + 1):
                                    for r in range(m):
                                        k = jj * m + r
                                        if k <= m * K_BAR:
                                            sx[k] += w_outer[jj] * w_inner[r]
                                return sx

                            sx1 = sx_shape(th_o1, th_i1)
                            sx2 = sx_shape(th_o2, th_i2)

                            by = float(np.dot(wy_kf, sy) / max(1e-15, np.dot(sy, sy)))
                            b1 = float(np.dot(wx_kf_all[0], sx1) / max(1e-15, np.dot(sx1, sx1)))
                            b2 = float(np.dot(wx_kf_all[1], sx2) / max(1e-15, np.dot(sx2, sx2)))

                            wy_m = by * sy
                            wx1_m = b1 * sx1
                            wx2_m = b2 * sx2

                            return float(np.sum((wy_kf - wy_m) ** 2) +
                                         np.sum((wx_kf_all[0] - wx1_m) ** 2) +
                                         np.sum((wx_kf_all[1] - wx2_m) ** 2))

                        x0 = np.array([-0.2, 0.0,  -0.2, 0.0, -0.2, 0.0,  -0.2, 0.0, -0.2, 0.0])
                        res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": OPT_TOL, "fatol": OPT_TOL, "maxiter": 120_000})
                        return float(res.fun)

                    reg_vals[i_d, i_r] = fit_regular_two_x()
                    mul_vals[i_d, i_r] = fit_multiplicative_two_x()

            tag = "unequal" if unequal else "equal"
            out[f"Table2_m={m}_{tag}_regular"] = pd.DataFrame(reg_vals, index=D_GRID, columns=RHO_GRID)
            out[f"Table2_m={m}_{tag}_multiplicative"] = pd.DataFrame(mul_vals, index=D_GRID, columns=RHO_GRID)

    return out


def table3() -> Dict[str, pd.DataFrame]:

    out: Dict[str, pd.DataFrame] = {}

    # 2 facteurs latents
    a = np.array([0.9, 0.1])
    b = np.array([[0.1, 0.9]])  # un seul x

    for m in [3, 13]:
        print(f"m={m}")
        for h in [1, 4]:
            print(f"  h={h}")
            ratios = np.zeros((len(D_GRID), len(RHO_GRID)))
            for i_d, d in enumerate(D_GRID):
                for i_r, rho in enumerate(RHO_GRID):
                    # Construction de la matrice de covariance vraie Sigma sous le DGP à deux facteurs
                    # chaque facteur est AR(1) avec paramètre rho
                    var_f = 1.0 / (1.0 - rho * rho) if abs(rho) < 1 else 1e12

                    Vyy = var_f * float(a[0] ** 2 + a[1] ** 2)
                    Vxx = var_f * float(b[0, 0] ** 2 + b[0, 1] ** 2)
                    Vxy = var_f * float(a[0] * b[0, 0] + a[1] * b[0, 1])
                    V_factor = np.array([[Vyy, Vxy], [Vxy, Vxx]], dtype=float)

                    cov_yy, cov_xx, cov_xy = cov_one_or_two_factor_yx(
                        m=m, rho=rho, d=d,
                        V_factor=V_factor,
                        sig2_uy=1.0,
                        sig2_ux=1.0,
                    )
                    Sigma_true = sigma_matrix_for_upsilon(m=m, h=h, K_bar=K_BAR, cov_yy=cov_yy, cov_xx=cov_xx, cov_xy=cov_xy)

                    # SS1 meilleur PE variance sous vraie Sigma
                    pe_ss1 = ss1_best_pe_variance_under_true_sigma(
                        Sigma_true=Sigma_true, m=m, h=h, K_bar=K_BAR, d_fixed=d, rho_init=rho
                    )

                    # MIDAS régulier meilleur PE variance sous vraie Sigma
                    pe_midas, _, _ = fit_regular_midas_by_pe_variance(
                        Sigma=Sigma_true, m=m, h=h, K_bar=K_BAR
                    )

                    ratios[i_d, i_r] = pe_midas / pe_ss1 if pe_ss1 > 0 else np.nan

            out[f"Table3_m={m}_h={h}"] = pd.DataFrame(ratios, index=D_GRID, columns=RHO_GRID)

    return out


# -----------------------------
# Appel principal
# -----------------------------
def main():
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")

    print(f"Using K_BAR={K_BAR}\n")

    print("Computing Table 1...")
    t1 = table1()

    out_txt = "table1.txt"
    with open(out_txt, "w", encoding="utf-8") as f:

        # -------- TABLE 1 ----------
        for k, df in t1.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TABLE 1 - {k}\n")
            f.write("=" * 80 + "\n")
            f.write(df.to_string())
            f.write("\n")
    print("\nComputing Table 2...")
    t2 = table2()

    out_txt = "table2.txt"

    with open(out_txt, "w", encoding="utf-8") as f:

        # -------- TABLE 2 ----------
        for k, df in t2.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TABLE 2 - {k}\n")
            f.write("=" * 80 + "\n")
            f.write(df.to_string())
            f.write("\n")

    print("\nComputing Table 3...")
    t3 = table3()

    out_txt = "table3.txt"

    with open(out_txt, "w", encoding="utf-8") as f:

        # -------- TABLE 3 ----------
        for k, df in t3.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TABLE 3 - {k}\n")
            f.write("=" * 80 + "\n")
            f.write(df.to_string())
            f.write("\n")

    print(f"TXT écrit : {out_txt}")


if __name__ == "__main__":
    main()

