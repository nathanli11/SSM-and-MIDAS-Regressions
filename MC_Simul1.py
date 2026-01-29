import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple
from table1_2_3 import (
    build_measurement_mats,
    periodic_steady_state_kf,
    run_periodic_kf_filter,
    forecast_y_from_state,
    exp_almon_weights
)

@dataclass
class OneFactorParams:
    m: int = 3
    n_x: int = 1

    # Loadings (gamma1=gamma2=1)
    lam_y: float = 1.0
    lam_x: np.ndarray = None

    # AR params
    rho:float = 0.5
    d: float = 0.0

    # variances of innovations
    sig2_f: float = 1.0
    sig2_uy: float = 1.0
    sig2_ux: np.ndarray = None  # shape (n_x,)

    @property
    def dim_state(self) -> int:
        # state = [f, u_y, u_x1,...,u_xn]
        return 2 + self.n_x


@dataclass
class PeriodicKF:
    params: OneFactorParams
    P_pred: List[np.ndarray]
    K_gain: List[np.ndarray]
    Z_list: List[np.ndarray]
    H_list: List[np.ndarray]

    def G(self) -> np.ndarray:
        p = self.params
        return np.diag([p.rho] + [p.d] * (1 + p.n_x))

    def Q(self) -> np.ndarray:
        p = self.params
        return np.diag([p.sig2_f, p.sig2_uy] + list(p.sig2_ux))


def simulate_one_factor_dgp(T=40, m=3, rho=0.9, d=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # high frequency length
    Th = T*m

    # Innovations
    eta = np.random.normal(size=Th)
    eps1 = np.random.normal(size=Th)
    eps2 = np.random.normal(size=Th)

    # Facteur Latent
    f = np.zeros(Th)
    for t in range(1, Th):
        f[t] = rho * f[t-1] + eta[t]
    
    # Measurement errors
    u1 = np.zeros(Th)
    u2 = np.zeros(Th)
    for t in range(1, Th):
        u1[t] = d * u1[t-1] + eps1[t]
        u2[t] = d * u2[t-1] + eps2[t]
    
    # Observations
    y_star = f + u1
    x = f + u2

    # Low frequency aggregation (stock variable)
    y = y_star[m-1::m]

    return y, x.reshape(-1,1), f

def midas_regular_forecast(y, x, h=1, m=3, K=12):
    T = len(y)
    X = []
    Y = []

    for t in range(K, T-h):
        row = [1, y[t]]
        for j in range(K):
            row.append(x[t*m - j - 1, 0])
        X.append(row)
        Y.append(y[t + h])
    
    X = np.array(X)
    Y = np.array(Y)

    beta = lstsq(X, Y, rcond=None)[0]

    forecasts = X @ beta
    actuals = Y

    return forecasts, actuals

def midas_aggregate_x(
    x: np.ndarray,
    t: int,
    m: int,
    theta1: float,
    theta2: float
) -> float:
    """
    MIDAS aggregation of high-frequency x at low-frequency time t.
    x: shape (T_high, 1)
    """
    w = exp_almon_weights(m - 1, theta1, theta2)
    return x[t*m-1, 0] #end of period observation

def midas_x_term(
    x: np.ndarray,
    t: int,
    m: int,
    Kx: int,
    theta_x1: float,
    theta_x2: float
) -> float:
    w_x = exp_almon_weights(Kx, theta_x1, theta_x2)

    return sum(
        w_x[j] * midas_aggregate_x(x, t - j, m, theta_x1, theta_x2)
        for j in range(Kx + 1)
    )

def midas_y_term(
    y: np.ndarray,
    t: int,
    Ky: int,
    theta_y1: float,
    theta_y2: float
) -> float:
    w_y = exp_almon_weights(Ky, theta_y1, theta_y2)
    return sum(w_y[j] * y[t - j] for j in range(Ky + 1))

def adl_midas_forecast(
    y: np.ndarray,
    x: np.ndarray,
    h: int = 1,
    m: int = 3,
    Ky: int = 4,
    Kx: int = 4
):
    """
    ADL-MIDAS forecast following Eq. (2.25)-(2.26).
    Returns (forecasts, actuals).
    """
    T = len(y)
    t_start = max(Ky, Kx)

    def model(theta):
        beta_y, beta_x = theta[0], theta[1]
        th_y1, th_y2 = theta[2], theta[3]
        th_x1, th_x2 = theta[4], theta[5]

        forecasts = []
        actuals = []

        for t in range(t_start, T - h):
            y_part = midas_y_term(y, t, Ky, th_y1, th_y2)
            x_part = midas_x_term(x, t, m, Kx, th_x1, th_x2)

            yhat = beta_y * y_part + beta_x * x_part
            forecasts.append(yhat)
            actuals.append(y[t + h])

        return np.array(forecasts), np.array(actuals)

    def objective(theta):
        f, a = model(theta)
        err = a-f
        err = np.clip(err, -1e6, 1e6)  # avoid overflow
        return np.mean(err ** 2)

    # Initial values (important for convergence)
    theta0 = np.array([
        0.5,    # beta_y
        0.5,    # beta_x
       -0.1,    # theta_y1
       -0.01,   # theta_y2
       -0.1,    # theta_x1
       -0.01    # theta_x2
    ])

    res = minimize(objective, theta0, method="L-BFGS-B")

    return model(res.x)

def kalman_loglike_full(p: OneFactorParams, y: np.ndarray, x: np.ndarray) -> float:
    """
    Full (time-varying P) KF loglik with periodic measurement matrices.
    y: (T_low,)
    x: (T_high, n_x) with T_high = T_low*m
    """
    assert x.shape[0] == y.shape[0] * p.m
    assert x.shape[1] == p.n_x

    # build periodic mats
    Z_list, H_list = build_measurement_mats(p)
    G = np.diag([p.rho] + [p.d] * (1 + p.n_x))
    Q = np.diag([p.sig2_f, p.sig2_uy] + list(p.sig2_ux))

    dim = p.dim_state
    a = np.zeros(dim)
    P = np.eye(dim) * 10.0  # diffuse-ish init

    ll = 0.0
    low_idx = 0
    two_pi = np.log(2.0 * np.pi)

    for t_high in range(x.shape[0]):
        j = (t_high % p.m) + 1      # 1..m
        jj = j - 1                  # 0..m-1

        # predict
        a = G @ a
        P = G @ P @ G.T + Q

        # measurement
        if j < p.m:
            y_obs = x[t_high, :]                   # (n_x,)
            Z = Z_list[jj]                         # (n_x, dim)
            H = H_list[jj]                         # (n_x, n_x)
        else:
            y_obs = np.concatenate([[y[low_idx]], x[t_high, :]])  # (1+n_x,)
            Z = Z_list[jj]                         # (1+n_x, dim)
            H = H_list[jj]                         # (1+n_x, 1+n_x)
            low_idx += 1

        v = y_obs - (Z @ a)
        S = Z @ P @ Z.T + H

        # numerical stability
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return -np.inf

        # solve S^{-1}v using chol
        tmp = np.linalg.solve(L, v)
        Sinv_v = np.linalg.solve(L.T, tmp)
        quad = float(v.T @ Sinv_v)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        k = len(y_obs)

        ll += -0.5 * (logdet + quad + k * two_pi)

        # update
        # K = P Z' S^{-1} via chol solves
        # compute PZ' then solve for each column
        PZt = P @ Z.T
        # solve S^{-1} * (Z P)' = S^{-1} * (PZt)'
        # using chol: solve L w = (PZt)' then L.T u = w
        W = np.linalg.solve(L, PZt.T)
        U = np.linalg.solve(L.T, W)
        K = U.T  # (dim, k)

        a = a + K @ v
        P = P - K @ Z @ P

    return float(ll)

def fit_kalman_mle(y: np.ndarray, x: np.ndarray, m=3) -> OneFactorParams:
    n_x = x.shape[1]

    def neg_ll(theta):
        eps = 1e-8
        rho = np.tanh(theta[0])
        d = np.tanh(theta[1])
        sig2_f = np.exp(theta[2]) + eps
        sig2_uy = np.exp(theta[3]) + eps
        sig2_ux = np.exp(theta[4:4+n_x]) + eps

        p = OneFactorParams(
            m=m,
            n_x=n_x,
            lam_y=1.0,
            lam_x=np.ones(n_x),
            rho=rho,
            d=d,
            sig2_f=sig2_f,
            sig2_uy=sig2_uy,
            sig2_ux=sig2_ux
        )

        return -kalman_loglike_full(p, y, x)

    theta0 = np.array(
        [np.arctanh(0.2), np.arctanh(0.1),
         np.log(1.0), np.log(1.0)] + [np.log(1.0)] * n_x
    )

    res = minimize(neg_ll, theta0, method="L-BFGS-B")

    rho = np.tanh(res.x[0])
    d = np.tanh(res.x[1])
    sig2_f = np.exp(res.x[2])
    sig2_uy = np.exp(res.x[3])
    sig2_ux = np.exp(res.x[4:4+n_x])

    return OneFactorParams(
        m=m,
        n_x=n_x,
        lam_y=1.0,
        lam_x=np.ones(n_x),
        rho=rho,
        d=d,
        sig2_f=sig2_f,
        sig2_uy=sig2_uy,
        sig2_ux=sig2_ux
    )

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

def rmspe(forecast, actual):
    return np.sqrt(np.mean(((forecast - actual)) ** 2))

def monte_carlo_simulation_1(
        N=500, T=40, m=3, rho=0.9, d=0.5, h=1
):
    rmspe_midas = []
    rmspe_adl_midas = []
    rmspe_kf = []

    for i in range(N):
        y, x, _= simulate_one_factor_dgp(T=T, m=m, rho=rho, d=d, seed=i)
        # MIDAS Forecast
        midas_forecast, midas_actual = midas_regular_forecast(y, x, h=h, m=m)
        rmspe_midas.append(rmspe(midas_forecast, midas_actual))

        # ADL-MIDAS Forecast
        adl_forecast, adl_actual = adl_midas_forecast(y, x, h=h, m=m)
        rmspe_adl_midas.append(rmspe(adl_forecast, adl_actual))

        kf_forecast, kf_actual = kalman_filter_forecast(y, x, h=h, m=m)
        rmspe_kf.append(rmspe(kf_forecast, kf_actual))

    return {
        "KF / MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_midas),
        "KF / ADL-MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_adl_midas),
    }




# ----------------------------------------------------
# GRID as in the paper
# ----------------------------------------------------
RHO_GRID = [-0.9, -0.5, 0.5, 0.95]
D_GRID   = [-0.9, -0.5, 0.0, 0.5, 0.95]



# ----------------------------------------------------
# Wrapper for one panel
# ----------------------------------------------------
def run_panel(
    h: int,
    N: int = 500,
    T: int = 40,
    m: int = 3
):
    """
    Runs one panel of Table 4 (fixed horizon h).
    Returns two DataFrames:
      - KF / MIDAS
      - KF / ADL-MIDAS
    """
    res_midas = pd.DataFrame(index=D_GRID, columns=RHO_GRID, dtype=float)
    res_adl   = pd.DataFrame(index=D_GRID, columns=RHO_GRID, dtype=float)

    for d in D_GRID:
        for rho in RHO_GRID:
            out = monte_carlo_simulation_1(
                N=N,
                T=T,
                m=m,
                rho=rho,
                d=d,
                h=h
            )

            res_midas.loc[d, rho] = out["KF / MIDAS"]
            res_adl.loc[d, rho]   = out["KF / ADL-MIDAS"]

            print(f"h={h} | d={d:>4} | rho={rho:>5} | done")

    return res_midas, res_adl

# ----------------------------------------------------
# Generate all four panels
# ----------------------------------------------------
def generate_table_4(N=500):
    print("=== Panel A: One-Factor DGP, h = 1 ===")
    A_midas, A_adl = run_panel(h=1, N=N)

    print("=== Panel B: One-Factor DGP, h = 4 ===")
    B_midas, B_adl = run_panel(h=4, N=N)

    return {
        "Panel A (h=1) - Regular MIDAS": A_midas,
        "Panel A (h=1) - ADL-MIDAS": A_adl,
        "Panel B (h=4) - Regular MIDAS": B_midas,
        "Panel B (h=4) - ADL-MIDAS": B_adl,
    }

tables = generate_table_4() # Reduced N for quicker runs

tables["Panel A (h=1) - Regular MIDAS"]
tables["Panel A (h=1) - ADL-MIDAS"]
tables["Panel B (h=4) - Regular MIDAS"]
tables["Panel B (h=4) - ADL-MIDAS"]


with open("table4.tex", "w") as f:
    for name, df in tables.items():
        f.write(f"% {name}\n")
        f.write(df.to_latex(float_format="%.2f"))
        f.write("\n\n")
