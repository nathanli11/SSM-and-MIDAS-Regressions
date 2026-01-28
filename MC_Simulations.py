#============================
# Imports
#============================
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple


RICCATI_MAX_ITERS = 50_000
RICCATI_TOL = 1e-12

#============================
# Kalman filter
#============================
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
class TwoFactorParams:
    m: int = 3
    n_x: int = 1

    rho1: float = 0.9
    rho2: float = 0.3
    d: float = 0.0

    sig2_f1: float = 1.0
    sig2_f2: float = 1.0
    sig2_uy: float = 1.0
    sig2_ux: float = 1.0

    @property
    def dim_state(self):
        # [f1, f2, u_y, u_x]
        return 4


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


def build_measurement_mats(p: OneFactorParams) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Measurement equation:
      - for j=1..m-1: observe x's only
      - for j=m: observe [y, x1, x2, ...] (y at low frequency, plus x at end-of-period)
    With state = [f, u_y, u_x1, ..., u_xn]
    and y* = lam_y f + u_y, xi = lam_xi f + u_xi
    """
    n_x = p.n_x
    dim = p.dim_state

    Z_list: List[np.ndarray] = []
    H_list: List[np.ndarray] = []

    # j=1..m-1: x only
    for _ in range(p.m - 1):
        Z = np.zeros((n_x, dim))
        # each row i: xi = lam_x[i]*f + u_xi
        Z[:, 0] = p.lam_x
        for i in range(n_x):
            Z[i, 2 + i] = 1.0
        Z_list.append(Z)
        H_list.append(np.zeros((n_x, n_x)))

    # j=m: y + x's
    Zm = np.zeros((1 + n_x, dim))
    # y row
    Zm[0, 0] = p.lam_y
    Zm[0, 1] = 1.0
    # x rows
    Zm[1:, 0] = p.lam_x
    for i in range(n_x):
        Zm[1 + i, 2 + i] = 1.0
    Z_list.append(Zm)
    H_list.append(np.zeros((1 + n_x, 1 + n_x)))

    return Z_list, H_list

def periodic_steady_state_kf(p: OneFactorParams) -> PeriodicKF:
    """
    Computes periodic steady-state P_{j|j-1} and gains K_{j|j-1} via Riccati iteration (Eq. 2.8).
    """
    Z_list, H_list = build_measurement_mats(p)
    G = np.diag([p.rho] + [p.d] * (1 + p.n_x))
    Q = np.diag([p.sig2_f, p.sig2_uy] + list(p.sig2_ux))

    # initialize P_{1|0}
    P = np.eye(p.dim_state)

    # iterate until periodic fixed point
    P_pred = [np.zeros_like(P) for _ in range(p.m)]
    K_gain = [np.zeros((p.dim_state, Z_list[j].shape[0])) for j in range(p.m)]

    for it in range(RICCATI_MAX_ITERS):
        P_old = P.copy()
        # sweep through subperiods j=1..m
        for j in range(p.m):
            # predict
            Pp = G @ P @ G.T + Q
            Z = Z_list[j]
            H = H_list[j]
            S = Z @ Pp @ Z.T + H
            # gain
            K = Pp @ Z.T @ np.linalg.inv(S)
            # update
            P = (np.eye(p.dim_state) - K @ Z) @ Pp

            P_pred[j] = Pp
            K_gain[j] = K

        # check periodic convergence (compare start-of-cycle covariance)
        diff = np.max(np.abs(P - P_old))
        if diff < RICCATI_TOL:
            break
    else:
        raise RuntimeError("Riccati iteration did not converge. Try loosening tolerances or check parameters.")

    return PeriodicKF(params=p, P_pred=P_pred, K_gain=K_gain, Z_list=Z_list, H_list=H_list)

def run_periodic_kf_filter(
    kf: PeriodicKF,
    obs_y: np.ndarray,
    obs_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the periodic steady-state KF (using fixed gains from kf) on a high-frequency panel.

    obs_y: shape (T_low,) with y at low-frequency dates. (We will align it to subperiod j=m)
    obs_x: shape (T_high, n_x) with x at every high-frequency step.

    Returns:
      filtered states at each high step (T_high, dim_state),
      filtered states at each low step (T_low, dim_state) after processing j=m update.
    """
    p = kf.params
    m = p.m
    n_x = p.n_x
    dim = p.dim_state

    T_high = obs_x.shape[0]
    T_low = obs_y.shape[0]
    assert T_high == T_low * m

    G = kf.G()

    # allocate
    state_filt_high = np.zeros((T_high, dim))
    state_filt_low = np.zeros((T_low, dim))

    a = np.zeros(dim)     # state estimate
    P = np.eye(dim)       # not used in steady-state gain updating (but we still propagate state)
    # Note: we use fixed K, so P here is irrelevant; keep for sanity

    low_idx = 0
    for t_high in range(T_high):
        j = (t_high % m) + 1  # 1..m
        jj = j - 1            # 0..m-1 index

        # predict
        a = G @ a

        # update with observations available at this subperiod
        if j < m:
            y_obs = obs_x[t_high, :]  # x-only vector (n_x,)
            Z = kf.Z_list[jj]
            K = kf.K_gain[jj]
            innov = y_obs - (Z @ a)
            a = a + K @ innov
        else:
            # j=m: y + x_end
            yx_obs = np.concatenate([[obs_y[low_idx]], obs_x[t_high, :]])
            Z = kf.Z_list[jj]
            K = kf.K_gain[jj]
            innov = yx_obs - (Z @ a)
            a = a + K @ innov

            state_filt_low[low_idx, :] = a
            low_idx += 1

        state_filt_high[t_high, :] = a

    return state_filt_high, state_filt_low

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

def kalman_loglike_2f(p: TwoFactorParams, y, x):
    m = p.m
    T_low = len(y)
    T_high = T_low * m

    G = np.diag([p.rho1, p.rho2, p.d, p.d])
    Q = np.diag([p.sig2_f1, p.sig2_f2, p.sig2_uy, p.sig2_ux])

    a = np.zeros(4)
    P = np.eye(4) * 10
    ll = 0.0
    low_idx = 0

    for t in range(T_high):
        j = (t % m) + 1
        a = G @ a
        P = G @ P @ G.T + Q

        if j < m:
            Z = np.array([[1, 0, 0, 1]])   # x = f1 + u_x
            y_obs = np.array([x[t, 0]])
        else:
            Z = np.array([
                [1, 1, 1, 0],             # y = f1 + f2 + u_y
                [1, 0, 0, 1]              # x
            ])
            y_obs = np.array([y[low_idx], x[t, 0]])
            low_idx += 1

        v = y_obs - Z @ a
        S = Z @ P @ Z.T
        ll += -0.5 * (np.log(np.linalg.det(S)) + v.T @ np.linalg.solve(S, v))
        K = P @ Z.T @ np.linalg.inv(S)
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

def kalman_forecast_series(y, x, h=1, m=3):
    p_hat = fit_kalman_mle(y, x, m=m)
    kf = periodic_steady_state_kf(p_hat)
    _, states_low = run_periodic_kf_filter(kf, y, x)

    fcasts = []
    actuals = []
    for t in range(len(y) - h):
        fcasts.append(forecast_y_from_state(p_hat, states_low[t], h))
        actuals.append(y[t+h])

    return np.array(fcasts), np.array(actuals), p_hat

def kalman_ic_1f(y, x, m=3):
    p = fit_kalman_mle(y, x, m=m)
    ll = kalman_loglike_full(p, y, x)
    k = 5   # rho, d, sig2_f, sig2_uy, sig2_ux
    return ll, k, p

def kalman_ic_2f(y, x, m=3):
    p = TwoFactorParams(m=m)
    ll = kalman_loglike_2f(p, y, x)
    k = 6   # rho1, rho2, sig2_f1, sig2_f2, sig2_uy, sig2_ux
    return ll, k, p


#============================
# MIDAS & ADL-MIDAS
#============================
def exp_almon_weights(K: int, theta1: float, theta2: float) -> np.ndarray:
    """
    Numerically stable exponential Almon lag polynomial weights.
    """
    j = np.arange(K + 1, dtype=float)
    z = theta1 * j + theta2 * j * j

    # Numerical stabilization
    z = z - np.max(z)

    a = np.exp(z)
    s = a.sum()

    if not np.isfinite(s) or s <= 0:
        out = np.zeros(K + 1)
        out[0] = 1.0
        return out

    return a / s

def regular_midas_forecast(y, x, h=1, m=3, Ky=4, Kx=12):
    """
    Regular MIDAS (Eq. 3.5 in the paper):
      y_{t+h} = b0 + b_y * sum_{j=0..Ky} w_y(j;theta_y) y_{t-j}
                    + b_x * sum_{j=0..Kx} w_x(j;theta_x) x_{t - j/m}
                    + e_{t+h}
    No aggregator scheme for x: we directly use high-frequency lags x_{t - j/m}.
    """
    y = np.asarray(y).astype(float)
    x = np.asarray(x).astype(float)
    T = len(y)
    assert x.shape[0] == T * m

    # Need y lags up to Ky and HF x lags up to Kx at low time t
    t_min = max(Ky, int(np.ceil((Kx + 1) / m)))
    t_idx = np.arange(t_min, T - h)
    if len(t_idx) < 5:
        return np.array([]), np.array([])

    Y = y[t_idx + h]

    def build_terms(theta_y1, theta_y2, theta_x1, theta_x2):
        w_y = exp_almon_weights(Ky, theta_y1, theta_y2)   # Ky+1
        w_x = exp_almon_weights(Kx, theta_x1, theta_x2)   # Kx+1

        Yterm = np.zeros(len(t_idx))
        Xterm = np.zeros(len(t_idx))

        for ii, t in enumerate(t_idx):
            # MIDAS on y (low-frequency lags)
            yt = 0.0
            for j in range(Ky + 1):
                yt += w_y[j] * y[t - j]
            Yterm[ii] = yt

            # MIDAS on x (high-frequency lags) at low-frequency time t
            xt = 0.0
            for j in range(Kx + 1):
                val = hf_lag_at_low_t(x, t, m, j)  # HF lag j from end of period t
                if np.isnan(val):
                    xt = np.nan
                    break
                xt += w_x[j] * val
            Xterm[ii] = xt

        return Yterm, Xterm

    def mse(theta):
        th_y1, th_y2, th_x1, th_x2 = map(float, theta)
        Yterm, Xterm = build_terms(th_y1, th_y2, th_x1, th_x2)
        if np.any(np.isnan(Xterm)) or np.any(np.isnan(Yterm)):
            return 1e18

        # Profile out betas by OLS: Y ≈ b0 + b_y*Yterm + b_x*Xterm
        Xreg = np.column_stack([np.ones(len(t_idx)), Yterm, Xterm])
        beta = lstsq(Xreg, Y, rcond=None)[0]
        resid = Y - Xreg @ beta
        return float(np.mean(resid ** 2))

    theta0 = np.array([-0.1, -0.01,  -0.1, -0.01])
    bnds = [(-10, 10)] * 4
    res = minimize(mse, theta0, method="L-BFGS-B", bounds=bnds)

    th_y1, th_y2, th_x1, th_x2 = map(float, res.x)
    Yterm, Xterm = build_terms(th_y1, th_y2, th_x1, th_x2)
    Xreg = np.column_stack([np.ones(len(t_idx)), Yterm, Xterm])
    beta = lstsq(Xreg, Y, rcond=None)[0]

    forecasts = Xreg @ beta
    actuals = Y
    return np.asarray(forecasts), np.asarray(actuals)

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
    val = 0.0
    for k in range(m):
        idx = t*m-1-k
        if idx<0:
            continue
        val += w[k] * x[idx, 0]

    return val

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

def multiplicative_midas_forecast(y, x, h=1, m=3, Ky=4, Kx=4):
    """
    Paper-style Multiplicative MIDAS (ADL-MIDAS):
      y_{t+h} = b_y * sum_{j=0..Ky} w_y(j;theta_y) y_{t-j}
              + b_x * sum_{j=0..Kx} w_x_inter(j;theta_x_inter) x_agg(t-j;theta_x_intra)
              + e

    where:
      x_agg(t;theta_x_intra) = sum_{k=0..m-1} w_x_intra(k;theta_x_intra) x_{t - k/m}
      and w_* are exponential Almon weights.
    """
    y = np.asarray(y).astype(float)
    x = np.asarray(x).astype(float)
    T = len(y)

    # Need y-lags up to Ky and x inter-lags up to Kx plus intra m-1
    t_min = max(Ky, Kx) + 1
    t_idx = np.arange(t_min, T - h)
    if len(t_idx) < 5:
        return np.array([]), np.array([])

    Y = y[t_idx + h]

    def x_intra_agg(t, th_intra1, th_intra2):
        w_intra = exp_almon_weights(m - 1, th_intra1, th_intra2)  # length m
        s = 0.0
        for k in range(m):
            val = hf_lag_at_low_t(x, t, m, k)  # lag_hf=k within the quarter
            if np.isnan(val):
                return np.nan
            s += w_intra[k] * val
        return s

    def build_terms(theta):
        th_y1, th_y2, th_xi1, th_xi2, th_xa1, th_xa2 = map(float, theta)

        w_y = exp_almon_weights(Ky, th_y1, th_y2)       # Ky+1
        w_xi = exp_almon_weights(Kx, th_xi1, th_xi2)    # inter Kx+1

        Yterm = np.zeros(len(t_idx))
        Xterm = np.zeros(len(t_idx))

        for ii, t in enumerate(t_idx):
            # MIDAS on y (low-frequency lags)
            yt = 0.0
            for j in range(Ky + 1):
                yt += w_y[j] * y[t - j]
            Yterm[ii] = yt

            # double MIDAS on x: intra -> inter
            xt = 0.0
            for j in range(Kx + 1):
                xa = x_intra_agg(t - j, th_xa1, th_xa2)
                if np.isnan(xa):
                    xt = np.nan
                    break
                xt += w_xi[j] * xa
            Xterm[ii] = xt

        return Yterm, Xterm

    def mse(theta):
        Yterm, Xterm = build_terms(theta)
        if np.any(np.isnan(Xterm)) or np.any(np.isnan(Yterm)):
            return 1e18

        # Profile out betas by OLS: Y ≈ b_y*Yterm + b_x*Xterm
        Xreg = np.column_stack([Yterm, Xterm])
        beta = lstsq(Xreg, Y, rcond=None)[0]
        resid = Y - Xreg @ beta
        return float(np.mean(resid ** 2))

    theta0 = np.array([-0.1, -0.01,  -0.1, -0.01,  -0.1, -0.01])
    bnds = [(-10, 10)] * 6
    res = minimize(mse, theta0, method="L-BFGS-B", bounds=bnds)

    Yterm, Xterm = build_terms(res.x)
    Xreg = np.column_stack([Yterm, Xterm])
    beta = lstsq(Xreg, Y, rcond=None)[0]

    forecasts = Xreg @ beta
    actuals = Y
    return np.asarray(forecasts), np.asarray(actuals)
#============================
# DGP
#============================
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

def simulate_two_factor_dgp(
        T=40,
        m=3,
        rho=0.9,
        d=0.5,
        seed=None
):
    if seed is not None:
        np.random.seed(seed)
    Th = T * m

    # Innovations
    eta1 = np.random.normal(size=Th)
    eta2 = np.random.normal(size=Th)
    epsy = np.random.normal(size=Th)
    epsx = np.random.normal(size=Th)

    # Factors
    f1 = np.zeros(Th)
    f2 = np.zeros(Th)
    for t in range(1, Th):
        f1[t] = rho * f1[t - 1] + eta1[t]
        f2[t] = rho * f2[t - 1] + eta2[t]
    
    # Measurment errors
    uy = np.zeros(Th)
    ux = np.zeros(Th)
    for t in range(1, Th):
        uy[t] = d * uy[t - 1] + epsy[t]
        ux[t] = d * ux[t - 1] + epsx[t]
    
    # Observations
    y_star = f1 + f2 + uy
    x = f1 + ux

    # Low frequency
    y = y_star[m-1::m]

    return y, x.reshape(-1,1), f1, f2

#============================
# Utilities
#============================
def rmspe(forecast, actual):
    return np.sqrt(np.mean(((forecast - actual)) ** 2))

def gaussian_loglike(residuals):
    residuals = np.asarray(residuals)
    # clip pour éviter explosions numériques
    residuals = np.clip(residuals, -1e6, 1e6)
    T = len(residuals)
    sigma2 = np.mean(residuals**2)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return -np.inf
    return -0.5 * T * (np.log(2*np.pi*sigma2) + 1)

def aic(loglike: float, k: int) -> float:
    return -2 * loglike + 2 * k

def bic(loglike: float, k: int, T: int) -> float:
    return -2 * loglike + k * np.log(T)

def midas_ic(y, x, h=1, m=3, K=12):
    fcst, act = regular_midas_forecast(y, x, h=h, m=m, K=K)
    residuals = act - fcst
    loglike = gaussian_loglike(residuals)
    # params: b0,b1,b2 + theta1,theta2  => 5
    k = 5
    return loglike, k, len(residuals)

def adl_midas_ic(y, x, h=1, m=3, Ky=4, Kx=4):
    fcst, act = multiplicative_midas_forecast(y, x, h=h, m=m, Ky=Ky, Kx=Kx)
    resid = act - fcst
    ll = gaussian_loglike(resid)
    # betas(2) + theta_y(2) + theta_x_inter(2) + theta_x_intra(2) => 8
    k = 8
    return ll, k, len(resid)

def kalman_ic(y, x, m=3):
    p_hat = fit_kalman_mle(y, x, m=m)
    ll = kalman_loglike_full(p_hat, y, x)
    k = 2 + 3   # rho, d + 3 variances
    return ll, k, p_hat

def hf_lag_at_low_t(x: np.ndarray, t: int, m: int, lag_hf: int) -> float:
    """
    x is high-frequency array shape (T*m, 1) or (T*m,)
    Low-frequency time t corresponds to end-of-period high-frequency index t*m-1.
    lag_hf=0 means x at end of period t.
    """
    if x.ndim == 2:
        x1 = x[:, 0]
    else:
        x1 = x
    idx = t * m - 1 - lag_hf
    if idx < 0 or idx >= len(x1):
        return np.nan
    return float(x1[idx])
#============================
# Monte Carlo Simulation
#============================
def monte_carlo_simulation_1(
        N=500, T=40, m=3, rho=0.9, d=0.5, h=1
):
    rmspe_midas = []
    rmspe_adl_midas = []
    rmspe_kf = []

    for i in range(N):
        y, x, _= simulate_one_factor_dgp(T=T, m=m, rho=rho, d=d, seed=i)
        # MIDAS Forecast
        midas_forecast, midas_actual = regular_midas_forecast(y, x, h=h, m=m)
        rmspe_midas.append(rmspe(midas_forecast, midas_actual))

        # ADL-MIDAS Forecast
        adl_forecast, adl_actual = multiplicative_midas_forecast(y, x, h=h, m=m)
        rmspe_adl_midas.append(rmspe(adl_forecast, adl_actual))

        kf_forecast, kf_actual = kalman_filter_forecast(y, x, h=h, m=m)
        rmspe_kf.append(rmspe(kf_forecast, kf_actual))

    return {
        "KF / MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_midas),
        "KF / ADL-MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_adl_midas),
    }

def monte_carlo_simulation_2(
        N=500,
        T=40,
        m=3,
        rho=0.9,
        d=0.5,
        h=1
):
    rmspe_midas = []
    rmspe_adl   = []
    rmspe_kf    = []

    for i in range(N):
        y, x, _, _ = simulate_two_factor_dgp(
            T=T,
            m=m,
            rho=rho,
            d=d,
            seed=i
        )

        # MIDAS Forecast
        midas_forecast, midas_actual = regular_midas_forecast(y, x, h=h, m=m)
        rmspe_midas.append(rmspe(midas_forecast, midas_actual))

        # ADL-MIDAS Forecast
        adl_forecast, adl_actual = multiplicative_midas_forecast(y, x, h=h, m=m)
        rmspe_adl.append(rmspe(adl_forecast, adl_actual))

        # Kalman Filter Forecast (MISSPECIFIED)
        try:    
            kf_forecast, kf_actual = kalman_filter_forecast(y, x, h=h, m=m)
            rmspe_kf.append(rmspe(kf_forecast, kf_actual))
        except RuntimeError:
            # Riccati did not converge; skip this iteration
            continue
    return {
        "KF / MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_midas),
        "KF / ADL-MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_adl),
    }

def monte_carlo_simulation_3(
    N=500,
    T=40,
    m=3,
    rho=0.9,
    d=0.5,
    h=1,
    criterion="AIC"
):
    rmspe_kf = []
    rmspe_midas = []
    rmspe_adl = []

    for i in range(N):
        y, x, _ = simulate_one_factor_dgp(
            T=T, m=m, rho=rho, d=d, seed=i
        )

        # ======================
        # MIDAS
        # ======================
        f_m, a_m = regular_midas_forecast(y, x, h=h, m=m)
        rmspe_midas.append(rmspe(f_m, a_m))

        # ======================
        # ADL-MIDAS
        # ======================
        f_a, a_a = multiplicative_midas_forecast(y, x, h=h, m=m)
        rmspe_adl.append(rmspe(f_a, a_a))

        # ======================
        # Kalman: select #factors by IC
        # ======================
        ll1, k1, p1 = kalman_ic_1f(y, x, m)
        ll2, k2, p2 = kalman_ic_2f(y, x, m)

        if criterion == "AIC":
            ic1 = aic(ll1, k1)
            ic2 = aic(ll2, k2)
        else:
            ic1 = bic(ll1, k1, T)
            ic2 = bic(ll2, k2, T)

        # Le papier produit toujours le modèle 1 facteur
        # on suit cette convention ici
        p_hat = p1 
        kf = periodic_steady_state_kf(p_hat)
        _, states_low = run_periodic_kf_filter(kf, y, x)

        fcast = []
        actual = []
        for t in range(len(y) - h):
            fcast.append(forecast_y_from_state(p_hat, states_low[t], h))
            actual.append(y[t + h])

        rmspe_kf.append(rmspe(np.array(fcast), np.array(actual)))

    return {
        "KF / MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_midas),
        "KF / ADL-MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_adl),
    }


#===========================
# Panels
#===========================

RHO_GRID = [-0.9, -0.5, 0.5, 0.95]   # observed factor (in x)
D_GRID    = [-0.9, -0.5, 0.0, 0.5, 0.95]

def run_panel_simulation_1(
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

def run_panel_simulation_2(
    h: int,
    N: int = 500,
    T: int = 40,
    m: int = 3
):
    res_midas = pd.DataFrame(index=D_GRID, columns=RHO_GRID)
    res_adl   = pd.DataFrame(index=D_GRID, columns=RHO_GRID)

    for d in D_GRID:
        for rho in RHO_GRID:
            out = monte_carlo_simulation_2(
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


def run_panel_simulation_3(
    h: int,
    criterion: str,
    N: int = 500,
    T: int = 40,
    m: int = 3
):
    """
    Runs one panel of Table 5 (Simulation 3).
    Rows: d
    Columns: rho
    """
    res_midas = pd.DataFrame(index=D_GRID, columns=RHO_GRID, dtype=float)
    res_adl   = pd.DataFrame(index=D_GRID, columns=RHO_GRID, dtype=float)

    for d in D_GRID:
        for rho in RHO_GRID:
            out = monte_carlo_simulation_3(
                N=N,
                T=T,
                m=m,
                rho=rho,
                d=d,
                h=h,
                criterion=criterion
            )

            res_midas.loc[d, rho] = out["KF / MIDAS"]
            res_adl.loc[d, rho]   = out["KF / ADL-MIDAS"]

            print(f"{criterion} | h={h} | d={d:>4} | rho={rho:>5} | done")

    return res_midas, res_adl


#===========================
# Tables
#===========================
def generate_table_4A(N=500):
    print("=== Panel A: One-Factor DGP, h = 1 ===")
    A_midas, A_adl = run_panel_simulation_1(h=1, N=N)

    print("=== Panel B: One-Factor DGP, h = 4 ===")
    B_midas, B_adl = run_panel_simulation_1(h=4, N=N)

    return {
        "Panel A (h=1) - Regular MIDAS": A_midas,
        "Panel A (h=1) - Multiplicative MIDAS": A_adl,
        "Panel B (h=4) - Regular MIDAS": B_midas,
        "Panel B (h=4) - Multiplicative MIDAS": B_adl,
    }

def generate_table_4B(N=500):
    print("=== Table 4 – Panel C (Two-Factor DGP, h=1) ===")
    C_midas, C_adl = run_panel_simulation_2(h=1, N=N)

    print("=== Table 4 – Panel D (Two-Factor DGP, h=4) ===")
    D_midas, D_adl = run_panel_simulation_2(h=4, N=N)

    return {
        "Panel C (h=1) - Regular MIDAS": C_midas,
        "Panel C (h=1) - Multiplicative MIDAS": C_adl,
        "Panel D (h=4) - Regular MIDAS": D_midas,
        "Panel D (h=4) - Multiplicative MIDAS": D_adl,
    }


def generate_table_5(N: int = 500):
    """
    Generates all panels of Table 5 (Simulation 3).
    """

    print("=== Table 5, Panel A: AIC, h = 1 ===")
    A_midas, A_adl = run_panel_simulation_3(
        h=1, criterion="AIC", N=N
    )

    print("=== Table 5, Panel B: BIC, h = 1 ===")
    B_midas, B_adl = run_panel_simulation_3(
        h=1, criterion="BIC", N=N
    )

    print("=== Table 5, Panel C: AIC, h = 4 ===")
    C_midas, C_adl = run_panel_simulation_3(
        h=4, criterion="AIC", N=N
    )

    print("=== Table 5, Panel D: BIC, h = 4 ===")
    D_midas, D_adl = run_panel_simulation_3(
        h=4, criterion="BIC", N=N
    )

    return {
        "Panel A (AIC, h=1) - Regular MIDAS": A_midas,
        "Panel A (AIC, h=1) - Multiplicative MIDAS": A_adl,
        "Panel B (BIC, h=1) - Regular MIDAS": B_midas,
        "Panel B (BIC, h=1) - Multiplicative MIDAS": B_adl,
        "Panel C (AIC, h=4) - Regular MIDAS": C_midas,
        "Panel C (AIC, h=4) - Multiplicative MIDAS": C_adl,
        "Panel D (BIC, h=4) - Regular MIDAS": D_midas,
        "Panel D (BIC, h=4) - Multiplicative MIDAS": D_adl,
    }

tables_4B = generate_table_4B(N=5)

tables_4B["Panel C (h=1) - Regular MIDAS"].to_excel("Table_4_PanelC_MIDAS.xlsx")
tables_4B["Panel C (h=1) - Multiplicative MIDAS"].to_excel("Table_4_PanelC_ADL_MIDAS.xlsx")

tables_4B["Panel D (h=4) - Regular MIDAS"].to_excel("Table_4_PanelD_MIDAS.xlsx")
tables_4B["Panel D (h=4) - Multiplicative MIDAS"].to_excel("Table_4_PanelD_ADL_MIDAS.xlsx")


tables_4A = generate_table_4A(N=5)
tables_4A["Panel A (h=1) - Regular MIDAS"].to_excel(
    "Table_4A_PanelA_MIDAS.xlsx"
)
tables_4A["Panel A (h=1) - Multiplicative MIDAS"].to_excel(
    "Table_4A_PanelA_Multiplicative_MIDAS.xlsx"
)

tables_4A["Panel B (h=4) - Regular MIDAS"].to_excel(
    "Table_4A_PanelB_MIDAS.xlsx"
)

tables_4A["Panel B (h=4) - Multiplicative MIDAS"].to_excel(
    "Table_4A_PanelB_Multiplicative_MIDAS.xlsx"
)



tables_5 = generate_table_5(N=5)

tables_5["Panel A (AIC, h=1) - Regular MIDAS"].to_excel(
    "Table_5_PanelA_MIDAS.xlsx"
)
tables_5["Panel A (AIC, h=1) - Multiplicative MIDAS"].to_excel(
    "Table_5_PanelA_Multiplicative_MIDAS.xlsx"
)

tables_5["Panel B (BIC, h=1) - Regular MIDAS"].to_excel(
    "Table_5_PanelB_MIDAS.xlsx"
)
tables_5["Panel B (BIC, h=1) - Multiplicative MIDAS"].to_excel(
    "Table_5_PanelB_Multiplicative_MIDAS.xlsx"
)

tables_5["Panel C (AIC, h=4) - Regular MIDAS"].to_excel(
    "Table_5_PanelC_MIDAS.xlsx"
)
tables_5["Panel C (AIC, h=4) - Multiplicative MIDAS"].to_excel(
    "Table_5_PanelC_Multiplicative_MIDAS.xlsx"
)

tables_5["Panel D (BIC, h=4) - Regular MIDAS"].to_excel(
    "Table_5_PanelD_MIDAS.xlsx"
)
tables_5["Panel D (BIC, h=4) - Multiplicative MIDAS"].to_excel(
    "Table_5_PanelD_Multiplicative_MIDAS.xlsx"
)
