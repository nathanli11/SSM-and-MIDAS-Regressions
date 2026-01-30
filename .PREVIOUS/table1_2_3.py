
"""
Replicate Tables 1, 2, 3 from:
Bai, Ghysels, Wright (2013) "State Space Models and MIDAS Regressions"

What this script does (population / asymptotic objects):
- Table 1: minimizes the L2 distance (Eq. 3.8) between Kalman-filter implied weights and MIDAS weights
          (regular MIDAS Eq. 3.5 and multiplicative/ADL-MIDAS Eq. 2.25) for the one-factor DGP (Sec. 3.1).
- Table 2: same L2 distance idea for two high-frequency series (Eq. 3.9), equal vs unequal noise variance.
- Table 3: PE distance ratios (Eq. 3.7), MIDAS vs misspecified one-factor state space model (Sec. 3.3),
          with MIDAS parameters chosen to minimize the PE variance objective (Eq. 3.6).

IMPORTANT PRACTICAL NOTES (why you might not match the paper on the first run):
1) The paper truncates infinite Kalman weights at lag length K̄ (they denote it K̄). The PDF excerpt says
   "Assuming KF weights negligible beyond lag length K̄" but the exact numeric K̄ used for Tables 1–3
   is not always explicit in the plain-text parse. I set K_BAR=10 by default (common in the paper’s notation
   with 3K̄ high-freq lags when m=3). If your numbers differ, the FIRST knob to check is K_BAR.
2) Periodic steady-state Kalman filtering is used (missing y within the period). This code computes the
   periodic steady state by Riccati iteration.
3) For Tables 1–2, MIDAS parameters are chosen to minimize Eq. (3.8) (L2 distance between *weights*).
4) For Table 3, MIDAS parameters are chosen to minimize Eq. (3.6) (prediction error variance).
   The misspecified SS1 model parameters are chosen here by minimizing the SS1 PE variance under the
   true (two-factor) covariance—this matches the spirit of Sec. 3.3’s “pick parameters to best fit”.

Dependencies: numpy, scipy, pandas
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from scipy.optimize import minimize

# -----------------------------
# Global numerical knobs
# -----------------------------
K_BAR = 40                 # lag truncation K̄ used in Eqs (3.4)-(3.5) and Appendix weight vectors
RICCATI_MAX_ITERS = 50_000
RICCATI_TOL = 1e-12
KF_WARMUP_PERIODS = 400    # warmup in low-freq periods when extracting impulse-response weights
OPT_TOL = 1e-10

# Grid for tables (as in the paper)
D_GRID = [-0.9, -0.5, 0.0, 0.5, 0.95]
RHO_GRID = [-0.9, -0.5, 0.0, 0.5, 0.95]

# -----------------------------
# Utilities
# -----------------------------
def exp_almon_weights(K: int, theta1: float, theta2: float) -> np.ndarray:
    """
    Exponential Almon lag polynomial weights (Eq. 2.19) over j=0..K
    normalized to sum to 1.
    """
    j = np.arange(K + 1, dtype=float)
    a = np.exp(theta1 * j + theta2 * j * j)
    s = a.sum()
    if not np.isfinite(s) or s <= 0:
        # fail-safe: return something harmless rather than nan
        out = np.zeros(K + 1)
        out[0] = 1.0
        return out
    return a / s


def block_diag(*mats: np.ndarray) -> np.ndarray:
    """Simple block diagonal constructor."""
    n = sum(m.shape[0] for m in mats)
    out = np.zeros((n, n))
    i = 0
    for m in mats:
        k = m.shape[0]
        out[i:i+k, i:i+k] = m
        i += k
    return out


# -----------------------------
# State-space model definitions
# -----------------------------
@dataclass(frozen=True)
class OneFactorParams:
    m: int
    rho: float        # factor AR(1)
    d: float          # measurement error AR(1) (same for all u's here)
    lam_y: float      # loading for y*
    lam_x: np.ndarray # loadings for each x series (shape = (n_x,))
    sig2_f: float     # Var(eps_f)
    sig2_uy: float    # Var(eps_u_y)
    sig2_ux: np.ndarray # Var(eps_u_xi) (shape = (n_x,))

    @property
    def n_x(self) -> int:
        return int(self.lam_x.shape[0])

    @property
    def dim_state(self) -> int:
        # state = [f, u_y, u_x1, ..., u_xn]
        return 2 + self.n_x


@dataclass(frozen=True)
class TwoFactorDGPParams:
    m: int
    rho: float
    d: float
    # factor loadings: y* = a1 f1 + a2 f2 + u_y ; xi = b1_i f1 + b2_i f2 + u_xi
    a: np.ndarray      # shape (2,)
    b: np.ndarray      # shape (n_x, 2)
    sig2_f1: float = 1.0
    sig2_f2: float = 1.0
    sig2_uy: float = 1.0
    sig2_ux: Optional[np.ndarray] = None  # shape (n_x,)

    def __post_init__(self):
        if self.sig2_ux is None:
            object.__setattr__(self, "sig2_ux", np.ones(self.b.shape[0], dtype=float))

    @property
    def n_x(self) -> int:
        return int(self.b.shape[0])


# -----------------------------
# Periodic steady-state Kalman filter (Riccati iteration)
# -----------------------------
@dataclass
class PeriodicKF:
    params: OneFactorParams
    # steady-state objects per subperiod j=1..m
    P_pred: List[np.ndarray]  # P_{j|j-1}
    K_gain: List[np.ndarray]  # K_{j|j-1}
    Z_list: List[np.ndarray]  # Z_j
    H_list: List[np.ndarray]  # measurement noise covariance per j (here always 0 because u's are in state)

    def G(self) -> np.ndarray:
        p = self.params
        # diag(rho, d, d, ..., d)
        diag = np.array([p.rho] + [p.d] * (1 + p.n_x), dtype=float)
        return np.diag(diag)

    def Q(self) -> np.ndarray:
        p = self.params
        diag = np.array([p.sig2_f, p.sig2_uy] + list(p.sig2_ux), dtype=float)
        return np.diag(diag)


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


# -----------------------------
# Run the (periodic) KF on data and extract weights by impulses
# -----------------------------
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


def kalman_weights_by_impulses(p: OneFactorParams, h: int, K_bar: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract KF weights w_KF,y,j for j=0..K_bar and w_KF,x,k for k=0..m*K_bar
    from the periodic steady-state filter, via linear impulse-response identification.

    We build a long sample of zeros, inject a single 1 in one lagged regressor,
    run the steady-state periodic KF, and measure the effect on the forecast.

    This is robust and avoids re-deriving Appendix A closed forms.
    """
    kf = periodic_steady_state_kf(p)
    m = p.m
    n_x = p.n_x

    # sample length in low-freq periods
    T_low = KF_WARMUP_PERIODS + (K_bar + 5) + (h + 2)
    T_high = T_low * m

    # baseline (all zeros)
    base_y = np.zeros(T_low)
    base_x = np.zeros((T_high, n_x))

    # run baseline to define "zero" (should be zero)
    _, base_states_low = run_periodic_kf_filter(kf, base_y, base_x)
    t0 = KF_WARMUP_PERIODS + K_bar + 1  # low-frequency index where we read forecast (after update)
    yhat0 = forecast_y_from_state(p, base_states_low[t0], h=h)

    # weights for y-lags (low-frequency y_t-j)
    w_y = np.zeros(K_bar + 1)
    for j in range(K_bar + 1):
        y = base_y.copy()
        # impulse at y_{t0 - j}
        y[t0 - j] = 1.0
        _, states_low = run_periodic_kf_filter(kf, y, base_x)
        yhat = forecast_y_from_state(p, states_low[t0], h=h)
        w_y[j] = yhat - yhat0

    # weights for x-lags at high frequency x_{t0 - k/m}
    # align: x at high index corresponding to end-of-period for t0 is (t0*m + (m-1))
    end_high = t0 * m + (m - 1)
    w_x = np.zeros(m * K_bar + 1)

    for k in range(m * K_bar + 1):
        x = base_x.copy()
        # impulse in *first* x series if multiple? For Tables 1 & 3 n_x=1; for Table 2 handle separately.
        # Here we return weights for x1 only if n_x>1; for Table 2 we use a different function.
        x[end_high - k, 0] = 1.0
        _, states_low = run_periodic_kf_filter(kf, base_y, x)
        yhat = forecast_y_from_state(p, states_low[t0], h=h)
        w_x[k] = yhat - yhat0

    return w_y, w_x


def kalman_weights_by_impulses_multi_x(p: OneFactorParams, h: int, K_bar: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same as kalman_weights_by_impulses, but returns x-weights for ALL x-series:
      w_y: (K_bar+1,)
      w_x: (n_x, m*K_bar+1)  where w_x[i,k] is weight on x_i,t-k/m
    """
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
    """
    Regular MIDAS Eq. (3.5):
      y_{t+h} = beta_y * sum_{j=0..K} wj(theta_y) y_{t-j}
             + beta_x * sum_{k=0..mK} wk(theta_x) x_{t-k/m} + error
    Returns coefficient vectors (w_y, w_x).
    """
    wy = beta_y * exp_almon_weights(K_bar, theta_y[0], theta_y[1])
    wx = beta_x * exp_almon_weights(m * K_bar, theta_x[0], theta_x[1])
    return wy, wx


def midas_multiplicative_coeffs(K_bar: int, m: int,
                                theta_y: Tuple[float, float],
                                theta_outer_x: Tuple[float, float],
                                theta_inner_x: Tuple[float, float],
                                beta_y: float, beta_x: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multiplicative/ADL-MIDAS idea (Eq. 2.25 referenced in Sec. 3.1):
      - an "inner" aggregation of the m high-freq observations within each low-freq period
      - then an "outer" distributed lag over low-frequency lags
    This yields x coefficients as a convolution:
      coeff on x_{t-(j*m + r)/m} = beta_x * w_outer[j] * w_inner[r],  j=0..K, r=0..m-1
    Returned wx is indexed by k=0..mK with k = j*m + r.
    """
    wy = beta_y * exp_almon_weights(K_bar, theta_y[0], theta_y[1])
    w_outer = exp_almon_weights(K_bar, theta_outer_x[0], theta_outer_x[1])
    w_inner = exp_almon_weights(m - 1, theta_inner_x[0], theta_inner_x[1])  # r=0..m-1 (length m)
    # NOTE: exp_almon_weights(K) gives length K+1. To get length m, use K=m-1.

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
    """
    Minimize Eq. (3.8) over theta_y, theta_x.
    For fixed thetas, optimal betas are simple least-squares projections onto the weight shapes.
    """
    wy_shape = lambda th: exp_almon_weights(K_bar, th[0], th[1])
    wx_shape = lambda th: exp_almon_weights(m * K_bar, th[0], th[1])

    def obj(u: np.ndarray) -> float:
        th_y = (u[0], u[1])
        th_x = (u[2], u[3])
        sy = wy_shape(th_y)
        sx = wx_shape(th_x)

        # optimal betas (separable L2)
        by = float(np.dot(wy_kf, sy) / max(1e-15, np.dot(sy, sy)))
        bx = float(np.dot(wx_kf, sx) / max(1e-15, np.dot(sx, sx)))

        wy_m, wx_m = midas_regular_coeffs(K_bar, m, th_y, th_x, by, bx)
        return l2_distance_weights(wy_kf, wx_kf, wy_m, wx_m)

    # initial guess: mild decay
    x0 = np.array([-0.2, 0.0, -0.2, 0.0])
    res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": OPT_TOL, "fatol": OPT_TOL, "maxiter": 50_000})
    return float(res.fun)


def fit_multiplicative_midas_to_kf_by_l2(wy_kf: np.ndarray, wx_kf: np.ndarray, K_bar: int, m: int) -> float:
    """
    Minimize Eq. (3.8) over theta_y, theta_outer_x, theta_inner_x.
    Betas again fitted by projection for fixed shapes (jointly for x because convolution scale).
    """
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
# Covariance matrices for PE objective (Eq. 3.6)
# -----------------------------
def cov_one_or_two_factor_yx(
    m: int,
    rho: float,
    d: float,
    # factor part covariance matrix at lag ℓ: rho^|ℓ| * V_factor, where V_factor is 2x2 for (y*, x*)
    V_factor: np.ndarray,
    sig2_uy: float,
    sig2_ux: float,
) -> Tuple[callable, callable, callable]:
    """
    Returns functions for covariances at high-frequency lags (in units of 1/m):
      Cov(y*_{t-i/m}, y*_{t-j/m}), Cov(x_{t-i/m}, x_{t-j/m}), Cov(x_{t-i/m}, y*_{t-j/m})
    where |i-j| is in high-frequency steps.

    Factor AR(1): innovations variance embedded in V_factor already (i.e. unconditional scale).
    Measurement errors: AR(1) with parameter d and innovation variances sig2_uy, sig2_ux
      => Var(u) = sig2_u / (1-d^2), Cov(u_t, u_{t-ℓ}) = d^|ℓ| * Var(u)
    """
    # note: V_factor is unconditional contemporaneous covariance contribution from factors at lag 0.
    # At lag L: rho^|L| * V_factor

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
        return float((rho ** L) * V_factor[1, 0])  # measurement errors independent across series

    return cov_yy, cov_xx, cov_xy


def sigma_matrix_for_upsilon(m: int, h: int, K_bar: int,
                            cov_yy, cov_xx, cov_xy) -> np.ndarray:
    """
    Build Σ_xy for upsilon_t = (x_{t+h}, y*_{t+h}, x_{t+h-1/m}, y*_{t+h-1/m}, ..., x_{t-K̄}, y*_{t-K̄})
    as described just before Eq. (3.6) / in Sec. 3.1.

    Index i,j in high-frequency steps. Represent each element as position in that upsilon vector.
    """
    # high-frequency indices run from -m*h ... m*K_bar  (paper uses i,j = -3h,...,3K̄ for m=3)
    # We map each entry in upsilon to a high-frequency offset.
    # Construction pattern: for each high-frequency step s from -m*h ... m*K_bar:
    #   include x_{t - s/m} and y*_{t - s/m} BUT the paper's ordering starts at future (t+h) then goes backward by 1/m.
    # We'll build explicitly in the given order:
    steps = list(range(-m * h, m * K_bar + 1))  # integer high-frequency step offsets
    # upsilon order: (x at step -m*h, y at step -m*h), (x at -m*h+1, y at -m*h+1), ..., (x at m*K, y at m*K)
    # but note: x_{t+h} corresponds to i = -m*h when we write x_{t - i/m}.
    # We'll use i indexes directly as above.
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
    """
    Build the forecast error weight vector w (Appendix A bottom):
      w = (1, 0_{1×(mh-1)}, -wx0, -wy0, -wx1, 0, -wx2, 0, ..., -wx(mK), -wy(K))
    matching upsilon_t ordering.
    """
    # upsilon begins with x_{t+h}, y*_{t+h}, x_{t+h-1/m}, y*_{t+h-1/m}, ..., x_{t-K}, y*_{t-K}
    # The forecast is y_{t+h} - sum wy_j y_{t-j} - sum wx_k x_{t-k/m}
    # In upsilon coordinates, y_{t+h} is the y* component at step -m*h, which is index 1 in our vector,
    # BUT the paper writes w' upsilon where the FIRST element is x_{t+h}. Their w starts with (1, 0..., -wx0, -wy0, ...)
    # To stay consistent with our sigma_matrix_for_upsilon ordering (x then y for each step),
    # we place weight on y*_{t+h} (the second element) as +1, and weight on x_{t+h} as 0.
    #
    # So our w differs by a permutation from the paper’s displayed w, but the quadratic form w'Σw is invariant
    # as long as Σ and w share the same ordering. Here they do.

    steps = list(range(-m * h, m * K_bar + 1))
    dim = 2 * len(steps)
    w = np.zeros(dim)

    # +1 on y*_{t+h} which corresponds to step = -m*h, y-position
    w[1] = 1.0

    # subtract x coefficients for x_{t - k/m}: that corresponds to step = k (since i = k for x_{t - i/m})
    for k in range(m * K_bar + 1):
        # find index of step = k
        idx = steps.index(k)
        w[2 * idx] -= wx[k]

    # subtract y coefficients for y_{t-j}: that corresponds to step = m*j
    for j in range(K_bar + 1):
        idx = steps.index(m * j)
        w[2 * idx + 1] -= wy[j]

    return w


# -----------------------------
# Table 3: fit MIDAS by minimizing PE variance (Eq. 3.6)
# -----------------------------
def fit_regular_midas_by_pe_variance(Sigma: np.ndarray, m: int, h: int, K_bar: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Choose MIDAS params (theta_y, theta_x, beta_y, beta_x) to minimize w' Σ w (Eq. 3.6).
    We solve betas in closed form (quadratic) for each (theta_y, theta_x).
    """
    steps = list(range(-m * h, m * K_bar + 1))

    def build_w_from_params(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        th_y = (u[0], u[1])
        th_x = (u[2], u[3])
        sy = exp_almon_weights(K_bar, th_y[0], th_y[1])
        sx = exp_almon_weights(m * K_bar, th_x[0], th_x[1])

        # w = w0 + beta_y * wy_part + beta_x * wx_part, but note w includes "-coeffs"
        # so it is affine in betas with negative signs.
        # We'll compute optimal betas by minimizing quadratic form in (beta_y, beta_x).
        # Construct base w0 (only +1 on y_{t+h})
        w0 = np.zeros(2 * len(steps))
        w0[1] = 1.0

        # basis vectors for beta_y and beta_x
        wy_basis = w_vector_from_coeffs(m, h, K_bar, wy=sy, wx=np.zeros(m * K_bar + 1)) - w0
        wx_basis = w_vector_from_coeffs(m, h, K_bar, wy=np.zeros(K_bar + 1), wx=sx) - w0

        # w(beta) = w0 + beta_y * wy_basis + beta_x * wx_basis
        return w0, wy_basis, wx_basis

    def obj(u: np.ndarray) -> float:
        w0, wyb, wxb = build_w_from_params(u)
        # minimize over betas analytically: min_b (w0 + B b)' Σ (w0 + B b)
        B = np.column_stack([wyb, wxb])  # dim x 2
        A = B.T @ Sigma @ B
        c = B.T @ Sigma @ w0
        # objective = w0'Σw0 + 2 b'c + b'A b
        # minimizer b* = -A^{-1} c
        try:
            b = -np.linalg.solve(A, c)
        except np.linalg.LinAlgError:
            return 1e50
        w = w0 + B @ b
        return float(w.T @ Sigma @ w)

    x0 = np.array([-0.2, 0.0, -0.2, 0.0])
    res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": OPT_TOL, "fatol": OPT_TOL, "maxiter": 80_000})

    # reconstruct optimal betas and coeffs
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
# Table 3: misspecified SS1 — choose one-factor parameters by minimizing PE variance under true Σ
# -----------------------------
def ss1_best_pe_variance_under_true_sigma(
    Sigma_true: np.ndarray,
    m: int,
    h: int,
    K_bar: int,
    d_fixed: float,
    rho_init: float,
) -> float:
    """
    Choose SS1 (one-factor) parameters to minimize PE variance under Sigma_true.
    We optimize over (rho, lam_y, lam_x, sig2_f, sig2_uy, sig2_ux) with d fixed (as table varies d).
    Then:
      - compute SS1 KF weights (wy_kf, wx_kf) from those parameters
      - build w_KF and compute w_KF' Sigma_true w_KF
    """
    def unpack(u: np.ndarray) -> OneFactorParams:
        rho = np.tanh(u[0])  # keep in (-1,1)
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
        # get KF weights (truncated)
        try:
            wy_kf, wx_kf = kalman_weights_by_impulses(p, h=h, K_bar=K_bar)
        except Exception:
            return 1e50
        w = w_vector_from_coeffs(m, h, K_bar, wy_kf, wx_kf)
        return float(w.T @ Sigma_true @ w)

    # initial guess: rho near table rho, loadings ~1, variances ~1
    x0 = np.array([np.arctanh(max(-0.99, min(0.99, rho_init))), 1.0, 1.0, 0.0, 0.0, 0.0])
    res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 20_000})
    return float(res.fun)


# -----------------------------
# Build Tables
# -----------------------------
def table1() -> Dict[str, pd.DataFrame]:
    """
    Table 1: one-factor DGP (Sec 3.1), L2 distances (Eq. 3.8), for m=3 and m=13,
    horizons h=1 and h=4, regular and multiplicative MIDAS.
    """
    out: Dict[str, pd.DataFrame] = {}

    for m in [3, 13]:
        for h in [1, 4]:
            reg_vals = np.zeros((len(D_GRID), len(RHO_GRID)))
            mul_vals = np.zeros((len(D_GRID), len(RHO_GRID)))

            for i_d, d in enumerate(D_GRID):
                for i_r, rho in enumerate(RHO_GRID):
                    # one-factor DGP baseline used in Sec 3.1: lam_y=lam_x=1, variances all 1, d1=d2=d
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
    """
    Table 2: two high-frequency series (Eq. 3.9), L2 distances like Table 1.
    Panels: m=3 and m=13, h=1, equal vs unequal noise variance.
    """
    out: Dict[str, pd.DataFrame] = {}
    h = 1

    for m in [3, 13]:
        for unequal in [False, True]:
            reg_vals = np.zeros((len(D_GRID), len(RHO_GRID)))
            mul_vals = np.zeros((len(D_GRID), len(RHO_GRID)))

            for i_d, d in enumerate(D_GRID):
                for i_r, rho in enumerate(RHO_GRID):
                    sig2_ux = np.array([1.0, 1.0]) if not unequal else np.array([1.0, 10.0])  # var(u2)=var(u3)/10 => u3 is 10x noisier

                    p = OneFactorParams(
                        m=m, rho=rho, d=d,
                        lam_y=1.0, lam_x=np.array([1.0, 1.0]),
                        sig2_f=1.0, sig2_uy=1.0, sig2_ux=sig2_ux
                    )

                    wy_kf, wx_kf_all = kalman_weights_by_impulses_multi_x(p, h=h, K_bar=K_BAR)
                    # For MIDAS-with-two-x’s, we allow separate polynomials per x-series and sum them.
                    # We fit by minimizing total L2 over y and BOTH x weight vectors.

                    def fit_regular_two_x() -> float:
                        def obj(u: np.ndarray) -> float:
                            # u: theta_y(2), then for x1 theta(2), for x2 theta(2)
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
                            # u: theta_y(2), then (outer,inner) for x1 (4), (outer,inner) for x2 (4)
                            th_y = (u[0], u[1])

                            th_o1 = (u[2], u[3])
                            th_i1 = (u[4], u[5])

                            th_o2 = (u[6], u[7])
                            th_i2 = (u[8], u[9])

                            sy = exp_almon_weights(K_BAR, th_y[0], th_y[1])

                            # build x shapes
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
    """
    Table 3: PE distance ratios MIDAS/SS1 (Eq. 3.7), DGP is two-factor (Sec 3.2),
    with rho1=rho2=rho and d1=d2=d.

    We implement:
      - True two-factor DGP loadings as in the paper’s Monte Carlo description:
          lam_y = [0.9, 0.1], lam_x = [0.1, 0.9]   (single x series)
        innovations variances = 1.
      - For each (m,h,d,rho):
          * build true Σ for upsilon_t
          * compute SS1 best PE variance under Σ (opt over one-factor params with d fixed)
          * compute MIDAS best PE variance under Σ (opt over MIDAS params)
          * ratio = PE_MIDAS / PE_SS1
    """
    out: Dict[str, pd.DataFrame] = {}

    # two-factor loadings (y and single x)
    a = np.array([0.9, 0.1])
    b = np.array([[0.1, 0.9]])  # one x series

    for m in [3, 13]:
        for h in [1, 4]:
            ratios = np.zeros((len(D_GRID), len(RHO_GRID)))
            for i_d, d in enumerate(D_GRID):
                for i_r, rho in enumerate(RHO_GRID):
                    # Build V_factor (unconditional contemporaneous covariance from factors at lag 0)
                    # Each factor AR(1) with innovation variance 1 => Var(f) = 1/(1-rho^2)
                    var_f = 1.0 / (1.0 - rho * rho) if abs(rho) < 1 else 1e12
                    # factor contribution to y*,x:
                    # y* = a1 f1 + a2 f2 ; x = b1 f1 + b2 f2
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

                    # SS1 best PE variance under true Sigma (misspecified one-factor)
                    pe_ss1 = ss1_best_pe_variance_under_true_sigma(
                        Sigma_true=Sigma_true, m=m, h=h, K_bar=K_BAR, d_fixed=d, rho_init=rho
                    )

                    # MIDAS best PE variance under true Sigma (regular MIDAS only, as in the table)
                    pe_midas, _, _ = fit_regular_midas_by_pe_variance(
                        Sigma=Sigma_true, m=m, h=h, K_bar=K_BAR
                    )

                    ratios[i_d, i_r] = pe_midas / pe_ss1 if pe_ss1 > 0 else np.nan

            out[f"Table3_m={m}_h={h}"] = pd.DataFrame(ratios, index=D_GRID, columns=RHO_GRID)

    return out


# -----------------------------
# Main: compute and print
# -----------------------------
def main():
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")

    print(f"Using K_BAR={K_BAR}\n")

    print("Computing Table 1...")
    t1 = table1()
    for k, df in t1.items():
        print("\n" + k)
        print(df)

    print("\nComputing Table 2...")
    t2 = table2()
    for k, df in t2.items():
       print("\n" + k)
       print(df)

    print("\nComputing Table 3...")
    t3 = table3()
    for k, df in t3.items():
        print("\n" + k)
        print(df)


if __name__ == "__main__":
    main()

