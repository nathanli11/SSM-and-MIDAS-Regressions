from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.optimize import minimize

from midas.almon import exp_almon_weights


@dataclass(frozen=True)
class ADLMidasResult:
    """
    Result object for multiplicative ADL-MIDAS.

    Paper-style structure (eq. (2.25)-(2.26)):

        y_{t+h} = beta0
                + beta_y * sum_{j=0..Ky-1} w_j(theta_y) * y_{t-j}
                + beta_x * sum_{j=0..Kx-1} w_j(theta_xLF) * x_agg_{t-j}
                + e_{t+h}

    with within-period aggregator:
        x_agg_t = sum_{k=0..m-1} v_k(theta_xHF) * x_{t - k/m}

    Here:
      - theta_y = (theta_y1, theta_y2) for LF y-lag weights (Almon)
      - theta_xLF = (theta_xLF1, theta_xLF2) for LF lag weights on x_agg
      - theta_xHF = (theta_xHF1, theta_xHF2) for within-period weights (length m)

    All weights are normalized to sum to 1.
    """
    beta0: float
    beta_y: float
    beta_x: float

    theta_y1: float
    theta_y2: float
    theta_xLF1: float
    theta_xLF2: float
    theta_xHF1: float
    theta_xHF2: float

    w_y: np.ndarray        # (Ky,)
    w_xLF: np.ndarray      # (Kx,)
    w_xHF: np.ndarray      # (m,)

    y_agg: np.ndarray      # (T_eff,)
    x_agg: np.ndarray      # (T_eff,)
    y: np.ndarray          # (T_eff,)
    y_hat: np.ndarray      # (T_eff,)
    resid: np.ndarray      # (T_eff,)
    sse: float
    success: bool
    message: str


def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(-1)


def _build_lf_lags(y: np.ndarray, K: int) -> np.ndarray:
    """
    Build LF lag matrix: for t index, row is [y_t, y_{t-1}, ..., y_{t-K+1}]
    Most recent first.
    """
    y = _as_1d(y)
    T = y.shape[0]
    if K < 1 or K > T:
        raise ValueError("K must satisfy 1 <= K <= len(y).")
    X = np.zeros((T - K + 1, K), dtype=float)
    for i in range(T - K + 1):
        window = y[i:i + K]
        X[i, :] = window[::-1]
    return X


def _build_within_period_aggregator(
    x_hf: np.ndarray,
    m: int,
) -> np.ndarray:
    """
    Convert HF series x_hf (length T_lf*m) into "end-of-period aligned" blocks:
    For LF period t (1..T_lf), collect within-period HF values in reverse time order:
        [x(t), x(t-1/m), ..., x(t-(m-1)/m)].

    Returns Xblock of shape (T_lf, m) with most recent first.
    """
    x_hf = _as_1d(x_hf)
    if x_hf.shape[0] % m != 0:
        raise ValueError("len(x_hf) must be a multiple of m.")
    T_lf = x_hf.shape[0] // m
    X = np.zeros((T_lf, m), dtype=float)
    for t in range(1, T_lf + 1):
        idx_end = t * m - 1
        for k in range(m):
            X[t - 1, k] = x_hf[idx_end - k]
    return X


def fit_adl_midas_multiplicative(
    *,
    y_lf: np.ndarray,
    x_hf: np.ndarray,
    m: int,
    Ky: int,
    Kx: int,
    h: int = 0,
    include_intercept: bool = True,
    theta_y_init: Tuple[float, float] = (-0.1, -0.01),
    theta_xLF_init: Tuple[float, float] = (-0.1, -0.01),
    theta_xHF_init: Tuple[float, float] = (-0.1, -0.01),
    method: str = "L-BFGS-B",
) -> ADLMidasResult:
    """
    Estimate multiplicative ADL-MIDAS by nonlinear least squares.

    Steps (high-level)
    ------------------
    1) Build within-period HF blocks of length m for each LF t.
    2) For candidate (theta_y, theta_xLF, theta_xHF):
       - compute w_xHF (length m) and build x_agg_t (LF series)
       - build LF lag matrices for y and x_agg (length Ky and Kx)
       - compute y_agg_t = sum w_y * y-lags
         and xLF_agg_t = sum w_xLF * x_agg-lags
       - regress y_{t+h} on [1, y_agg_t, xLF_agg_t] (OLS conditional on thetas)
    3) Optimize thetas to minimize SSE.

    This mirrors the paper's multiplicative setup (2.25)-(2.26). :contentReference[oaicite:6]{index=6}
    """
    y_lf = _as_1d(y_lf)
    x_hf = _as_1d(x_hf)

    if m < 1:
        raise ValueError("m must be >= 1.")
    if h < 0:
        raise ValueError("h must be >= 0.")

    # Build within-period blocks, then x_agg_t depends on theta_xHF
    Xblock = _build_within_period_aggregator(x_hf, m=m)  # (T_lf, m)
    T_lf = y_lf.shape[0]
    if Xblock.shape[0] != T_lf:
        raise ValueError("y_lf length and x_hf length/m mismatch.")

    # We will need lags Ky and Kx and horizon h.
    # We'll align everything on LF index t where y_{t+h} is available.
    # Minimal t for y-lags: t >= Ky
    # Minimal t for x_agg lags: t >= Kx
    # And t+h <= T_lf
    t_min = max(Ky, Kx)
    t_max = T_lf - h
    if t_max < t_min:
        raise ValueError("Not enough data after accounting for Ky, Kx, and h.")

    t_used = np.arange(t_min, t_max + 1)  # LF indices (1-based)

    def objective(theta: np.ndarray) -> float:
        th_y1, th_y2 = float(theta[0]), float(theta[1])
        th_xLF1, th_xLF2 = float(theta[2]), float(theta[3])
        th_xHF1, th_xHF2 = float(theta[4]), float(theta[5])

        # weights
        w_y = exp_almon_weights(th_y1, th_y2, Ky, start_at_one=False)
        w_xLF = exp_almon_weights(th_xLF1, th_xLF2, Kx, start_at_one=False)
        w_xHF = exp_almon_weights(th_xHF1, th_xHF2, m, start_at_one=False)

        # within-period aggregation: x_agg_t = Xblock[t] @ w_xHF
        x_agg_full = Xblock @ w_xHF  # (T_lf,)

        # Build LF lag matrices (most recent first)
        Ylags_full = _build_lf_lags(y_lf, Ky)         # rows correspond to t=Ky..T_lf
        Xlags_full = _build_lf_lags(x_agg_full, Kx)   # rows correspond to t=Kx..T_lf

        # Map LF t (1-based) to row in lag-matrices:
        # For Ylags_full: row index = (t - Ky)
        # For Xlags_full: row index = (t - Kx)
        y_agg = np.zeros(t_used.size, dtype=float)
        x_lf_agg = np.zeros(t_used.size, dtype=float)
        y_target = np.zeros(t_used.size, dtype=float)

        for i, t in enumerate(t_used):
            y_agg[i] = Ylags_full[t - Ky, :] @ w_y
            x_lf_agg[i] = Xlags_full[t - Kx, :] @ w_xLF
            y_target[i] = y_lf[(t - 1) + h]

        # OLS for betas conditional on thetas
        if include_intercept:
            Xreg = np.column_stack([np.ones_like(y_agg), y_agg, x_lf_agg])
        else:
            Xreg = np.column_stack([y_agg, x_lf_agg])

        beta_hat, *_ = np.linalg.lstsq(Xreg, y_target, rcond=None)
        y_hat = Xreg @ beta_hat
        resid = y_target - y_hat
        return float(resid @ resid)

    x0 = np.array(
        [theta_y_init[0], theta_y_init[1],
         theta_xLF_init[0], theta_xLF_init[1],
         theta_xHF_init[0], theta_xHF_init[1]],
        dtype=float
    )

    res = minimize(objective, x0=x0, method=method)

    # Recompute everything at optimum for output
    th_y1, th_y2, th_xLF1, th_xLF2, th_xHF1, th_xHF2 = map(float, res.x)

    w_y = exp_almon_weights(th_y1, th_y2, Ky, start_at_one=False)
    w_xLF = exp_almon_weights(th_xLF1, th_xLF2, Kx, start_at_one=False)
    w_xHF = exp_almon_weights(th_xHF1, th_xHF2, m, start_at_one=False)

    x_agg_full = Xblock @ w_xHF

    Ylags_full = _build_lf_lags(y_lf, Ky)
    Xlags_full = _build_lf_lags(x_agg_full, Kx)

    y_agg = np.zeros(t_used.size, dtype=float)
    x_lf_agg = np.zeros(t_used.size, dtype=float)
    y_target = np.zeros(t_used.size, dtype=float)
    for i, t in enumerate(t_used):
        y_agg[i] = Ylags_full[t - Ky, :] @ w_y
        x_lf_agg[i] = Xlags_full[t - Kx, :] @ w_xLF
        y_target[i] = y_lf[(t - 1) + h]

    if include_intercept:
        Xreg = np.column_stack([np.ones_like(y_agg), y_agg, x_lf_agg])
    else:
        Xreg = np.column_stack([y_agg, x_lf_agg])

    beta_hat, *_ = np.linalg.lstsq(Xreg, y_target, rcond=None)
    y_hat = Xreg @ beta_hat
    resid = y_target - y_hat
    sse = float(resid @ resid)

    if include_intercept:
        beta0, beta_y, beta_x = float(beta_hat[0]), float(beta_hat[1]), float(beta_hat[2])
    else:
        beta0, beta_y, beta_x = 0.0, float(beta_hat[0]), float(beta_hat[1])

    return ADLMidasResult(
        beta0=beta0,
        beta_y=beta_y,
        beta_x=beta_x,
        theta_y1=th_y1,
        theta_y2=th_y2,
        theta_xLF1=th_xLF1,
        theta_xLF2=th_xLF2,
        theta_xHF1=th_xHF1,
        theta_xHF2=th_xHF2,
        w_y=w_y,
        w_xLF=w_xLF,
        w_xHF=w_xHF,
        y_agg=y_agg,
        x_agg=x_lf_agg,
        y=y_target,
        y_hat=y_hat,
        resid=resid,
        sse=sse,
        success=bool(res.success),
        message=str(res.message),
    )
