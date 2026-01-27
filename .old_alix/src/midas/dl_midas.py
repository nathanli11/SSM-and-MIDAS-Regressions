from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from midas.almon import exp_almon_weights


@dataclass(frozen=True)
class DLMidasResult:
    """
    Result object for DL-MIDAS estimation.

    Model:
        y_{t+h} = beta0 + beta1 * sum_{j=0..K-1} w_j(theta) * x_{t - j/m} + e_{t+h}

    Notes
    -----
    - w_j(theta) are normalized exponential Almon weights (paper eq. (2.19)).
    - This is a nonlinear least squares problem in (beta0, beta1, theta1, theta2).
    """
    beta0: float
    beta1: float
    theta1: float
    theta2: float
    weights: np.ndarray            # (K,)
    x_agg: np.ndarray              # (T_eff,)
    y: np.ndarray                  # (T_eff,)
    y_hat: np.ndarray              # (T_eff,)
    resid: np.ndarray              # (T_eff,)
    sse: float
    success: bool
    message: str


def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(-1)


def _build_hf_lags_aligned(
    x_hf: np.ndarray,
    m: int,
    K: int,
    h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build HF lag matrix aligned for predicting y_{t+h} at low-frequency t.

    We assume x_hf is stacked in HF time, with exactly m observations per LF period.
    Let LF index t correspond to end-of-period time t (like quarter end).

    We need, for each LF t, the vector [x(t), x(t-1/m), ..., x(t-(K-1)/m)].

    Alignment choice:
    - Define the "information time" at the end of LF period t (i.e., last HF obs in that LF).
    - Then the most recent HF value for period t is x_hf[t*m - 1].

    For forecasting y_{t+h}, we use x information up to LF t (not beyond).
    This matches the timing convention of standard MIDAS nowcast/forecast setups.

    Returns
    -------
    Xlags : (T_eff, K) where column 0 is most recent, column K-1 oldest
    t_index : (T_eff,) LF indices used (so caller can align y)
    """
    x_hf = _as_1d(x_hf)
    if m < 1:
        raise ValueError("m must be >= 1.")
    if K < 1:
        raise ValueError("K must be >= 1.")
    if h < 0:
        raise ValueError("h must be >= 0.")

    T_hf = x_hf.shape[0]
    if T_hf % m != 0:
        raise ValueError("len(x_hf) must be multiple of m.")
    T_lf = T_hf // m

    # We need at least K HF observations before each t.
    # End-of-period HF index for LF t (1-based t): idx_end = t*m - 1 (0-based)
    # For LF t = 1, idx_end = m-1. Need idx_end - (K-1) >= 0 => t*m >= K
    t_min = int(np.ceil(K / m))  # smallest LF t that has K HF lags
    # For y_{t+h} we need y available up to t+h <= T_lf  => t <= T_lf - h
    t_max = T_lf - h
    if t_max < t_min:
        raise ValueError("Not enough data after accounting for K and h.")

    t_used = np.arange(t_min, t_max + 1)  # LF t (1-based)
    X = np.zeros((t_used.size, K), dtype=float)

    for row, t in enumerate(t_used):
        idx_end = t * m - 1
        # collect most recent first
        for j in range(K):
            X[row, j] = x_hf[idx_end - j]

    return X, t_used


def fit_dl_midas(
    *,
    y_lf: np.ndarray,
    x_hf: np.ndarray,
    m: int,
    K: int,
    h: int = 0,
    include_intercept: bool = True,
    theta_init: Tuple[float, float] = (-0.1, -0.01),
    method: str = "L-BFGS-B",
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> DLMidasResult:
    """
    Estimate DL-MIDAS by nonlinear least squares.

    Parameters
    ----------
    y_lf : (T_lf,)
        Low-frequency target series.
    x_hf : (T_lf*m,)
        High-frequency regressor stacked in time.
    m : int
        HF observations per LF period.
    K : int
        Number of HF lags in the MIDAS polynomial.
    h : int
        Forecast horizon in LF units (0=nowcast/current, 1=one-period ahead, ...).
    include_intercept : bool
        Whether to include beta0.
    theta_init : (theta1, theta2)
        Initial Almon parameters.
    bounds : optional bounds for (theta1, theta2). If None, unbounded.

    Returns
    -------
    DLMidasResult
    """
    y_lf = _as_1d(y_lf)
    Xlags, t_used = _build_hf_lags_aligned(x_hf=x_hf, m=m, K=K, h=h)

    # Align y: we want y_{t+h} for each t in t_used
    # y_lf is assumed indexed 1..T_lf in concept; python 0..T_lf-1
    y = np.array([y_lf[(t - 1) + h] for t in t_used], dtype=float)

    def objective(theta: np.ndarray) -> float:
        th1, th2 = float(theta[0]), float(theta[1])
        w = exp_almon_weights(th1, th2, K, start_at_one=False)  # j=0..K-1
        x_agg = Xlags @ w  # (T_eff,)

        if include_intercept:
            # OLS for beta given theta: y = beta0 + beta1*x_agg
            Xreg = np.column_stack([np.ones_like(x_agg), x_agg])
        else:
            Xreg = x_agg[:, None]

        beta_hat, *_ = np.linalg.lstsq(Xreg, y, rcond=None)
        y_hat = Xreg @ beta_hat
        resid = y - y_hat
        return float(resid @ resid)  # SSE

    x0 = np.array(theta_init, dtype=float)
    if bounds is None:
        opt_bounds = None
    else:
        opt_bounds = list(bounds)

    res = minimize(objective, x0=x0, method=method, bounds=opt_bounds)

    th1, th2 = float(res.x[0]), float(res.x[1])
    w = exp_almon_weights(th1, th2, K, start_at_one=False)
    x_agg = Xlags @ w

    if include_intercept:
        Xreg = np.column_stack([np.ones_like(x_agg), x_agg])
    else:
        Xreg = x_agg[:, None]

    beta_hat, *_ = np.linalg.lstsq(Xreg, y, rcond=None)
    if include_intercept:
        beta0, beta1 = float(beta_hat[0]), float(beta_hat[1])
    else:
        beta0, beta1 = 0.0, float(beta_hat[0])

    y_hat = Xreg @ beta_hat
    resid = y - y_hat
    sse = float(resid @ resid)

    return DLMidasResult(
        beta0=beta0,
        beta1=beta1,
        theta1=th1,
        theta2=th2,
        weights=w,
        x_agg=x_agg,
        y=y,
        y_hat=y_hat,
        resid=resid,
        sse=sse,
        success=bool(res.success),
        message=str(res.message),
    )
