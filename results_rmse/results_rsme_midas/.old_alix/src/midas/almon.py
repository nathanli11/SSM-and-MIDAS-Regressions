from __future__ import annotations

from typing import Tuple
import numpy as np


def exp_almon_weights(
    theta1: float,
    theta2: float,
    K: int,
    *,
    start_at_one: bool = True,
) -> np.ndarray:
    """
    Exponential Almon lag weights (normalized), common in MIDAS.

    w_k(theta) = exp(theta1 * k + theta2 * k^2) / sum_{j} exp(theta1*j + theta2*j^2)

    Parameters
    ----------
    theta1, theta2 : float
        Almon parameters.
    K : int
        Number of lags.
    start_at_one : bool
        If True, k runs 1..K (common in MIDAS papers).
        If False, k runs 0..K-1.

    Returns
    -------
    w : np.ndarray shape (K,)
        Nonnegative weights summing to 1.
    """
    if K < 1:
        raise ValueError("K must be >= 1.")
    if start_at_one:
        k = np.arange(1, K + 1, dtype=float)
    else:
        k = np.arange(0, K, dtype=float)

    x = theta1 * k + theta2 * k**2

    # numerical stabilization: subtract max to avoid overflow
    x = x - np.max(x)
    w_unnorm = np.exp(x)
    w = w_unnorm / np.sum(w_unnorm)
    return w


def exp_almon_weights_and_grad(
    theta1: float,
    theta2: float,
    K: int,
    *,
    start_at_one: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return weights and gradient wrt (theta1, theta2).
    Gradient is useful if you later do custom optimizers.

    Returns
    -------
    w : (K,)
    grad : (K, 2) where grad[k,0]=dw_k/dtheta1, grad[k,1]=dw_k/dtheta2
    """
    w = exp_almon_weights(theta1, theta2, K, start_at_one=start_at_one)
    if start_at_one:
        k = np.arange(1, K + 1, dtype=float)
    else:
        k = np.arange(0, K, dtype=float)

    # Softmax gradient: dw_i/dtheta = w_i (x_i' - sum_j w_j x_j')
    x1 = k
    x2 = k**2
    mu1 = np.sum(w * x1)
    mu2 = np.sum(w * x2)

    grad1 = w * (x1 - mu1)
    grad2 = w * (x2 - mu2)
    grad = np.column_stack([grad1, grad2])
    return w, grad


def build_hf_lag_matrix(x: np.ndarray, K: int) -> np.ndarray:
    """
    Construct lagged HF regressor matrix with K lags.

    Given x of length T, returns X of shape (T-K+1, K) with:
        X[t, :] = [x[t+K-1], x[t+K-2], ..., x[t]]

    So the first column is the most recent lag (lag 0 in that trimmed index),
    and last column is the oldest (lag K-1).

    This orientation is convenient when you later do X @ w.

    Parameters
    ----------
    x : (T,)
    K : int

    Returns
    -------
    X : (T-K+1, K)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    T = x.shape[0]
    if K < 1 or K > T:
        raise ValueError("K must satisfy 1 <= K <= len(x).")

    X = np.zeros((T - K + 1, K), dtype=float)
    for t in range(T - K + 1):
        window = x[t:t + K]
        X[t, :] = window[::-1]  # reverse => most recent first
    return X
