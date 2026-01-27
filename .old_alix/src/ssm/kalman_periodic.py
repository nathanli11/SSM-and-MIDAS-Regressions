from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .steady_state_riccati import PeriodicSteadyState, _sym


@dataclass
class KalmanFilterOutput:
    """
    Outputs for periodic Kalman filter run.
    """
    x_pred: np.ndarray   # (Tsteps, n_state)
    x_filt: np.ndarray   # (Tsteps, n_state)
    P_pred: np.ndarray   # (Tsteps, n_state, n_state)
    P_filt: np.ndarray   # (Tsteps, n_state, n_state)
    innov: List[np.ndarray]  # list of innovation vectors
    S: List[np.ndarray]      # list of innovation covariances


def kalman_filter_periodic(
    *,
    y_seq: List[np.ndarray],
    G: np.ndarray,
    Q_tilde: np.ndarray,
    Z_list: List[np.ndarray],
    H_list: Optional[List[np.ndarray]] = None,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    steady_state: Optional[PeriodicSteadyState] = None,
    jitter: float = 1e-12,
) -> KalmanFilterOutput:
    """
    Periodic Kalman filter with either:
      - steady_state gains (fast, replicates the "steady-state periodic Kalman filter"), or
      - time-varying Riccati recursion (if steady_state is None).

    Parameters
    ----------
    y_seq : list of np.ndarray
        Observations at each high-frequency step.
        The dimension must match Z_list[j].shape[0] where j = step % m.
    steady_state : PeriodicSteadyState | None
        If provided, uses its K[j], S[j], and P_pred[j] as steady-state.
        Otherwise, runs full time-varying recursion.

    Returns
    -------
    KalmanFilterOutput
    """
    m = len(Z_list)
    if m < 1:
        raise ValueError("Z_list length must be >= 1.")
    n_state = G.shape[0]

    Tsteps = len(y_seq)
    if Tsteps < 1:
        raise ValueError("y_seq must be non-empty.")

    if x0 is None:
        x0 = np.zeros((n_state,), dtype=float)
    if P0 is None:
        P0 = np.eye(n_state, dtype=float)

    x_pred = np.zeros((Tsteps, n_state), dtype=float)
    x_filt = np.zeros((Tsteps, n_state), dtype=float)
    P_pred = np.zeros((Tsteps, n_state, n_state), dtype=float)
    P_filt = np.zeros((Tsteps, n_state, n_state), dtype=float)

    innov_list: List[np.ndarray] = []
    S_list: List[np.ndarray] = []

    # Initialize
    x_prev = x0.copy()
    P_prev = _sym(P0)

    # If steady-state provided, we ignore P_prev except for initial transient:
    # simplest: start with steady-state P_pred for the first subperiod.
    if steady_state is not None:
        # align to subperiod 0
        P_prev = steady_state.P_pred[0].copy()

    for t in range(Tsteps):
        j = t % m
        Z = Z_list[j]
        y = np.asarray(y_seq[t], dtype=float).reshape(-1)

        if y.shape[0] != Z.shape[0]:
            raise ValueError(f"At step t={t}, obs dim {y.shape[0]} != Z_list[{j}].shape[0]={Z.shape[0]}.")

        # Predict
        x_t_pred = G @ x_prev
        P_t_pred = _sym(G @ P_prev @ G.T + Q_tilde)

        # Update
        if steady_state is not None:
            K = steady_state.K[j]
            S = steady_state.S[j]
            # Use steady-state S; still compute innovation from data.
            v = y - (Z @ x_t_pred)
            # Solve for S^{-1} v
            Sv = np.linalg.solve(_sym(S) + jitter*np.eye(S.shape[0]), v)
            x_t_filt = x_t_pred + K @ v

            # For P_filt, you can store steady-state updated covariance approx
            # Using Joseph stabilized update on P_t_pred with steady-state K:
            I = np.eye(n_state)
            P_t_filt = _sym((I - K @ Z) @ P_t_pred @ (I - K @ Z).T )

        else:
            ZP = Z @ P_t_pred
            S = _sym(ZP @ Z.T)
            if jitter > 0:
                S = S + jitter * np.eye(S.shape[0])
            K = np.linalg.solve(S, ZP).T
            v = y - (Z @ x_t_pred)
            x_t_filt = x_t_pred + K @ v
            I = np.eye(n_state)
            P_t_filt = _sym((I - K @ Z) @ P_t_pred)

        x_pred[t, :] = x_t_pred
        x_filt[t, :] = x_t_filt
        P_pred[t, :, :] = P_t_pred
        P_filt[t, :, :] = P_t_filt
        innov_list.append(v)
        S_list.append(S)

        x_prev = x_t_filt
        P_prev = P_t_filt

    return KalmanFilterOutput(
        x_pred=x_pred,
        x_filt=x_filt,
        P_pred=P_pred,
        P_filt=P_filt,
        innov=innov_list,
        S=S_list,
    )
