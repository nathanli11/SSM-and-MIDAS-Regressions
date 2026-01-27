from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class PeriodicSteadyState:
    """
    Periodic steady-state solution for Kalman filter.

    For each subperiod j = 1..m (index 0..m-1):
      - P_pred[j] = Var(alpha_t | info up to t-1 step)  == P_{j|j-1}
      - K[j]      = Kalman gain
      - S[j]      = innovation covariance: Z P Z'
      - A[j]      = closed-loop transition: G - K Z G   (useful for weights)
    """
    P_pred: List[np.ndarray]
    K: List[np.ndarray]
    S: List[np.ndarray]
    A: List[np.ndarray]


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def solve_periodic_steady_state(
    *,
    G: np.ndarray,
    Q_tilde: np.ndarray,
    Z_list: List[np.ndarray],
    tol: float = 1e-12,
    max_iter: int = 50_000,
    jitter: float = 1e-12,
    verbose: bool = False,
) -> PeriodicSteadyState:
    """
    Solve periodic steady-state Riccati by fixed-point iteration over one full cycle.

    Model:
        alpha_t = G alpha_{t-1} + w_t,   w_t ~ N(0, Q_tilde)
        y_t^j   = Z_j alpha_t + v_t^j,   v_t^j ~ N(0, H_j)

    Notes:
    - In our paper-style builder, Q_tilde = R Q R' and H_j = 0 typically.
    - We iterate over the m subperiods repeatedly until all P_pred[j] converge.

    Returns
    -------
    PeriodicSteadyState with lists of length m.
    """
    m = len(Z_list)
    if m < 1:
        raise ValueError("Z_list must have length >= 1.")
    n_state = G.shape[0]
    if G.shape != (n_state, n_state):
        raise ValueError("G must be square.")
    if Q_tilde.shape != (n_state, n_state):
        raise ValueError("Q_tilde must be (n_state x n_state).")

    # Initialize P_pred[j] (PSD-ish)
    P_pred = [np.eye(n_state, dtype=float) for _ in range(m)]

    def riccati_step(P: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given P_pred at subperiod j, compute:
          S = Z P Z'
          K = P Z' S^{-1}
          P_upd = (I - K Z) P (Joseph optional)
        """
        ZP = Z @ P
        S = ZP @ Z.T
        S = _sym(S)
        # Stabilize S
        if jitter > 0:
            S = S + jitter * np.eye(S.shape[0])

        # Solve for K' via linear system: S X = (Z P)' = P Z'
        # K = P Z' S^{-1} => K' = S^{-1} Z P
        # We'll compute K = (solve(S, ZP)).T
        K = np.linalg.solve(S, ZP).T

        I = np.eye(P.shape[0])
        P_upd = (I - K @ Z) @ P         # P_{j|j} = (I - K Z) P_{j|j-1} (update part)
        P_upd = _sym(P_upd)
        return K, S, P_upd

    # Iterate
    for it in range(1, max_iter + 1):
        P_old = [P.copy() for P in P_pred]

        # Cycle through subperiods j=1..m
        for j in range(m):
            Z = Z_list[j]

            # Update at j using observation at that subperiod
            K_j, S_j, P_upd = riccati_step(P_pred[j], Z)

            # Predict to next subperiod (j+1 mod m)
            P_next = G @ P_upd @ G.T + Q_tilde      #P_{j+1∣j} = GP_{j|j} G' + Q
            P_next = _sym(P_next)

            next_j = (j + 1) % m
            P_pred[next_j] = P_next

        # Convergence check (max norm over all j)
        diffs = [np.max(np.abs(P_pred[j] - P_old[j])) for j in range(m)]
        err = float(np.max(diffs))

        if verbose and (it == 1 or it % 100 == 0):
            print(f"[Riccati] iter={it}, max|ΔP|={err:.3e}")

        if err < tol:
            break
    else:
        raise RuntimeError(f"Periodic Riccati did not converge after max_iter={max_iter}.")

    # Once converged, compute K,S,A for each j from final P_pred
    K_list: List[np.ndarray] = []
    S_list: List[np.ndarray] = []
    A_list: List[np.ndarray] = []

    for j in range(m):
        Z = Z_list[j]
        K_j, S_j, _ = riccati_step(P_pred[j], Z)
        A_j = G - K_j @ Z @ G
        K_list.append(K_j)
        S_list.append(S_j)
        A_list.append(A_j)

    return PeriodicSteadyState(P_pred=P_pred, K=K_list, S=S_list, A=A_list)
