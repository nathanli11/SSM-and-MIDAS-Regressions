from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np


@dataclass(frozen=True)
class PeriodicLinearSystem:
    """
    Representation of the steady-state *linear* recursion induced by a periodic Kalman filter.

    We use the predict-state recursion (for a_{t|t-1}):

        a_{t+1|t} = F_j a_{t|t-1} + B_j y_t

    where j = t mod m (subperiod index).
    Here:
        F_j = G (I - K_j Z_j)
        B_j = G K_j

    If your Kalman recursion uses a different ordering convention, adjust F_j, B_j accordingly.

    Attributes
    ----------
    m : int
        Period length.
    F_list : list[np.ndarray]
        F_j matrices (n_state x n_state), j=0..m-1.
    B_list : list[np.ndarray]
        B_j matrices (n_state x d_j), j=0..m-1.
    Z_list : list[np (optional)
        measurement matrices; only used for convenience, not required for weight computation.
    """
    m: int
    F_list: List[np.ndarray]
    B_list: List[np.ndarray]
    Z_list: Optional[List[np.ndarray]] = None


def build_periodic_linear_system_from_steady_state(
    *,
    G: np.ndarray,
    K_list: List[np.ndarray],
    Z_list: List[np.ndarray],
) -> PeriodicLinearSystem:
    """
    Build (F_j, B_j) given steady-state periodic Kalman gains.

    Convention used:
      - predict: a_pred = G a_filt_prev
      - update:  a_filt = a_pred + K_j (y - Z_j a_pred)
      - next predict: a_pred_next = G a_filt

    Combining:
      a_pred_next = G (I - K_j Z_j) a_pred + G K_j y

    Therefore:
      F_j = G (I - K_j Z_j)
      B_j = G K_j
    """
    m = len(Z_list)
    if len(K_list) != m:
        raise ValueError("K_list and Z_list must have the same length m.")
    n_state = G.shape[0]
    I = np.eye(n_state)

    F_list: List[np.ndarray] = []
    B_list: List[np.ndarray] = []
    for j in range(m):
        Kj = K_list[j]
        Zj = Z_list[j]
        Fj = G @ (I - Kj @ Zj)
        Bj = G @ Kj
        F_list.append(Fj)
        B_list.append(Bj)

    return PeriodicLinearSystem(m=m, F_list=F_list, B_list=B_list, Z_list=Z_list)


def _phase_index(t: int, m: int) -> int:
    return t % m


def periodic_impulse_weights(
    *,
    system: PeriodicLinearSystem,
    C_target: np.ndarray,
    horizon_steps: int,
    K_lags: int,
    start_phase: int = 0,
) -> List[np.ndarray]:
    """
    Compute distributed-lag weights mapping past observations to a *target linear form*
    of the predicted state at horizon.

    We consider the recursion on predicted states:
        a_{t+1} = F_{j(t)} a_t + B_{j(t)} y_t
    where j(t) = (start_phase + t) mod m.

    Target at horizon h:
        target = C_target a_{h}
    with a_0 the predicted state at "time 0" (conditioning origin).
    We compute weights W_k such that:
        target ≈ sum_{k=0..K_lags-1} W_k y_{-k}
    where y_{-k} means the observation vector k steps in the past relative to time 0.

    Implementation detail:
    The weight on y_{-k} equals:
        W_k = C_target * ( Π_{s= -k+1}^{h-1} F_{j(s)} ) * B_{j(-k)}
    with the convention that an empty product is identity.

    Parameters
    ----------
    system : PeriodicLinearSystem
        Contains F_j and B_j.
    C_target : np.ndarray
        (d_target x n_state) matrix selecting the target from the predicted state a_h.
        For a scalar target, use shape (1, n_state).
    horizon_steps : int
        h in high-frequency steps (e.g., h=0 for nowcast of current predicted state,
        h=1 for 1-step-ahead in HF time, etc.).
    K_lags : int
        Number of past HF steps to include (0..K_lags-1).
    start_phase : int
        Phase index of time 0 (0..m-1). If your time origin is aligned so that
        time 0 corresponds to subperiod j=0, keep default 0.

    Returns
    -------
    W_list : list[np.ndarray]
        List of length K_lags. Each W_k has shape (d_target, d_{j(-k)}),
        where d_{j} is the observation dimension at that subperiod.
    """
    m = system.m
    F_list = system.F_list
    B_list = system.B_list

    C_target = np.asarray(C_target, dtype=float)
    d_target, n_state = C_target.shape
    if n_state != F_list[0].shape[0]:
        raise ValueError("C_target must have shape (d_target, n_state).")

    if horizon_steps < 0:
        raise ValueError("horizon_steps must be >= 0.")
    if K_lags < 1:
        raise ValueError("K_lags must be >= 1.")

    # Precompute forward products:
    # M_s = Π_{r=0}^{s-1} F_{j(r)} for s=0..horizon_steps
    # so that a_h = M_h a_0 + Σ M_{h-1-r} B_{j(r)} y_r
    # Here we need products from various past indices; easiest is compute product function.

    def phase_at(rel_t: int) -> int:
        return (start_phase + rel_t) % m

    def F_at(rel_t: int) -> np.ndarray:
        return F_list[phase_at(rel_t)]

    def B_at(rel_t: int) -> np.ndarray:
        return B_list[phase_at(rel_t)]

    # Utility: product of F from rel_t=a to rel_t=b inclusive:
    # Π_{s=a}^{b} F_s, in increasing time order.
    def prod_F(a: int, b: int) -> np.ndarray:
        if a > b:
            return np.eye(n_state)
        M = np.eye(n_state)
        for s in range(a, b + 1):
            M = F_at(s) @ M
        return M

    # Weight for lag k uses y_{-k} (a past observation).
    # The observation y_{-k} enters via B_{-k} at time rel_t = -k,
    # then is propagated forward through F_{-k+1}, ..., F_{h-1}.
    W_list: List[np.ndarray] = []
    for k in range(K_lags):
        rel_t_inj = -k
        # Propagate from (-k+1) .. (horizon_steps-1)
        M = prod_F(rel_t_inj + 1, horizon_steps - 1)
        Wk = C_target @ (M @ B_at(rel_t_inj))
        W_list.append(Wk)

    return W_list


def flatten_weights(
    W_list: List[np.ndarray],
    obs_dims_by_phase: Sequence[int],
    *,
    system_m: int,
    start_phase: int = 0,
) -> np.ndarray:
    """
    Optional helper: embed variable-dimension weights into a fixed-width matrix
    using a block layout by phase.

    Suppose each phase j has observation dimension d_j. For each lag k we have W_k
    of shape (d_target, d_{phase(-k)}). This function maps them into a fixed-width
    array of shape (d_target, K_lags * sum_j d_j) with zeros where not applicable.

    This is useful if you want to compare to a single long MIDAS weight vector.
    """
    m = system_m
    if len(obs_dims_by_phase) != m:
        raise ValueError("obs_dims_by_phase must have length m.")

    K_lags = len(W_list)
    d_target = W_list[0].shape[0]
    total_d = int(np.sum(obs_dims_by_phase))
    out = np.zeros((d_target, K_lags * total_d), dtype=float)

    # Offsets per phase within one "super-vector"
    offsets = np.cumsum([0] + list(obs_dims_by_phase[:-1]))

    for k, Wk in enumerate(W_list):
        phase = (start_phase - k) % m
        d_phase = obs_dims_by_phase[phase]
        if Wk.shape[1] != d_phase:
            raise ValueError("Wk dimension mismatch with obs_dims_by_phase.")
        col0 = k * total_d + offsets[phase]
        out[:, col0:col0 + d_phase] = Wk

    return out
