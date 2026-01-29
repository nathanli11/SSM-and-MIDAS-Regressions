import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from params import OneFactorParams
from measurement import build_measurement_mats


# -----------------------------
# Global numerical knobs
# -----------------------------
RICCATI_MAX_ITERS = 50_000
RICCATI_TOL = 1e-12
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