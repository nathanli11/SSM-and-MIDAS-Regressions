import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

RICCATI_MAX_ITERS = 50_000
RICCATI_TOL = 1e-12

# Classe model à un facteur avec KF périodique
@dataclass
class OneFactorParams:
    m: int = 3
    n_x: int = 1

    # Coeff (gamma1=gamma2=1)
    lam_y: float = 1.0
    lam_x: np.ndarray = None

    # AR 
    rho:float = 0.5
    d: float = 0.0

    # Variances des innovations
    sig2_f: float = 1.0
    sig2_uy: float = 1.0
    sig2_ux: np.ndarray = None  

    @property
    def dim_state(self) -> int:
        # state = [f, u_y, u_x1,...,u_xn]
        return 2 + self.n_x

# Classe pour stocker les matrices du KF périodique
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


def exp_almon_weights(K: int, theta1: float, theta2: float) -> np.ndarray:
    """
        Fonction de poids exponentiel d'Almon
    """
    j = np.arange(K + 1, dtype=float)
    z = theta1 * j + theta2 * j * j

    # Stabilisation numérique
    z = z - np.max(z)

    a = np.exp(z)
    s = a.sum()

    # Gestion des cas infinis et négatifs
    if not np.isfinite(s) or s <= 0:
        out = np.zeros(K + 1)
        out[0] = 1.0
        return out

    return a / s

def build_measurement_mats(p: OneFactorParams) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Construit les matrices de mesure périodiques Z_j et H_j pour j=1..m
    selon le modèle à un facteur avec observations à fréquence mixte.
    Retourne:
      Z_list: Liste de Z_j matrices
      H_list: Liste de H_j matrices
    """
    n_x = p.n_x
    dim = p.dim_state

    Z_list: List[np.ndarray] = []
    H_list: List[np.ndarray] = []

    # j=1..m-1: x seulement
    for _ in range(p.m - 1):
        Z = np.zeros((n_x, dim))
        # x lignes
        Z[:, 0] = p.lam_x
        for i in range(n_x):
            Z[i, 2 + i] = 1.0
        Z_list.append(Z)
        H_list.append(np.zeros((n_x, n_x)))

    # j=m: y + x's
    Zm = np.zeros((1 + n_x, dim))

    Zm[0, 0] = p.lam_y
    Zm[0, 1] = 1.0

    Zm[1:, 0] = p.lam_x
    for i in range(n_x):
        Zm[1 + i, 2 + i] = 1.0
    Z_list.append(Zm)
    H_list.append(np.zeros((1 + n_x, 1 + n_x)))

    return Z_list, H_list

def periodic_steady_state_kf(p: OneFactorParams) -> PeriodicKF:
    """
    Variance P_{j|j-1} et gains K_{j|j-1} périodiques en régime permanent via itération de Riccati (Eq. 2.8).
    """
    Z_list, H_list = build_measurement_mats(p)
    G = np.diag([p.rho] + [p.d] * (1 + p.n_x))
    Q = np.diag([p.sig2_f, p.sig2_uy] + list(p.sig2_ux))

    # Init P_{1|0}
    P = np.eye(p.dim_state)

    # itererations de Riccati
    P_pred = [np.zeros_like(P) for _ in range(p.m)]
    K_gain = [np.zeros((p.dim_state, Z_list[j].shape[0])) for j in range(p.m)]

    for it in range(RICCATI_MAX_ITERS):
        P_old = P.copy()
        #j=1...m
        for j in range(p.m):
            # prediction
            Pp = G @ P @ G.T + Q
            Z = Z_list[j]
            H = H_list[j]
            S = Z @ Pp @ Z.T + H
            # gain
            K = Pp @ Z.T @ np.linalg.inv(S)
            # maj
            P = (np.eye(p.dim_state) - K @ Z) @ Pp

            P_pred[j] = Pp
            K_gain[j] = K

        # check convergence
        diff = np.max(np.abs(P - P_old))
        if diff < RICCATI_TOL:
            break
    else:
        raise RuntimeError("Riccati ne converge pas")

    return PeriodicKF(params=p, P_pred=P_pred, K_gain=K_gain, Z_list=Z_list, H_list=H_list)

def run_periodic_kf_filter(
    kf: PeriodicKF,
    obs_y: np.ndarray,
    obs_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run le filtre de kalmann périodique en régime permanent (avec gains fixes de kf) sur un panel haute fréquence.
    """
    p = kf.params
    m = p.m
    n_x = p.n_x
    dim = p.dim_state

    T_high = obs_x.shape[0]
    T_low = obs_y.shape[0]
    assert T_high == T_low * m

    G = kf.G()

    # allocation
    state_filt_high = np.zeros((T_high, dim))
    state_filt_low = np.zeros((T_low, dim))

    a = np.zeros(dim)     # etat initial
    P = np.eye(dim)       # variance initiale

    low_idx = 0
    for t_high in range(T_high):
        j = (t_high % m) + 1  # 1..m
        jj = j - 1            # 0..m-1 index

        # prediction
        a = G @ a

        # maj avec les observations
        if j < m:
            y_obs = obs_x[t_high, :]  
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
    Stock variable cas: y_{t+h} = y*_{t+h} = lam_y f_{t+h} + u_y,t+h.

    """
    rho_pow = p.rho ** (p.m * h)
    d_pow = p.d ** (p.m * h)
    f_t = state_at_t[0]
    uy_t = state_at_t[1]
    return p.lam_y * rho_pow * f_t + d_pow * uy_t
