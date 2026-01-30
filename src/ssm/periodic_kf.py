import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from src.ssm.params import OneFactorParams, TwoFactorParams
from src.ssm.measurement import build_measurement_mats


# -----------------------------
# Global numerical knobs
# -----------------------------
RICCATI_MAX_ITERS = 50_000
RICCATI_TOL = 1e-12
# -----------------------------


# -----------------------------
# SSM-Filtre Kalman (Riccati)
# -----------------------------
# Classe pour stocker les matrices du KF périodique
@dataclass
class PeriodicKF:
    params: OneFactorParams
    P_pred: List[np.ndarray]    # P_{j|j-1}
    K_gain: List[np.ndarray]    # K_{j|j-1}
    Z_list: List[np.ndarray]    # Z_j
    H_list: List[np.ndarray]    #  var_cov des bruits pour j

    def G(self) -> np.ndarray:
        p = self.params
        # diag(rho, d, d, ..., d)
        return np.diag([p.rho] + [p.d] * (1 + p.n_x))

    def Q(self) -> np.ndarray:
        p = self.params
        return np.diag([p.sig2_f, p.sig2_uy] + list(p.sig2_ux))


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

def periodic_steady_state_kf_2f(p: TwoFactorParams):
    """
    Calcule les gains de Kalman périodiques (steady-state)
    pour le modèle à deux facteurs
    """
    m = p.m
    dim = p.dim_state

    # Matrice de transition
    G = np.diag([p.rho1, p.rho2, p.d, p.d])

    # Variance des innovations
    Q = np.diag([p.sig2_f1, p.sig2_f2, p.sig2_uy, p.sig2_ux])

    # Matrices de mesure selon la sous-période
    Z_list = []
    H_list = []

    for j in range(1, m + 1):
        if j < m:
            # x = f1 + u_x
            Z = np.array([[1, 0, 0, 1]])
            H = np.zeros((1, 1))
        else:
            # y = f1 + f2 + u_y
            # x = f1 + u_x
            Z = np.array([
                [1, 1, 1, 0],
                [1, 0, 0, 1]
            ])
            H = np.zeros((2, 2))

        Z_list.append(Z)
        H_list.append(H)

    # Initialisation Riccati
    P = np.eye(dim) * 10.0

    for _ in range(RICCATI_MAX_ITERS):
        P_old = P.copy()
        for j in range(m):
            Z = Z_list[j]
            H = H_list[j]

            S = Z @ P @ Z.T + H
            K = P @ Z.T @ np.linalg.inv(S)
            P = P - K @ Z @ P
            P = G @ P @ G.T + Q

        if np.max(np.abs(P - P_old)) < RICCATI_TOL:
            break

    # Gains de Kalman steady-state
    K_list = []
    for j in range(m):
        Z = Z_list[j]
        H = H_list[j]
        S = Z @ P @ Z.T + H
        K = P @ Z.T @ np.linalg.inv(S)
        K_list.append(K)

    return {
        "G": G,
        "Z_list": Z_list,
        "K_list": K_list,
    }


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

def run_periodic_kf_filter_2f(kf, y, x):
    """
    Applique le filtre de Kalman périodique à deux facteurs
    et retourne les états filtrés aux dates basse fréquence
    """
    G = kf["G"]
    Z_list = kf["Z_list"]
    K_list = kf["K_list"]

    m = len(Z_list)
    T_low = len(y)
    T_high = T_low * m

    a = np.zeros(G.shape[0])
    states_low = []

    low_idx = 0

    for t in range(T_high):
        j = t % m

        # Prediction
        a = G @ a

        # Observation
        if j < m - 1:
            y_obs = np.array([x[t, 0]])
        else:
            y_obs = np.array([y[low_idx], x[t, 0]])
            low_idx += 1

        Z = Z_list[j]
        K = K_list[j]

        # Innovation
        v = y_obs - Z @ a

        # Mise à jour
        a = a + K @ v

        # Sauvegarde à la fin de chaque période LF
        if j == m - 1:
            states_low.append(a.copy())

    return np.array(states_low)

