from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class PeriodicSSM:
    """
    Periodic state space model container.

    State:
        alpha_t = [F_t, F_{t-1/m}, ..., F_{t-(p-1)/m}, u_t]
        where F_t is nf-dim factor, u_t is n-dim measurement error vector.

    Transition:
        alpha_t = G alpha_{t-1/m} + R zeta_t
        zeta_t = [eta_t, eps_t]
        eta_t ~ N(0, Sigma_eta), eps_t ~ N(0, diag(Sigma_eps))

    Measurement (periodic):
        Y_t^j = Z_j alpha_t   (H_j = 0, because u_t is in the state)
    """
    m: int
    nf: int
    n: int
    p: int

    G: np.ndarray
    R: np.ndarray
    Q: np.ndarray  # covariance of zeta_t

    Z_list: List[np.ndarray]  # length m, for j=1..m (index 0..m-1)
    #H_list: List[np.ndarray]  # length m, usually zeros
    obs_names_list: List[List[str]]  # names of observables in each subperiod


def _block_diag(*blocks: np.ndarray) -> np.ndarray:
    if len(blocks) == 0:
        return np.zeros((0, 0))
    shapes = [b.shape for b in blocks]
    r = sum(s[0] for s in shapes)
    c = sum(s[1] for s in shapes)
    out = np.zeros((r, c), dtype=float)
    rr = 0
    cc = 0
    for b in blocks:
        br, bc = b.shape
        out[rr:rr+br, cc:cc+bc] = b
        rr += br
        cc += bc
    return out


def build_mixedfreq_dynamic_factor_ar1_me(
    *,
    m: int,
    Phi_list: List[np.ndarray],
    Sigma_eta: np.ndarray,
    Gamma: np.ndarray,
    d_u: np.ndarray,
    sigma_eps: np.ndarray,
    include_low_freq: bool = True,
    low_freq_name: str = "y",
    high_freq_names: Optional[List[str]] = None,
) -> PeriodicSSM:
    """
    Builder for a mixed-frequency dynamic factor model close to Bai-Ghysels-Wright (2013).

    Factors:
        F_t = Phi_1 F_{t-1/m} + ... + Phi_p F_{t-p/m} + eta_t
        (AR(p) in high-frequency time)

    Measurement errors (in the state, AR(1) per series):
        u_t = D u_{t-1/m} + eps_t, D = diag(d_u)

    Measurements:
        y*_t = gamma_1' F_t + u_{1,t}
        x_{i,t} = gamma_i' F_t + u_{i,t}, i=2..n
    Mixed-frequency observation pattern:
        for j=1..m-1 : observe x_{2..n} only
        for j=m      : observe y and x_{2..n}   (if include_low_freq=True)
                      otherwise observe x_{2..n} only

    Parameters
    ----------
    m : int
        Number of high-frequency steps per low-frequency period (e.g. 3 for monthly->quarterly).
    Phi_list : list[np.ndarray]
        [Phi_1,...,Phi_p], each (nf x nf).
    Sigma_eta : np.ndarray
        (nf x nf) covariance of factor innovations.
    Gamma : np.ndarray
        (n x nf) factor loadings. Row 0 is for y*, rows 1..n-1 for x2..xn.
    d_u : np.ndarray
        (n,) AR(1) coefficients for measurement errors u_i.
    sigma_eps : np.ndarray
        (n,) std dev for innovations e_i; covariance diag(sigma_eps^2).
    include_low_freq : bool
        Whether y is observed at j=m. If False, never observe y.
    """
    if m < 1:
        raise ValueError("m must be >= 1.")
    p = len(Phi_list)
    if p < 1:
        raise ValueError("Phi_list must have length >= 1.")
    nf = Phi_list[0].shape[0]
    for A in Phi_list:
        if A.shape != (nf, nf):
            raise ValueError("All Phi_l must be (nf x nf).")
    if Sigma_eta.shape != (nf, nf):
        raise ValueError("Sigma_eta must be (nf x nf).")
    n = Gamma.shape[0]
    if Gamma.shape[1] != nf:
        raise ValueError("Gamma must be (n x nf).")
    d_u = np.asarray(d_u, dtype=float).reshape(-1)
    sigma_eps = np.asarray(sigma_eps, dtype=float).reshape(-1)
    if d_u.shape[0] != n or sigma_eps.shape[0] != n:
        raise ValueError("d_u and sigma_eps must be length n (same as Gamma rows).")

    # State dimension: nf*p + n
    # alpha_t = [F_t (nf), F_{t-1/m} (nf), ..., F_{t-(p-1)/m} (nf), u_t (n)]
    dim_x = nf * p + n

    # ---- Build transition G ----
    G = np.zeros((dim_x, dim_x), dtype=float)

    # Top block: F_t depends on factor lags
    # F_t = sum_{l=1..p} Phi_l F_{t-l/m} + eta_t
    # In state, F_{t-l/m} is located at block l (0-indexed):
    # block 0: F_t, block 1: F_{t-1/m}, ..., block (p-1): F_{t-(p-1)/m}
    # So RHS uses blocks 1..p-1 and "block p"? Actually AR(p) uses F_{t-1/m}..F_{t-p/m}.
    # We store only up to F_{t-(p-1)/m}. For AR(p), we need p lags including F_{t-(p)/m}.
    # Common workaround: represent AR(p) as companion with p states.
    # Here we interpret Phi_list as companion mapping from stacked [F_{t-1/m},...,F_{t-p/m}].
    # So we store p lags INCLUDING F_{t-(p-1)/m}, and the p-th lag is the last block.
    # Meaning: block 1 is lag1, ..., block (p-1) is lag(p-1), and lag p is also block (p-1) if p=1.
    # For p>1, we DO have lag p as block (p-1) when indexing shift after update.
    # => We'll build the companion form properly below.

    # Companion: F_t (new) depends on previous stacked factors [F_{t-1/m},...,F_{t-p/m}]
    # In our state at time t-1/m, the factor stack is:
    #   [F_{t-1/m}, F_{t-2/m}, ..., F_{t-p/m}] which correspond to blocks 0..p-1 of alpha_{t-1/m}.
    # But alpha_{t} stores [F_t, F_{t-1/m},...,F_{t-(p-1)/m}].
    # Therefore, when transitioning alpha_{t-1/m} -> alpha_t:
    #   - top block uses Phi_list on blocks 0..p-1 of previous alpha
    #   - shifting: block 1 becomes old block 0, block 2 becomes old block 1, etc.
    #
    # So, in matrix terms:
    # alpha_t[0:nf] = Phi_1 * old_block0 + Phi_2 * old_block1 + ... + Phi_p * old_block(p-1) + eta_t
    for l, Phi_l in enumerate(Phi_list, start=1):
        # old block index = l-1
        c0 = (l - 1) * nf
        G[0:nf, c0:c0+nf] = Phi_l

    # Shift factor lags:
    # new block r (r=1..p-1) = old block (r-1)
    for r in range(1, p):
        rr = r * nf
        cc = (r - 1) * nf
        G[rr:rr+nf, cc:cc+nf] = np.eye(nf)

    # Measurement error dynamics: u_t = D u_{t-1/m} + eps_t
    D = np.diag(d_u)
    u_start = nf * p
    G[u_start:u_start+n, u_start:u_start+n] = D

    # ---- Innovations mapping R and covariance Q ----
    # zeta_t = [eta_t (nf), eps_t (n)]
    dim_eps = nf + n
    R = np.zeros((dim_x, dim_eps), dtype=float)

    # eta_t loads into F_t (top nf)
    R[0:nf, 0:nf] = np.eye(nf)

    # eps_t loads into u_t
    R[u_start:u_start+n, nf:nf+n] = np.eye(n)

    Q = _block_diag(Sigma_eta, np.diag(sigma_eps**2))

    # ---- Build periodic measurement matrices Z_j ----
    if high_freq_names is None:
        high_freq_names = [f"x{i}" for i in range(2, n+1)]  # x2..xn

    if len(high_freq_names) != n - 1:
        raise ValueError("high_freq_names must have length n-1 corresponding to x2..xn.")

    Z_list: List[np.ndarray] = []
    #H_list: List[np.ndarray] = []
    obs_names_list: List[List[str]] = []

    # Helper to build a row for series i (0-based for Gamma/u): series i has loading Gamma[i] and error u_i
    def row_for_series(i: int) -> np.ndarray:
        row = np.zeros((dim_x,), dtype=float)
        # Loading on current factor F_t is in state block 0
        row[0:nf] = Gamma[i, :]
        # Loading on measurement error u_i is in u part
        row[u_start + i] = 1.0
        return row

    for j in range(1, m+1):
        if j < m:
            # observe only x2..xn (i=1..n-1)
            rows = [row_for_series(i) for i in range(1, n)]
            names = high_freq_names
        else:
            if include_low_freq:
                rows = [row_for_series(0)] + [row_for_series(i) for i in range(1, n)]
                names = [low_freq_name] + high_freq_names
            else:
                rows = [row_for_series(i) for i in range(1, n)]
                names = high_freq_names

        Zj = np.vstack(rows) if len(rows) else np.zeros((0, dim_x), dtype=float)
        #Hj = np.zeros((Zj.shape[0], Zj.shape[0]), dtype=float)  # measurement noise handled via u_t

        Z_list.append(Zj)
        #H_list.append(Hj)
        obs_names_list.append(names)

    return PeriodicSSM(
        m=m, nf=nf, n=n, p=p,
        G=G, R=R, Q=Q,
        Z_list=Z_list, #H_list=H_list,
        obs_names_list=obs_names_list
    )
