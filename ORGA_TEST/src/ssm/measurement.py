import numpy as np
from params import OneFactorParams
from typing import List, Tuple

def build_measurement_mats(p: OneFactorParams) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Measurement equation:
      - for j=1..m-1: observe x's only
      - for j=m: observe [y, x1, x2, ...] (y at low frequency, plus x at end-of-period)
    With state = [f, u_y, u_x1, ..., u_xn]
    and y* = lam_y f + u_y, xi = lam_xi f + u_xi
    """
    n_x = p.n_x
    dim = p.dim_state

    Z_list: List[np.ndarray] = []
    H_list: List[np.ndarray] = []

    # j=1..m-1: x only
    for _ in range(p.m - 1):
        Z = np.zeros((n_x, dim))
        # each row i: xi = lam_x[i]*f + u_xi
        Z[:, 0] = p.lam_x
        for i in range(n_x):
            Z[i, 2 + i] = 1.0
        Z_list.append(Z)
        H_list.append(np.zeros((n_x, n_x)))

    # j=m: y + x's
    Zm = np.zeros((1 + n_x, dim))
    # y row
    Zm[0, 0] = p.lam_y
    Zm[0, 1] = 1.0
    # x rows
    Zm[1:, 0] = p.lam_x
    for i in range(n_x):
        Zm[1 + i, 2 + i] = 1.0
    Z_list.append(Zm)
    H_list.append(np.zeros((1 + n_x, 1 + n_x)))

    return Z_list, H_list

