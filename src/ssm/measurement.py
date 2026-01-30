import numpy as np
from src.ssm.params import OneFactorParams
from typing import List, Tuple

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
