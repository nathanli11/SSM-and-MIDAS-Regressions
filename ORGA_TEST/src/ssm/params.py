import numpy as np
from dataclasses import dataclass
from typing import Optional

# -----------------------------
# State-space model definitions
# -----------------------------
@dataclass(frozen=True)
class OneFactorParams:
    m: int
    rho: float        # factor AR(1)
    d: float          # measurement error AR(1) (same for all u's here)
    lam_y: float      # loading for y*
    lam_x: np.ndarray # loadings for each x series (shape = (n_x,))
    sig2_f: float     # Var(eps_f)
    sig2_uy: float    # Var(eps_u_y)
    sig2_ux: np.ndarray # Var(eps_u_xi) (shape = (n_x,))

    @property
    def n_x(self) -> int:
        return int(self.lam_x.shape[0])

    @property
    def dim_state(self) -> int:
        # state = [f, u_y, u_x1, ..., u_xn]
        return 2 + self.n_x

# @dataclass
# class OneFactorParams:
#     m: int = 3
#     n_x: int = 1

#     # Loadings (gamma1=gamma2=1)
#     lam_y: float = 1.0
#     lam_x: np.ndarray = None

#     # AR params
#     rho:float = 0.5
#     d: float = 0.0

#     # variances of innovations
#     sig2_f: float = 1.0
#     sig2_uy: float = 1.0
#     sig2_ux: np.ndarray = None  # shape (n_x,)

#     @property
#     def dim_state(self) -> int:
#         # state = [f, u_y, u_x1,...,u_xn]
#         return 2 + self.n_x

@dataclass
class TwoFactorParams:
    m: int = 3
    n_x: int = 1

    rho1: float = 0.9
    rho2: float = 0.3
    d: float = 0.0

    sig2_f1: float = 1.0
    sig2_f2: float = 1.0
    sig2_uy: float = 1.0
    sig2_ux: float = 1.0

    @property
    def dim_state(self):
        # [f1, f2, u_y, u_x]
        return 4


@dataclass(frozen=True)
class TwoFactorDGPParams:
    m: int
    rho: float
    d: float
    # factor loadings: y* = a1 f1 + a2 f2 + u_y ; xi = b1_i f1 + b2_i f2 + u_xi
    a: np.ndarray      # shape (2,)
    b: np.ndarray      # shape (n_x, 2)
    sig2_f1: float = 1.0
    sig2_f2: float = 1.0
    sig2_uy: float = 1.0
    sig2_ux: Optional[np.ndarray] = None  # shape (n_x,)

    def __post_init__(self):
        if self.sig2_ux is None:
            object.__setattr__(self, "sig2_ux", np.ones(self.b.shape[0], dtype=float))

    @property
    def n_x(self) -> int:
        return int(self.b.shape[0])

