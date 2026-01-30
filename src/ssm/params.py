import numpy as np
from dataclasses import dataclass
from typing import Optional

# -----------------------------
# State-space model definitions
# -----------------------------
"""
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
"""

"""
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
"""

# Classe model à un facteur avec KF périodique
@dataclass(frozen=True)
class OneFactorParams:
    m: int
    rho: float
    d: float
    lam_y: float
    lam_x: np.ndarray              # shape (n_x,)
    sig2_f: float
    sig2_uy: float
    sig2_ux: np.ndarray            # shape (n_x,)

    @property
    def n_x(self) -> int:
        return int(self.lam_x.shape[0])

    @property
    def dim_state(self) -> int:
        # [f, u_y, u_x1, ..., u_xn]
        return 2 + self.n_x

    @classmethod
    def from_nx(
        cls,
        *,
        m: int,
        n_x: int,
        rho: float,
        d: float,
        lam_y: float = 1.0,
        lam_x: Optional[np.ndarray] = None,
        sig2_f: float = 1.0,
        sig2_uy: float = 1.0,
        sig2_ux: Optional[np.ndarray] = None,
    ) -> "OneFactorParams":
        if lam_x is None:
            lam_x = np.ones(n_x, dtype=float)
        if sig2_ux is None:
            sig2_ux = np.ones(n_x, dtype=float)
        return cls(
            m=m, rho=rho, d=d, lam_y=lam_y,
            lam_x=np.asarray(lam_x, dtype=float),
            sig2_f=float(sig2_f), sig2_uy=float(sig2_uy),
            sig2_ux=np.asarray(sig2_ux, dtype=float),
        )

"""
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
"""

@dataclass(frozen=True)
class TwoFactorParams:
    """
    Paramètres du *SSM 2 facteurs* (pour KF / loglike / estimation).
    """
    m: int = 3
    rho1: float = 0.9
    rho2: float = 0.3
    d: float = 0.0
    lam_y: np.ndarray  = None             # shape (2,) : loadings de y sur (f1,f2)
    lam_x: np.ndarray  = None             # shape (n_x, 2) : loadings de chaque x_i sur (f1,f2)
    sig2_f1: float = 1.0
    sig2_f2: float = 1.0
    sig2_uy: float = 1.0
    sig2_ux: float = 1.0

    @property
    def n_x(self) -> int:
        return int(self.lam_x.shape[0])

    @property
    def dim_state(self) -> int:
        # exemple générique: [f1, f2, u_y, u_x1, ..., u_xn]
        return 3 + self.n_x


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

