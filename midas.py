import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class MIDASResult:
    params: Dict[str, Any]
    weights: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    sse: float
    success: bool
    message: str

class Midas:
    def __init__(self, m: int, Kx_quarters: int, Ky_quarters: Optional[int], include_intercept: bool = False):
        if Kx_quarters < 1 or not isinstance(Kx_quarters, int):
            raise ValueError("Lag K_x must be an integer >= 1")
        self.Kx_quarters = Kx_quarters

        if Ky_quarters < 0 or not isinstance(Ky_quarters, int):
            raise ValueError("Lag K_y must be an integer >= 1")
        self.Ky_quarters = Ky_quarters

        if m <= 0:
            raise ValueError("m must be >= 1")
        self.m = int(m)
        self.include_intercept = bool(include_intercept)

        self.theta_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None  # [c, beta] if intercept else [beta]
        self.result_: Optional[MIDASResult] = None

    # Defines the weighting scheme based on exponential Almon lag polynomial
    def weights(self, theta1: float, theta2: float, lags: int):
        # Le nombre de poids généré est Kx_quarters*m
        x = np.array([theta1 * j + theta2 * j**2 for j in range(1, lags+1)])
        w = np.exp(x - x.max())   # -x.max pour la stabilité numérique
        return w / w.sum()
    
    @staticmethod
    def second_month_of_quarter(q_period: pd.Period) -> pd.Timestamp:
        # q_period = Period('1979Q1', freq='Q')
        q_start = q_period.start_time  # Timestamp premier jour du trimestre
        return (q_start + pd.offsets.MonthBegin(1))  # +1 mois => début du 2e mois