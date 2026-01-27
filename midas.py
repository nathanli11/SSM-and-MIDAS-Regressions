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
    def __init__(self, m: int, Kx_quarters: int, Ky_quarters: int = 0, include_intercept: bool = False):
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
    def weights(self, theta1: float, theta2: float):
        # Le nombre de poids généré est Kx_quarters*m
        x = np.array([theta1 * j + theta2 * j**2 for j in range(1, (self.Kx_quarters*self.m)+1)])
        w = np.exp(x - x.max())   # -x.max pour la stabilité numérique
        return w / w.sum()
    
    @staticmethod
    def second_month_of_quarter(q_period: pd.Period) -> pd.Timestamp:
        # q_period = Period('1979Q1', freq='Q')
        q_start = q_period.start_time  # Timestamp premier jour du trimestre
        return (q_start + pd.offsets.MonthBegin(1))  # +1 mois => début du 2e mois

    def build_midas_xy(self,
        y: pd.Series,          # index PeriodIndex freq='Q' (ou DatetimeIndex convertible)
        x: pd.Series,          # index DatetimeIndex mensuel (freq MS idéalement)
        Kx_quarters: int,
        m: int = 3,
        h: int = 1,      # h trimestres à prévoir
    ):
        """
        Construit (Y, Xlags, meta_index) pour une régression Regular MIDAS:
        y_{t+h} ~ beta * sum_j w_j x_{t,2 - j}

        - Y : valeurs de y_{t+h} (cible)
        - Xlags : matrice (n_obs, m*Kx) avec les lags mensuels (du plus récent au plus ancien)
        - meta_index : index des trimestres t (origine des régressseurs)
        """
        # Harmonise index de y en PeriodIndex quarterly, et index de x en DateTimeIndex MS
        if not isinstance(y.index, pd.PeriodIndex):
            y = y.copy()
            y.index = pd.PeriodIndex(y.index, freq="Q")
        if isinstance(x.index, pd.PeriodIndex):
            x = x.copy()
            x.index = x.index.to_timestamp(how="start").to_period("M").to_timestamp() 
 
        x = x.copy()
        y = y.copy()

        Kx_months = m * Kx_quarters

        rows_Y = []
        rows_X = []
        rows_t = []

        for t in y.index:
            target = t - h  # y_{t+h}
            if target not in y.index:
                continue

            cutoff = self.second_month_of_quarter(t)  # info date pour le trimestre t

            # On veut les lags mensuels finissant à cutoff (inclus)
            # Exemple: n_month_lags=18 => cutoff, cutoff-1M, ..., cutoff-17M
            lag_months = pd.date_range(end=cutoff, periods=Kx_months, freq="MS")
            # Si cutoff n'est pas un MS exact, on aligne au MS le plus proche (début du mois)
            cutoff_ms = pd.Timestamp(cutoff).to_period("M").to_timestamp(how="start")
            lag_months = pd.date_range(end=cutoff_ms, periods=Kx_months, freq="MS")

            # Vérifie disponibilité
            if (lag_months[0] not in x.index) or (lag_months[-1] not in x.index):
                continue
            if x.loc[lag_months].isna().any():
                continue

            x_vec = x.loc[lag_months].to_numpy(dtype=float)  # du plus ancien au plus récent (date_range)
            x_vec = x_vec[::-1]  # on met "plus récent -> plus ancien" pour matcher weights(j=1..K)

            rows_X.append(x_vec)
            rows_Y.append(float(y.loc[target]))
            rows_t.append(t)

        if len(rows_Y) == 0:
            raise ValueError("Aucune observation utilisable (alignement dates / NaN / lags).")

        Y = np.asarray(rows_Y, dtype=float)
        Xlags = np.asarray(rows_X, dtype=float)
        meta = pd.PeriodIndex(rows_t, freq="Q")

        return Y, Xlags, meta