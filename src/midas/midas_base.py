from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MIDASFit:
    params: Dict[str, Any]
    coef: np.ndarray          # [c, phi, beta]
    theta: np.ndarray         # Paramètres de forme MIDAS (longueur dépend du modèle)
    success: bool
    message: str


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

def ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Coefficients OLS b (solution de min ||y - Xb||^2)"""
    y = np.asarray(y, float).reshape(-1, 1)
    X = np.asarray(X, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b.reshape(-1)

class MixedFreqIndexer:
    """mensuel -> trimestriel (indexing helpers)

    Construit un vecteur de poids de longueur K+1 à partir des paramètres
    (theta1, theta2), normalisé pour sommer à 1
    """

    def __init__(self, m: int = 3):
        if m < 1:
            raise ValueError("m doit être >= 1")
        self.m = int(m)

    @staticmethod
    def ensure_quarterly_period(y: pd.Series) -> pd.Series:
        if not isinstance(y.index, pd.PeriodIndex):
            y = y.copy()
            y.index = pd.PeriodIndex(pd.to_datetime(y.index), freq="Q")
        else:
            y = y.copy().asfreq("Q")
        return y.sort_index()

    @staticmethod
    def ensure_monthly_datetime(x: pd.Series) -> pd.Series:
        x = x.dropna().sort_index()
        if isinstance(x.index, pd.PeriodIndex):
            x = x.copy()
            x.index = x.index.to_timestamp(how="start") 
        if not isinstance(x.index, pd.DatetimeIndex):
            x = x.copy()
            x.index = pd.to_datetime(x.index)
        return x

    def monthly_panel(self, x: pd.Series) -> pd.DataFrame:
        """Renvoie un DataFrame avec colonnes: x, q (quarter Period), j (1..n dans le trimestre)."""
        x = self.ensure_monthly_datetime(x)
        df = pd.DataFrame({"x": x})
        idx = pd.DatetimeIndex(df.index)
        df["q"] = idx.to_period("Q")
        df["j"] = df.groupby("q").cumcount() + 1
        return df

    def stacked_hf_lags(self, df: pd.DataFrame, t: pd.Period, j_obs: int, Kx_LF: int) -> np.ndarray:
        """Regular MIDAS lags empilés: taille Kx_HF = m*Kx_LF, par ordre recent -> vieux"""
        Kx_HF = self.m * int(Kx_LF)

        g = df[df["q"] == t]
        if g.empty:
            raise ValueError(f"Pas de données HF dans le trimestre {t}")
        n_t = int(g["j"].max())
        j_cut = min(int(j_obs), n_t)

        pos = np.flatnonzero(((df["q"] == t) & (df["j"] == j_cut)).to_numpy())
        if pos.size == 0:
            raise ValueError(f"Impossible de localiser (t={t}, j={j_cut}) dans le panel HF")

        cut_pos = int(pos[0])
        start = cut_pos - (Kx_HF - 1)
        if start < 0:
            raise ValueError(f"Pas assez d'observations HF pour {t}: besoin de {Kx_HF} observations HF")

        block = df["x"].iloc[start:cut_pos + 1].to_numpy(dtype=float)  # vieux -> recent
        if np.isnan(block).any():
            raise ValueError("NaNs dans les lags empilés")
        return block[::-1]  # recent -> vieux

    def intra_block(self, df: pd.DataFrame, t: pd.Period, j_obs: int) -> np.ndarray:
        """Multiplicative MIDAS bloc intra-trimestriel (mois 1..j_cut), trié par recent -> vieux"""
        g = df[df["q"] == t]
        if g.empty:
            raise ValueError(f"Pas de données HF dans le trimestre {t}")
        n_t = int(g["j"].max())
        j_cut = min(int(j_obs), n_t)

        block = g[g["j"] <= j_cut]["x"].to_numpy(dtype=float)  # vieux -> recent
        if np.isnan(block).any():
            raise ValueError("NaNs dans le bloc intra-trimestriel")
        return block[::-1]  # recent -> vieux

def hf_lag_at_low_t(x: np.ndarray, t: int, m: int, lag_hf: int) -> float:
    
    if x.ndim == 2:
        x1 = x[:, 0]
    else:
        x1 = x
    idx = t * m - 1 - lag_hf
    if idx < 0 or idx >= len(x1):
        return np.nan
    return float(x1[idx])
