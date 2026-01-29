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
    def __init__(self, Kx_LF: int, m: int, include_intercept: bool = False):
        if Kx_LF < 1 or not isinstance(Kx_LF, int):
            raise ValueError("Lag K must be an integer >= 1")
        self.Kx_LF = Kx_LF

        if m <= 0:
            raise ValueError("m must be >= 1")
        self.m = int(m)
        self.include_intercept = bool(include_intercept)

        self.theta_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None  # [c, beta] if intercept else [beta]
        self.result_: Optional[MIDASResult] = None

    # Defines the weighting scheme based on exponential Almon lag polynomial
    def weights(self, theta1: float, theta2: float):
        """
        Exponential Almon weights on HF lags.

        Convention:
        - x_row[0] is the most recent HF observation available at origin t  (k=0)
        - weights[0] is the weight applied to that most recent HF observation (k=0)

        Length: Kx_HF = m * Kx_LF
        """
        # Le nombre de poids généré est Kx_LF*m
        Kx_HF = self.Kx_LF * self.m
        k = np.arange(Kx_HF, dtype=float)          # 0..Kx_HF-1
        x = theta1 * k + theta2 * k**2
        #x = np.array([theta1 * j + theta2 * j**2 for j in range(1, (self.Kx_LF*self.m)+1)])
        w = np.exp(x - x.max())   # -x.max pour la stabilité numérique
        return w / w.sum()

    def lagged_matrix(self, x: np.ndarray):
        """
        Construit la matrice laggé de x, et retourne x et x_lagged tronqué aux indices de K+1 à T
        """
        x = np.asarray(x)
        T = x.shape[0]
        
        if x.ndim != 1:
            raise ValueError("x doit être un vecteur 1D.")

        if self.Kx_LF < 1:
            raise ValueError("K doit être >= 1.")
        if T <= self.Kx_LF + 1:
            raise ValueError("Il faut n > K+1 pour construire y et X.")
        
        x_lagged = np.column_stack([x[self.Kx_LF+1 - j : T - j].T for j in range(1, self.Kx_LF+1)])
        x = x[self.Kx_LF+1:]

        return x, x_lagged
    
    def xlags_row_at_origin(
        self,
        x: pd.Series,        # DatetimeIndex HF
        t: pd.Period,        # origine LF (quarter)
        j_obs: int | None = None,
    ) -> np.ndarray:
        """
        Construit la ligne Xlags (shape = (Kx_HF,)) pour l'origine trimestrielle t,
        en prenant la sous-période intra-quarter j_cut=min(j_obs, n_t).

        Retourne un vecteur ordonné: plus récent -> plus ancien.
        """
        m = self.m
        Kx_HF = self.Kx_LF * m

        if j_obs is None:
            j_obs = m - 1
        if j_obs < 1:
            raise ValueError("j_obs must be >= 1")

        # x en DatetimeIndex
        x = x.dropna().sort_index()
        if isinstance(x.index, pd.PeriodIndex):
            x = x.copy()
            x.index = x.index.to_timestamp(how="start")
        elif not isinstance(x.index, pd.DatetimeIndex):
            x = x.copy()
            x.index = pd.to_datetime(x.index)

        df = pd.DataFrame({"x": x})
        idx = pd.DatetimeIndex(df.index)
        df["q"] = idx.to_period("Q")
        df["j"] = df.groupby("q").cumcount() + 1

        # sous-échantillon du trimestre t
        g = df[df["q"] == t]
        if g.empty:
            raise ValueError(f"Aucune donnée HF dans le trimestre {t}.")

        n_t = int(g["j"].max())
        j_cut = min(int(j_obs), n_t)

        pos = np.flatnonzero(((df["q"] == t) & (df["j"] == j_cut)).to_numpy())
        if pos.size == 0:
            raise ValueError(f"Impossible de trouver j_cut={j_cut} dans {t}.")
        cut_pos = int(pos[0])

        start = cut_pos - (Kx_HF - 1)
        if start < 0:
            raise ValueError(f"Pas assez d'historique HF pour {t} (besoin {Kx_HF} lags).")

        x_block = df["x"].iloc[start:cut_pos + 1].to_numpy(dtype=float)  # old -> recent
        x_vec = x_block[::-1]  # recent -> old
        if np.isnan(x_vec).any():
            raise ValueError(f"NaN dans les lags HF pour {t}.")

        return x_vec

    
    def build_midas_xy_generic(self,
        y: pd.Series,          # PeriodIndex 'Q'
        x: pd.Series,          # DatetimeIndex (HF)
        h: int = 1,
        j_obs: int | None = None,   # sous-période dispo dans le trimestre t (1..m)
    ) -> Tuple[np.ndarray, np.ndarray, pd.PeriodIndex]:
        """
        Construit (Y, Xlags, meta) pour une régression MIDAS générique :
        y_{t+h} ~ beta * sum_{k=0..m*Kx-1} w_k x_{t, j_obs - k}
       
        - Xlags row for origin t contains Kx_HF HF lags ending at subperiod j_obs of LF period t,
        ordered from most recent to oldest:
            Xlags[row, 0] = most recent available HF obs at origin t (k=0)
            ...
            Xlags[row, Kx_HF-1] = oldest

        - m = nb d'observations HF par trimestre (3 mensuel, 13 hebdo, ...)
        - j_obs = sous-période observée dans le trimestre t (1..m).
        Par défaut: j_obs = m-1 (A VOIR SI ON GARDE EN PARAM OU PAS)
        """

        m = self.m
        Kx_HF = m * self.Kx_LF  # nb de lags HF

        # j_obs = sous-période “cible” (ex: m-1)
        if j_obs is None:
            j_obs = m - 1
        if j_obs < 1:
            raise ValueError("j_obs must be >= 1")

        # y en PeriodIndex trimestriel
        if not isinstance(y.index, pd.PeriodIndex):
            y = y.copy()
            y.index = pd.PeriodIndex(y.index, freq="Q")
        else:
            y = y.copy()
            y.index = y.index.asfreq("Q")
        y = y.sort_index()

        # x en DatetimeIndex trié
        x = x.dropna().sort_index()
        if isinstance(x.index, pd.PeriodIndex):
            # si jamais tu as un PeriodIndex qui traîne, on convertit proprement
            x = x.copy()
            x.index = x.index.to_timestamp(how="start")
        elif not isinstance(x.index, pd.DatetimeIndex):
            x = x.copy()
            x.index = pd.to_datetime(x.index)
        #x = ensure_datetime_index(x.dropna().sort_index())

        # --- Construire un repérage HF : pour chaque obs HF, son trimestre + son rang intra-trimestre (1..m) ---
        df = pd.DataFrame({"x": x})
        idx = pd.DatetimeIndex(df.index)
        df["q"] = idx.to_period("Q")
        # rang 1..n dans le trimestre ; si tu as plus de m points HF dans un trimestre, c'est incohérent
        df["j"] = df.groupby("q").cumcount() + 1

        rows_Y, rows_X, rows_t = [], [], []

        for t in y.index:
            target = t + h
            if target not in y.index:
                continue
            
            g = df[df["q"] == t]
            if g.empty:
                continue
            n_t = int(g["j"].max())
            j_cut = min(int(j_obs), n_t)

            # On veut la date HF correspondant à (trimestre t, sous-période j_obs)
            #mask_cut = (df["q"] == t) & (df["j"] == j_cut)
            #pos = np.flatnonzero(mask_cut.values)
            pos = np.flatnonzero(((df["q"] == t) & (df["j"] == j_cut)).to_numpy())
            if pos.size == 0:
                continue
            cut_pos = int(pos[0])
            #if not mask_cut.any():
                #continue

            # position (index) dans df
            #cut_pos = np.where(mask_cut.values)[0][0]

            # prendre Kx_HF valeurs en remontant depuis cut_pos (incluse)
            start = cut_pos - (Kx_HF - 1)
            if start < 0:
                continue

            x_block = df["x"].iloc[start:cut_pos + 1].to_numpy(dtype=float)  # ancien -> récent
            # mettre récent -> ancien (k=0 = plus récent)
            x_vec = x_block[::-1]

            if np.isnan(x_vec).any():
                continue

            rows_X.append(x_vec)
            rows_Y.append(float(y.loc[target]))
            rows_t.append(t)

        if len(rows_Y) == 0:
            raise ValueError("Aucune observation utilisable (alignement / lags / NaN).")

        Y = np.asarray(rows_Y, dtype=float)
        Xlags = np.asarray(rows_X, dtype=float)   # (n_obs, m*Kx_LF)
        meta = pd.PeriodIndex(rows_t, freq="Q")
        return Y, Xlags, meta

    