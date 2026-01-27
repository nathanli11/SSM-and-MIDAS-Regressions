import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, Dict, Any
from scipy.optimize import minimize
from midas import Midas, MIDASResult

class DLMidas(Midas):
    def __init__(self, m, Kx_quarters, Ky_quarters, include_intercept = False):
        super().__init__(m, Kx_quarters, Ky_quarters, include_intercept)

    # Estimates intercept + beta
    @staticmethod
    def _ols_beta(y: np.ndarray, x: np.ndarray, include_intercept: bool) -> np.ndarray:
        y = y.reshape(-1, 1)
        x = x.reshape(-1, 1)

        if include_intercept:
            X = np.hstack([np.ones_like(x), x])
        else:
            X = x

        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return b.reshape(-1)

    def objective(self, theta: np.ndarray, Y: np.ndarray, Xlags: np.ndarray) -> float:
        theta1, theta2 = float(theta[0]), float(theta[1])
        w = self.weights(theta1, theta2)
        z = Xlags @ w  # MIDAS aggregated regressor
        b = self._ols_beta(Y, z, self.include_intercept)

        if self.include_intercept:
            c, beta = b[0], b[1]
            resid = Y - (c + beta * z)
        else:
            beta = b[0]
            resid = Y - (beta * z)

        sse = float(np.sum(resid ** 2))
        if not np.isfinite(sse):
            return 1e50
        return sse
    
    def build_midas_xy(self,
        y: pd.Series,          # index PeriodIndex freq='Q' (ou DatetimeIndex convertible)
        x: pd.Series,          # index DatetimeIndex mensuel (freq MS idéalement)
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
        y = y.copy()
        if not isinstance(y.index, pd.PeriodIndex):
            y.index = pd.PeriodIndex(y.index, freq="Q")
        x = x.copy()
        if isinstance(x.index, pd.PeriodIndex):
            x.index = x.index.to_timestamp(how="start").to_period("M").to_timestamp() 

        Kx_months = self.m * self.Kx_quarters

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
    
    def fit(
        self,
        y_est: pd.Series,
        x: pd.Series,
        h: int = 1,
        theta_init: Tuple[float, float] = (0.0, 0.0),
        opt_method: str = "Nelder-Mead",
        opt_options: Optional[Dict[str, Any]] = None,
    ) -> MIDASResult:
        """
        Estimate theta (theta1, theta2) by minimizing SSE, concentrating out beta via OLS.
        Returns MIDASResult and stores fitted parameters in self.theta_, self.beta_.
        """

        # Preparation de Y et X
        Y, Xlags, _ = self.build_midas_xy(y=y_est, x=x, h=h)

        # Vecteur des thetas initiaux
        x0 = np.array(theta_init, dtype=float)

        # Optimize tetha to minimize SSE
        res = minimize(
            self.objective,
            x0=x0,
            args=(Y, Xlags),
            method=opt_method,
            options=(opt_options or {"maxiter": 5000, "disp": False}),
        )

        theta_hat = np.array(res.x, dtype=float)
        w_hat = self.weights(*theta_hat)
        z_hat = Xlags @ w_hat
        beta_hat = self._ols_beta(Y, z_hat, self.include_intercept)

        # Extract regression parameters
        if self.include_intercept:
            c, beta = beta_hat[0], beta_hat[1]
            fitted = c + beta * z_hat
        else:
            beta = beta_hat[0]
            fitted = beta * z_hat

        # Compute SSE using optimal params
        resid = Y - fitted
        sse = float(np.sum(resid ** 2))

        out = MIDASResult(
            params={
                "theta1": float(theta_hat[0]),
                "theta2": float(theta_hat[1]),
                "K": self.Kx_quarters,
                "m": self.m,
                "include_intercept": self.include_intercept,
                "beta": float(beta),
                **({"intercept": float(c)} if self.include_intercept else {}),
            },
            weights=w_hat,
            fitted_values=fitted,
            residuals=resid,
            sse=sse,
            success=bool(res.success),
            message=str(res.message),
        )

        self.theta_ = theta_hat
        self.beta_ = beta_hat
        self.result_ = out
        return out

    def predict(
        self,
        y: pd.Series,
        x: pd.Series,
        h: int = 1,
        t: Optional[pd.Period] = None,
    ) -> float:
        """
        Prévoit y_{t+h}.
        - y: série trimestrielle (PeriodIndex 'Q' recommandé)
        - x: série mensuelle (DatetimeIndex MS recommandé)
        - h: horizon en trimestres
        - t: trimestre origine. Si None => dernier trimestre de y_q (observable)
        """
        if self.theta_ is None or self.beta_ is None:
            raise ValueError("Modèle non estimé: appelle fit() avant predict().")

        if not isinstance(y.index, pd.PeriodIndex):
            y = y.copy()
            y.index = pd.PeriodIndex(y.index, freq="Q")
        y = y.sort_index()
        x = x.sort_index()

        if t is None:
            t = y.index[-1]

        # On construit UNE ligne de Xlags au trimestre t (origine), avec Kx = self.Kx (en QUARTERS)
        # Ici: self.K doit être interprété comme "Kx en QUARTERS" => lags mensuels = m*K
        # (Si chez toi self.K = nb de lags mensuels, dis-le et on ajustera.)
        _, Xlags, _ = self.build_midas_xy(
            y=y.loc[:t],          # y dispo jusqu'à t
            x=x,
            Kx_quarters=self.Kx_quarters,
            m=self.m,
            h=h,
        )

        # La dernière ligne correspond à l'origine t (si elle est utilisable)
        x_row = Xlags[-1, :]  # shape (m*K,)

        theta1, theta2 = float(self.theta_[0]), float(self.theta_[1])
        w = self.weights(theta1, theta2)        # shape (K,) chez toi actuellement

        # IMPORTANT:
        # Ton weights() produit K poids, mais x_row a m*K éléments (mensuels).
        # Pour Regular MIDAS Table 7, le polynôme de poids est sur les *mois* => il faut m*K poids.
        # Donc: soit tu redéfinis self.K = m*Kx (nb de lags mensuels),
        # soit tu modifies weights() pour générer m*Kx poids.
        #
        # Ici je suppose que self.K == (m*Kx_quarters) = nombre de lags mensuels.
        if x_row.shape[0] != w.shape[0]:
            raise ValueError(
                f"Incohérence dimensions: x_row a {x_row.shape[0]} lags mensuels "
                f"mais weights() produit {w.shape[0]} poids. "
                f"Pour Table 7 (regular MIDAS), K doit compter des lags MENSUELS."
            )

        z = float(x_row @ w)

        if self.include_intercept:
            c, beta = float(self.beta_[0]), float(self.beta_[1])
            return c + beta * z
        else:
            beta = float(self.beta_[0])
            return beta * z