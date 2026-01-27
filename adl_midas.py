import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, Dict, Any
from scipy.optimize import minimize
from midas import Midas, MIDASResult

class ADLMidas(Midas):
    def __init__(self, m, Kx_quarters, Ky_quarters, include_intercept = False):
        super().__init__(m, Kx_quarters, Ky_quarters, include_intercept)

    # Estimates intercept + beta
    @staticmethod
    def _ols_beta(y: np.ndarray, x: np.ndarray, include_intercept: bool) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        X = np.asarray(x, dtype=float)

        # Force X en 2D (T,p)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"x doit être 1D ou 2D, reçu ndim={X.ndim}")

        T = y.shape[0]
        if X.shape[0] != T:
            raise ValueError(f"Incohérence dimensions: y a {T} obs, X en a {X.shape[0]}")

        if include_intercept:
            Xreg = np.column_stack([np.ones(T), X])  # (T, p+1)
        else:
            Xreg = X  # (T, p)

        # beta = argmin ||y - Xreg beta||^2
        b, *_ = np.linalg.lstsq(Xreg, y, rcond=None)
        return b.reshape(-1)

    def objective(
        self,
        theta: np.ndarray,
        Y_target: np.ndarray,         # (T,)
        Y_lags_raw: np.ndarray,       # (T, Ky+1)
        X_monthly_raw: np.ndarray,    # (T, Kx+1, m)
    ) -> float:
        """
        ADL-MIDAS multiplicatif (Eq. 2.25-2.26 du papier):
        y_{t+h} = c + beta_y * sum_{j=0..Ky} w_j(theta_y) y_{t-j}
                    + beta_x * sum_{j=0..Kx} w_j(theta_x1) [ sum_{k=0..m-1} w_k(theta_x2) x_{t-j,k} ]
                    + eps

        theta = [theta_y1, theta_y2, theta_x1_1, theta_x1_2, theta_x2_1, theta_x2_2]
        Retourne SSE (float).
        """
        try:
            theta = np.asarray(theta, dtype=float).reshape(-1)
            if theta.size != 6:
                raise ValueError(f"theta doit être de taille 6, reçu {theta.size}")

            # Unpack theta en input
            theta_y1, theta_y2, theta_x1_1, theta_x1_2, theta_x2_1, theta_x2_2 = map(float, theta)

            # Dimensions
            Y_target = np.asarray(Y_target, dtype=float).reshape(-1)
            Y_lags_raw = np.asarray(Y_lags_raw, dtype=float)
            X_monthly_raw = np.asarray(X_monthly_raw, dtype=float)

            T = Y_target.shape[0]
            if Y_lags_raw.shape[0] != T or X_monthly_raw.shape[0] != T:
                raise ValueError("Incohérence de dimensions T entre Y_target, Y_lags_raw, X_monthly_raw")
            
            # Weights
            w_y = self.weights(theta_y1, theta_y2, self.Ky_quarters)                # (Ky,)
            w_x_agg = self.weights(theta_x1_1, theta_x1_2, self.Kx_quarters)        # (Kx,)
            w_x = self.weights(theta_x2_1, theta_x2_2, self.m)                      # (m,)

            # Aggrégat Y_lagged sur Ky
            Y_lagged = Y_lags_raw @ w_y                                             # (T,)

            # Aggrégat intra-trimestre X sur m
            X_lagged = np.tensordot(X_monthly_raw, w_x, axes=([2], [0]))            # (T, Kx)

            # Aggrégat X_lagged sur Kx
            X_agg_lagged = X_lagged @ w_x_agg                                       # (T,)

            # --- OLS concentrée sur [g_y, g_x] (+ intercept si demandé) ---
            if self.include_intercept:
                Xreg = np.column_stack([np.ones(T), Y_lagged, X_agg_lagged])        # (T, 3)
            else:
                Xreg = np.column_stack([Y_lagged, X_agg_lagged])                    # (T, 2)

            # Regression: beta = argmin ||Y - X b||^2
            b, *_ = np.linalg.lstsq(Xreg, Y_target, rcond=None)
            fitted = Xreg @ b
            resid = Y_target - fitted

            sse = float(np.sum(resid ** 2))
            if not np.isfinite(sse):
                return 1e50
            return sse

        except Exception:
            return 1e50
        
    def build_midas_xy(self,
        y: pd.Series,          # quarterly (PeriodIndex 'Q' idéalement)
        x: pd.Series,          # monthly (DatetimeIndex MS idéalement)
        h: int = 1
    ):
        """
        Construit:
        - Y_target : y_{t+h}
        - Y_lags   : [y_t, y_{t-1}, ..., y_{t-Ky}]  (Ky = self.Ky_quarters)
        - X_agg_lags : [ x_agg(t), x_agg(t-1), ..., x_agg(t-Kx) ] (Kx = self.Kx_quarters)
            où x_agg(q) = sum_{k=0}^{m-1} w_k(theta_x2) * x_{month(q,k)}
            (agrégation intra-trimestre pondérée par les poids MIDAS)
        """

        # Harmonise index
        y = y.copy()
        if not isinstance(y.index, pd.PeriodIndex):
            y.index = pd.PeriodIndex(y.index, freq="Q")
        x = x.copy()
        if isinstance(x.index, pd.PeriodIndex):
            x.index = x.index.to_timestamp(how="start").to_period("M").to_timestamp()

        rows_Y = []
        rows_Ylags = []
        rows_Xmonthly = []
        rows_t = []

        # Itère sur les trimestres "origine" t pour lesquels on veut former les régressseurs
        for t in y.index:
            target = t + h
            if target not in y.index:
                continue

            # Lags low frequency de y
            y_lag_periods = [t - j for j in range(1, self.Ky_quarters + 1)]
            if any(p not in y.index for p in y_lag_periods):
                continue
            y_lag_vals = y.loc[y_lag_periods].to_numpy(dtype=float)
            if np.isnan(y_lag_vals).any():
                continue

            # Aggregats quarterly de x
            x_blocks = []
            ok = True

            for j in range(0, self.Kx_quarters + 1):
                q = t - j

                # date d'info intra-trimestre: 2e mois du trimestre q
                cutoff = self.second_month_of_quarter(q)
                cutoff_ms = pd.Timestamp(cutoff).to_period("M").to_timestamp(how="start")

                # mois utilisés pour x_agg(q): cutoff_ms, cutoff_ms-1M, ..., cutoff_ms-(m-1)M
                months = pd.date_range(end=cutoff_ms, periods=self.m, freq="MS")

                # date_range retourne "du plus ancien au plus récent" -> on veut k=0 le plus récent
                months = list(months)[::-1]  # [cutoff, cutoff-1M, ...]

                if (months[-1] not in x.index) or (months[0] not in x.index):
                    ok = False
                    break

                x_vals = x.loc[months].to_numpy(dtype=float)
                if np.isnan(x_vals).any():
                    ok = False
                    break

                x_blocks.append(x_vals)  # shape (m,)

            if not ok:
                continue

            # Stockage des matrices
            rows_Y.append(float(y.loc[target]))
            rows_Ylags.append(y_lag_vals)            # (Ky+1,)
            rows_Xmonthly.append(x_blocks)           # (Kx+1, m)
            rows_t.append(t)

        if len(rows_Y) == 0:
            raise ValueError("Aucune observation utilisable (dates / NaN / lags).")

        Y_target = np.asarray(rows_Y, dtype=float)                     # (T,)
        Y_lags_raw = np.asarray(rows_Ylags, dtype=float)               # (T, Ky+1)
        X_monthly_raw = np.asarray(rows_Xmonthly, dtype=float)         # (T, Kx+1, m)
        meta = pd.PeriodIndex(rows_t, freq="Q")

        return Y_target, Y_lags_raw, X_monthly_raw, meta
    
    def fit(
        self,
        y_est: pd.Series,
        x: pd.Series,
        h: int = 1,
        theta_y_init: Tuple[float, float] = (0.0, 0.0),
        theta_x_agg_init: Tuple[float, float] = (0.0, 0.0),
        theta_x_init: Tuple[float, float] = (0.0, 0.0),
        opt_method: str = "Nelder-Mead",
        opt_options: Optional[Dict[str, Any]] = None,
    ) -> MIDASResult:
        """
        Estimate thetas by minimizing SSE, concentrating out beta via OLS.
        Returns MIDASResult and stores fitted parameters in self.theta_, self.beta_.
        """

        # Preparation de Y et X
        Y, Ylags, Xlags_monthly, _ = self.build_midas_xy(y=y_est, x=x, h=h)

        # Vecteur des thetas initiaux
        x0 = np.array(list(theta_y_init) + list(theta_x_agg_init) + list(theta_x_init), dtype=float)

        # Optimize tetha to minimize SSE
        res = minimize(
            self.objective,
            x0=x0,
            args=(Y, Ylags, Xlags_monthly),
            method=opt_method,
            options=(opt_options or {"maxiter": 5000, "disp": False}),
        )

        # Optimal thetas
        theta_hat = np.array(res.x, dtype=float)

        # Weights
        w_y = self.weights(theta_hat[0], theta_hat[1], self.Ky_quarters)            # (Ky,)
        w_x_agg = self.weights(theta_hat[2], theta_hat[3], self.Kx_quarters)        # (Kx,)
        w_x = self.weights(theta_hat[4], theta_hat[5], self.m)                      # (m,)

        # Aggrégat Y_lagged sur Ky
        Z_Y = Ylags @ w_y                                                           # (T,)

        # Aggrégat intra-trimestre X sur m
        Z_X = np.tensordot(Xlags_monthly, w_x, axes=([2], [0]))                     # (T, Kx)

        # Aggrégat X_lagged sur Kx
        Z_X_agg = Z_X @ w_x_agg 

        # Assemblage des regresseurs
        z_hat = np.column_stack([Z_Y, Z_X_agg])
        beta_hat = self._ols_beta(Y, z_hat, self.include_intercept)

        # Extract regression parameters
        if self.include_intercept:
            c, beta_y, beta_x = beta_hat
            fitted = c + beta_y * Z_Y + beta_x * Z_X_agg
        else:
            beta_y, beta_x = beta_hat
            fitted = beta_y * Z_Y + beta_x * Z_X_agg

        # Compute SSE using optimal params
        resid = Y - fitted
        sse = float(np.sum(resid ** 2))

        out = MIDASResult(
            params={
                "theta_y": (float(theta_hat[0]), float(theta_hat[1])),
                "theta_x1": (float(theta_hat[2]), float(theta_hat[3])),
                "theta_x2": (float(theta_hat[4]), float(theta_hat[5])),
                "K_y": self.Ky_quarters,
                "K_x": self.Kx_quarters,
                "m": self.m,
                "include_intercept": self.include_intercept,
                "beta_y": float(beta_y),
                "beta_x": float(beta_x),
                **({"intercept": float(c)} if self.include_intercept else {}),
            },
            weights=np.hstack([w_y, w_x_agg, w_x]).astype(float, copy=False),
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
        # Assurer que le fit a été fait
        if getattr(self, "theta_", None) is None or getattr(self, "beta_", None) is None:
            raise ValueError("Modèle non estimé: appelle fit() avant predict().")

        # Harmoniser les index
        y = y.copy()
        if not isinstance(y.index, pd.PeriodIndex):
            y.index = pd.PeriodIndex(y.index, freq="Q")
        x = x.copy()
        if isinstance(x.index, pd.PeriodIndex):
            x.index = x.index.to_timestamp(how="start").to_period("M").to_timestamp()

        if t is None:
            t = y.index[-1]

        # Construire les matrices nécessaires
        Y_target, Y_lags, X_monthly_raw, meta = self.build_midas_xy(
            y=y.loc[:t],
            x=x,
            h=h,
        )

        # On veut la ligne correspondant EXACTEMENT à l'origine t
        # (build_midas_xy peut avoir "skip" des trimestres non utilisables)
        if t not in meta:
            raise ValueError(
                f"Impossible de prédire pour t={t}: pas assez de lags/valeurs x disponibles "
                f"(ou NaN) pour construire les régressseurs."
            )
        i = int(np.where(meta == t)[0][-1])  # index de la dernière occurrence de t (normalement unique)

        ylags_row = Y_lags[i, :]          # (Ky+1,)
        xmonthly_row = X_monthly_raw[i, :, :] # (Kx+1, m)

        # Unpack theta (6 paramètres)
        theta_hat = np.asarray(self.theta_, dtype=float).reshape(-1)
        if theta_hat.size != 6:
            raise ValueError(f"theta_ doit être de taille 6 (ADL-MIDAS), reçu {theta_hat.size}")

        theta_y1, theta_y2, theta_x1_1, theta_x1_2, theta_x2_1, theta_x2_2 = map(float, theta_hat)

        # Weights
        w_y = self.weights(theta_hat[0], theta_hat[1], self.Ky_quarters)            # (Ky,)
        w_x_agg = self.weights(theta_hat[2], theta_hat[3], self.Kx_quarters)        # (Kx,)
        w_x = self.weights(theta_hat[4], theta_hat[5], self.m)                      # (m,)

        # Aggrégat Y_lagged sur Ky
        Z_Y = float(ylags_row @ w_y)                                                       # (T,)

        # Aggrégat intra-trimestre X sur m
        Z_X = xmonthly_row @ w_x                                                       # (T, Kx)

        # Aggrégat X_lagged sur Kx
        Z_X_agg = float(Z_X @ w_x_agg) 

        # Application des betas estimés
        b = np.asarray(self.beta_, dtype=float).reshape(-1)

        if self.include_intercept:
            if b.size != 3:
                raise ValueError(f"beta_ attendu taille 3 [c, beta_y, beta_x], reçu {b.size}")
            c, beta_y, beta_x = map(float, b)
            yhat = c + beta_y * Z_Y + beta_x * Z_X_agg
        else:
            if b.size != 2:
                raise ValueError(f"beta_ attendu taille 2 [beta_y, beta_x], reçu {b.size}")
            beta_y, beta_x = map(float, b)
            yhat = beta_y * Z_Y + beta_x * Z_X_agg

        return float(yhat)