import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from scipy.optimize import minimize
from midas_old.midas import Midas, MIDASResult

class DLMidas(Midas):
    def __init__(self, Kx_LF, m, include_intercept = False, j_obs: Optional[int] = None):
        super().__init__(Kx_LF, m, include_intercept)
        self.j_obs = j_obs  # if None -> default in build_midas_xy_generic is m-1

    
    # Estimates intercept + beta
    @staticmethod
    def _ols_beta(y: np.ndarray, x: np.ndarray, include_intercept: bool) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        x = np.asarray(x, dtype=float).reshape(-1, 1)

        if include_intercept:
            X = np.hstack([np.ones_like(x), x])
        else:
            X = x

        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return b.reshape(-1)

    def objective(self, theta: np.ndarray, Y: np.ndarray, Xlags: np.ndarray) -> float:
        theta1, theta2 = float(theta[0]), float(theta[1])
        w = self.weights(theta1, theta2)
        if Xlags.shape[1] != w.shape[0]:
            return 1e50  # incohérence dimensions
        z = Xlags @ w  # MIDAS aggregated regressor
        b = self._ols_beta(Y, z, self.include_intercept)

        if self.include_intercept:
            c, beta = b[0], b[1]
            resid = Y - (c + beta * z)
        else:
            beta = b[0]
            resid = Y - (beta * z)

        sse = float(np.sum(resid ** 2))
        return sse if np.isfinite(sse) else 1e50
        
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

        # basic validation
        Kx_HF = self.m * self.Kx_LF
        if Xlags.ndim != 2:
            raise ValueError("Xlags must be 2D (n_obs, n_lags).")
        if Xlags.shape[1] != Kx_HF:
            raise ValueError(f"Xlags must have {Kx_HF} columns (= m*Kx_LF).")
        if Y.shape[0] != Xlags.shape[0]:
            raise ValueError("Y and Xlags must have same number of rows.")
        
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
        w_hat = self.weights(theta_hat[0], theta_hat[1], self.Kx_quarters*self.m)
        z_hat = Xlags @ w_hat
        beta_hat = self._ols_beta(Y, z_hat, self.include_intercept)

        # Extract regression parameters
        if self.include_intercept:
            c, beta = float(beta_hat[0]), float(beta_hat[1])
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
                "Kx_LF": int(self.Kx_LF),
                "Kx_HF": int(self.Kx_LF * self.m),
                "m": self.m,
                "j_obs": (None if self.j_obs is None else int(self.j_obs)),
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

        if self.theta_ is None or self.beta_ is None:
            raise ValueError("Modèle non estimé: appelle fit() avant predict().")

        if t is None:
            # si t n’est pas fourni, on prend le dernier quarter “possible” côté y
            if not isinstance(y.index, pd.PeriodIndex):
                y = y.copy()
                y.index = pd.PeriodIndex(y.index, freq="Q")
            t = y.index[-1]

        t = pd.Period(t, freq="Q")

        # <-- IMPORTANT: x_row sans dépendre de y contenant t
        x_row = self.xlags_row_at_origin(x=x, t=t, j_obs=self.j_obs)

        theta1, theta2 = float(self.theta_[0]), float(self.theta_[1])
        w = self.weights(theta1, theta2)

        if x_row.shape[0] != w.shape[0]:
            raise ValueError(
                f"Incohérence dimensions: x_row {x_row.shape[0]} vs weights {w.shape[0]}."
            )

        z = float(x_row @ w)

        if self.include_intercept:
            c, beta = float(self.beta_[0]), float(self.beta_[1])
            return c + beta * z
        else:
            beta = float(self.beta_[0])
            return beta * z

    