from midas import Midas, MIDASResult
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, Dict, Any
from scipy.optimize import minimize

class ADLMidas(Midas):
    def __init__(self, K, m, include_intercept = False):
        super().__init__(K, m, include_intercept)

    def _build_regressors(self, y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct Y_target, Y_lags, X_lags
        """

        # Make sure to have 1d arrays
        y = np.asarray(y, dtype=float).reshape(-1)
        x = np.asarray(x, dtype=float).reshape(-1)

        # Check that the regressor x has same period of observation as y
        T = y.size
        needed_x = T * self.m
        if x.size < needed_x:
            raise ValueError(f"Regressor X must have at least T*m={needed_x} observations (got {x.size}).")

        # Build lagged matrixes
        y, y_lagged = self._lagged_matrix(y)
        _, x_lagged = self._lagged_matrix(x)

        return y, y_lagged, x_lagged

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

    def fit(
        self,
        y: np.ndarray,
        x: np.ndarray,
        theta_init: Tuple[float, float] = (0.0, 0.0),
        opt_method: str = "Nelder-Mead",
        opt_options: Optional[Dict[str, Any]] = None,
        is_theta_fitted: bool = False,
    ) -> MIDASResult:
        """
        Estimate theta (theta1, theta2) by minimizing SSE, concentrating out beta via OLS.
        Returns MIDASResult and stores fitted parameters in self.theta_, self.beta_.
        """
        Y, Ylags, Xlags = self._build_regressors(y, x)

        def objective(theta: np.ndarray) -> float:
            theta1, theta2 = float(theta[0]), float(theta[1])
            w = self.weights_expalmon(theta1, theta2)
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

        x0 = np.array(theta_init, dtype=float)

        res = minimize(
            objective,
            x0=x0,
            method=opt_method,
            options=(opt_options or {"maxiter": 5000, "disp": False}),
        )

        theta_hat = np.array(res.x, dtype=float)
        w_hat = self.weights_expalmon(theta_hat[0], theta_hat[1])
        z_hat = Xlags @ w_hat
        beta_hat = self._ols_beta(Y, z_hat, self.include_intercept)

        if self.include_intercept:
            c, beta = beta_hat[0], beta_hat[1]
            fitted = c + beta * z_hat
        else:
            beta = beta_hat[0]
            fitted = beta * z_hat

        resid = Y - fitted
        sse = float(np.sum(resid ** 2))

        out = MIDASResult(
            params={
                "theta1": float(theta_hat[0]),
                "theta2": float(theta_hat[1]),
                "K": self.K,
                "m": self.m,
                "h": self.h,
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
        y_lf: np.ndarray,
        x_hf: np.ndarray,
        theta: Optional[Tuple[float, float]] = None,
        beta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict y_{t+h} for all feasible t given y_lf and x_hf, using stored parameters
        (or user-provided theta/beta).

        Returns predictions aligned with the internal estimation sample (after trimming lags),
        not a full-length vector with NaNs.
        """
        if theta is None:
            if self.theta_ is None:
                raise RuntimeError("Model not fitted and no theta provided.")
            theta = (float(self.theta_[0]), float(self.theta_[1]))
        if beta is None:
            if self.beta_ is None:
                raise RuntimeError("Model not fitted and no beta provided.")
            beta = self.beta_

        Y, Xlags = self._build_hf_lag_matrix(y_lf, x_hf)  # Y just for alignment
        w = self.weights_expalmon(theta[0], theta[1])
        z = Xlags @ w

        if self.include_intercept:
            c, b = float(beta[0]), float(beta[1])
            return c + b * z
        else:
            b = float(beta[0])
            return b * z

    # --- Helpers to connect with eq (2.18) ---
    @staticmethod
    def geometric_weights_from_kalman(rho: float, kappa: float, K: int) -> np.ndarray:
        """
        Eq (2.18) implies weights proportional to (rho - rho*kappa)^j (geometric).
        This returns normalized weights on j=0..K. :
        """
        a = float(rho - rho * kappa)
        j = np.arange(K + 1, dtype=float)
        w = a ** j
        s = np.sum(w)
        return w / s if s != 0 else np.ones(K + 1) / (K + 1)