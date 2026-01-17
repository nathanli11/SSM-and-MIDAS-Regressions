import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, Dict, Any
from scipy.optimize import minimize
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
    def __init__(self, K: int):
        if K < 1 or not isinstance(K, int):
            raise ValueError("Lag K must be an integer >= 1")
        self.K = K

    # Defines the weighting scheme based on exponential Almon lag polynomial
    def weights(self, theta1: float, theta2: float):
        x = np.array([theta1 * j + theta2 * j**2 for j in range(1, self.K+1)])
        w = np.exp(x - x.max())   # -x.max pour la stabilité numérique

        return w / w.sum()
    
class DLMidas(Midas):

    def __init__(self, K: int, m: int, include_intercept: bool = True):
        super().__init__(K)
        if m <= 0:
            raise ValueError("m must be >= 1")
        self.m = int(m)
        self.include_intercept = bool(include_intercept)

        self.theta_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None  # [c, beta] if intercept else [beta]
        self.result_: Optional[MIDASResult] = None

    def _build_hf_lag_matrix(self, y_lf: np.ndarray, x_hf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct (Y_target, X_lags) aligned as:
            target row corresponds to y_{t+h}
            regressors use x at time t in LF, i.e. x_hf[t*m] and its HF lags.

        Alignment assumption:
            - y_lf[t] corresponds to low-frequency time t (0..T-1).
            - x_hf is high-frequency with length >= T*m,
              and x_hf[t*m] corresponds to 'time t' (the HF observation aligned with y_t date).
            This is a common, explicit convention; adjust if your timestamps differ.
        """
        y_lf = np.asarray(y_lf, dtype=float).reshape(-1)
        x_hf = np.asarray(x_hf, dtype=float).reshape(-1)

        T = y_lf.size
        needed_x = T * self.m
        if x_hf.size < needed_x:
            raise ValueError(f"x_high must have at least T*m={needed_x} observations (got {x_hf.size}).")

        # We will use t = 0..T-h-1 as "information time", target is y[t+h]
        t_max = T - self.h - 1
        if t_max < 0:
            raise ValueError("Not enough y observations for the chosen horizon h.")

        # For each t, we need x index t*m - j for j=0..K, so require t*m - K >= 0
        valid_t0 = int(np.ceil(self.K / self.m))  # smallest t such that t*m >= K
        if valid_t0 > t_max:
            raise ValueError("Not enough history in x for the chosen K and m (after accounting for horizon h).")

        t_idx = np.arange(valid_t0, t_max + 1)
        Y = y_lf[t_idx + self.h]

        Xlags = np.empty((t_idx.size, self.K + 1), dtype=float)
        for r, t in enumerate(t_idx):
            anchor = t * self.m  # x at LF time t
            # x_{t}, x_{t-1/m}, ..., in HF steps => x_high[anchor - j]
            js = np.arange(self.K + 1, dtype=int)
            Xlags[r, :] = x_hf[anchor - js]

        return Y, Xlags

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
        y_lf: np.ndarray,
        x_hf: np.ndarray,
        theta_init: Tuple[float, float] = (0.0, 0.0),
        opt_method: str = "Nelder-Mead",
        opt_options: Optional[Dict[str, Any]] = None,
    ) -> MIDASResult:
        """
        Estimate theta (theta1, theta2) by minimizing SSE, concentrating out beta via OLS.

        Returns MIDASResult and stores fitted parameters in self.theta_, self.beta_.
        """
        Y, Xlags = self._build_hf_lag_matrix(y_lf, x_hf)

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





model = DLMidas(K=30, m=3, h=1, include_intercept=True)
res = model.fit(y_low, x_high, theta_init=(0.0, 0.0))

print(res.params)
print("Somme des poids:", res.weights.sum())
yhat = model.predict(y_low, x_high)