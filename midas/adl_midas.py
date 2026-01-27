from midas.midas import Midas, MIDASResult
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, Dict, Any
from scipy.optimize import minimize

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from scipy.optimize import minimize
from midas.midas import Midas, MIDASResult


class ADLMidas(Midas):
    """
    ADL-MIDAS:
        y_{t+h} = c + sum_{i=1..p} phi_i y_{t+1-i} + beta * (Xlags_t @ w(theta)) + e_{t+h}

    Conventions:
    - Xlags_t est construit à l'origine t (quarter) via build_midas_xy_generic(...)
    - w(theta) est de taille Kx_HF = m*Kx_LF
    """

    def __init__(
        self,
        Kx_LF: int,
        m: int,
        p: int = 1,
        include_intercept: bool = True,
        j_obs: Optional[int] = None,
    ):
        super().__init__(Kx_LF, m, include_intercept=include_intercept)
        if p < 0 or not isinstance(p, int):
            raise ValueError("p must be an integer >= 0")
        self.p = p
        self.j_obs = j_obs

    # ---------- OLS helper ----------
    @staticmethod
    def _ols(params_X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        OLS coefficients for y ~ params_X (2D).
        Returns b (k,).
        """
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        X = np.asarray(params_X, dtype=float)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return b.reshape(-1)

    def _build_adl_design(self, Y: np.ndarray, y_lags: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Assemble design matrix [1?, y_lags (p cols), z] according to include_intercept.
        Shapes:
          - Y: (n,)
          - y_lags: (n, p) or (n,0)
          - z: (n,)
        """
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        parts = []
        if self.include_intercept:
            parts.append(np.ones((Y.shape[0], 1), dtype=float))
        if self.p > 0:
            parts.append(np.asarray(y_lags, dtype=float))
        parts.append(z)
        return np.hstack(parts)

    # ---------- Build Y, Xlags, and y-lags aligned ----------
    def build_adl_midas_xy(
        self,
        y: pd.Series,     # quarterly
        x: pd.Series,     # HF
        h: int = 1,
        j_obs: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.PeriodIndex]:
        """
        Returns:
          Y      : (n_obs,)
          Xlags  : (n_obs, Kx_HF)
          Ylags  : (n_obs, p)  where col i is y_{t-i} in "origin time" t
          meta   : PeriodIndex of origins t (quarter)
        """
        # 1) build MIDAS blocks
        Y, Xlags, meta = self.build_midas_xy_generic(y=y, x=x, h=h, j_obs=j_obs)

        # 2) build y-lags at origins meta
        if not isinstance(y.index, pd.PeriodIndex):
            y_q = y.copy()
            y_q.index = pd.PeriodIndex(y_q.index, freq="Q")
        else:
            y_q = y.copy()
            y_q.index = y_q.index.asfreq("Q")
        y_q = y_q.sort_index()

        if self.p == 0:
            Ylags = np.zeros((len(meta), 0), dtype=float)
            return Y, Xlags, Ylags, meta

        # For each origin t, take y_{t}, y_{t-1}, ...? careful:
        # Model uses y_{t}, y_{t-1}, ... as regressors when predicting y_{t+h}.
        # Standard ADL-MIDAS in nowcast: regressors are y_{t}, y_{t-1}, ..., y_{t-p+1}
        # But in your recursive setup, at origin t you only know y up to t-1.
        # So we use y_{t-1}, y_{t-2}, ..., y_{t-p}.
        # (This matches your training sample where y_est ends at est_end=t-1.)
        rows = []
        keep = []
        for idx_t, t in enumerate(meta):
            vals = []
            ok = True
            for i in range(1, self.p + 1):
                tt = t - i
                if tt not in y_q.index or pd.isna(y_q.loc[tt]):
                    ok = False
                    break
                vals.append(float(y_q.loc[tt]))
            if ok:
                rows.append(vals)
                keep.append(idx_t)

        if len(keep) == 0:
            raise ValueError("No usable observations after adding y-lags (p).")

        keep = np.asarray(keep, dtype=int)
        Y2 = Y[keep]
        X2 = Xlags[keep, :]
        Ylags = np.asarray(rows, dtype=float)
        meta2 = meta[keep]
        return Y2, X2, Ylags, meta2

    # ---------- Objective ----------
    def objective(self, theta: np.ndarray, Y: np.ndarray, Xlags: np.ndarray, Ylags: np.ndarray) -> float:
        theta1, theta2 = float(theta[0]), float(theta[1])
        w = self.weights(theta1, theta2)

        if Xlags.shape[1] != w.shape[0]:
            return 1e50

        z = Xlags @ w
        X = self._build_adl_design(Y, Ylags, z)
        b = self._ols(X, Y)

        fitted = X @ b
        resid = Y - fitted
        sse = float(np.sum(resid ** 2))
        return sse if np.isfinite(sse) else 1e50

    # ---------- Fit ----------
    def fit(
        self,
        Y: np.ndarray,
        Xlags: np.ndarray,
        Ylags: np.ndarray,
        theta_init: Tuple[float, float] = (0.0, 0.0),
        opt_method: str = "Nelder-Mead",
        opt_options: Optional[Dict[str, Any]] = None,
    ) -> MIDASResult:

        Kx_HF = self.m * self.Kx_LF
        if Xlags.ndim != 2 or Xlags.shape[1] != Kx_HF:
            raise ValueError(f"Xlags must be (n_obs, {Kx_HF}).")
        if Y.shape[0] != Xlags.shape[0]:
            raise ValueError("Y and Xlags must have same number of rows.")
        if Ylags.shape[0] != Y.shape[0] or Ylags.shape[1] != self.p:
            raise ValueError(f"Ylags must be (n_obs, p={self.p}).")

        x0 = np.array(theta_init, dtype=float)

        res = minimize(
            self.objective,
            x0=x0,
            args=(Y, Xlags, Ylags),
            method=opt_method,
            options=(opt_options or {"maxiter": 5000, "disp": False}),
        )

        theta_hat = np.array(res.x, dtype=float)
        w_hat = self.weights(theta_hat[0], theta_hat[1])
        z_hat = Xlags @ w_hat

        X = self._build_adl_design(Y, Ylags, z_hat)
        b_hat = self._ols(X, Y)

        fitted = X @ b_hat
        resid = Y - fitted
        sse = float(np.sum(resid ** 2))

        # unpack coefficients for params
        idx = 0
        c = None
        if self.include_intercept:
            c = float(b_hat[idx]); idx += 1
        phis = []
        if self.p > 0:
            phis = [float(v) for v in b_hat[idx:idx + self.p]]
            idx += self.p
        beta = float(b_hat[idx])

        out = MIDASResult(
            params={
                "theta1": float(theta_hat[0]),
                "theta2": float(theta_hat[1]),
                "Kx_LF": int(self.Kx_LF),
                "Kx_HF": int(self.Kx_LF * self.m),
                "m": int(self.m),
                "p": int(self.p),
                "j_obs": (None if self.j_obs is None else int(self.j_obs)),
                "include_intercept": bool(self.include_intercept),
                "beta": beta,
                **({"intercept": c} if self.include_intercept else {}),
                **({f"phi{i+1}": phis[i] for i in range(len(phis))} if self.p > 0 else {}),
            },
            weights=w_hat,
            fitted_values=fitted,
            residuals=resid,
            sse=sse,
            success=bool(res.success),
            message=str(res.message),
        )

        self.theta_ = theta_hat
        self.beta_ = b_hat
        self.result_ = out
        return out

    # ---------- Predict ----------
    def predict(
        self,
        y: pd.Series,
        x: pd.Series,
        h: int = 1,
        t: Optional[pd.Period] = None,
    ) -> float:
        """
        Forecast y_{t+h} from origin t using:
          - HF block ending at t (via xlags_row_at_origin)
          - y-lags available at origin t: y_{t-1},...,y_{t-p}
        """
        if self.theta_ is None or self.beta_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # normalize t
        if t is None:
            if not isinstance(y.index, pd.PeriodIndex):
                y = y.copy()
                y.index = pd.PeriodIndex(y.index, freq="Q")
            t = y.index[-1]
        t = pd.Period(t, freq="Q")

        # ensure y quarterly index
        if not isinstance(y.index, pd.PeriodIndex):
            yq = y.copy()
            yq.index = pd.PeriodIndex(yq.index, freq="Q")
        else:
            yq = y.copy()
            yq.index = yq.index.asfreq("Q")
        yq = yq.sort_index()

        # y-lags at origin t: y_{t-1}..y_{t-p}
        ylags = []
        for i in range(1, self.p + 1):
            tt = t - i
            if tt not in yq.index or pd.isna(yq.loc[tt]):
                raise ValueError(f"Missing y lag {tt} for prediction at origin {t}.")
            ylags.append(float(yq.loc[tt]))
        ylags = np.asarray(ylags, dtype=float).reshape(1, -1)  # (1,p) or (1,0)

        # HF block at origin t
        x_row = self.xlags_row_at_origin(x=x, t=t, j_obs=self.j_obs)

        theta1, theta2 = float(self.theta_[0]), float(self.theta_[1])
        w = self.weights(theta1, theta2)
        if x_row.shape[0] != w.shape[0]:
            raise ValueError("x_row and weights dimension mismatch.")
        z = float(x_row @ w)

        # build design row
        Y_dummy = np.zeros(1, dtype=float)
        Xrow = self._build_adl_design(Y_dummy, ylags, np.array([z], dtype=float))
        yhat = float(Xrow @ self.beta_)
        return yhat


# class ADLMidas(Midas):
#     def __init__(self, K, m, include_intercept = False):
#         super().__init__(K, m, include_intercept)

#     def _build_regressors(self, y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Construct Y_target, Y_lags, X_lags
#         """

#         # Make sure to have 1d arrays
#         y = np.asarray(y, dtype=float).reshape(-1)
#         x = np.asarray(x, dtype=float).reshape(-1)

#         # Check that the regressor x has same period of observation as y
#         T = y.size
#         needed_x = T * self.m
#         if x.size < needed_x:
#             raise ValueError(f"Regressor X must have at least T*m={needed_x} observations (got {x.size}).")

#         # Build lagged matrixes
#         y, y_lagged = self.lagged_matrix(y)
#         _, x_lagged = self.lagged_matrix(x)

#         return y, y_lagged, x_lagged

#     # Estimates intercept + beta
#     @staticmethod
#     def _ols_beta(y: np.ndarray, x: np.ndarray, include_intercept: bool) -> np.ndarray:
#         y = y.reshape(-1, 1)
#         x = x.reshape(-1, 1)

#         if include_intercept:
#             X = np.hstack([np.ones_like(x), x])
#         else:
#             X = x

#         b, *_ = np.linalg.lstsq(X, y, rcond=None)
#         return b.reshape(-1)

#     def fit(
#         self,
#         y: np.ndarray,
#         x: np.ndarray,
#         theta_init: Tuple[float, float] = (0.0, 0.0),
#         opt_method: str = "Nelder-Mead",
#         opt_options: Optional[Dict[str, Any]] = None,
#         is_theta_fitted: bool = False,
#     ) -> MIDASResult:
#         """
#         Estimate theta (theta1, theta2) by minimizing SSE, concentrating out beta via OLS.
#         Returns MIDASResult and stores fitted parameters in self.theta_, self.beta_.
#         """
#         Y, Ylags, Xlags = self._build_regressors(y, x)

#         def objective(theta: np.ndarray) -> float:
#             theta1, theta2 = float(theta[0]), float(theta[1])
#             w = self.weights_expalmon(theta1, theta2)
#             z = Xlags @ w  # MIDAS aggregated regressor
#             b = self._ols_beta(Y, z, self.include_intercept)

#             if self.include_intercept:
#                 c, beta = b[0], b[1]
#                 resid = Y - (c + beta * z)
#             else:
#                 beta = b[0]
#                 resid = Y - (beta * z)

#             sse = float(np.sum(resid ** 2))
#             if not np.isfinite(sse):
#                 return 1e50
#             return sse

#         x0 = np.array(theta_init, dtype=float)

#         res = minimize(
#             objective,
#             x0=x0,
#             method=opt_method,
#             options=(opt_options or {"maxiter": 5000, "disp": False}),
#         )

#         theta_hat = np.array(res.x, dtype=float)
#         w_hat = self.weights_expalmon(theta_hat[0], theta_hat[1])
#         z_hat = Xlags @ w_hat
#         beta_hat = self._ols_beta(Y, z_hat, self.include_intercept)

#         if self.include_intercept:
#             c, beta = beta_hat[0], beta_hat[1]
#             fitted = c + beta * z_hat
#         else:
#             beta = beta_hat[0]
#             fitted = beta * z_hat

#         resid = Y - fitted
#         sse = float(np.sum(resid ** 2))

#         out = MIDASResult(
#             params={
#                 "theta1": float(theta_hat[0]),
#                 "theta2": float(theta_hat[1]),
#                 "K": self.K,
#                 "m": self.m,
#                 "h": self.h,
#                 "include_intercept": self.include_intercept,
#                 "beta": float(beta),
#                 **({"intercept": float(c)} if self.include_intercept else {}),
#             },
#             weights=w_hat,
#             fitted_values=fitted,
#             residuals=resid,
#             sse=sse,
#             success=bool(res.success),
#             message=str(res.message),
#         )

#         self.theta_ = theta_hat
#         self.beta_ = beta_hat
#         self.result_ = out
#         return out

#     def predict(
#         self,
#         y_lf: np.ndarray,
#         x_hf: np.ndarray,
#         theta: Optional[Tuple[float, float]] = None,
#         beta: Optional[np.ndarray] = None,
#     ) -> np.ndarray:
#         """
#         Predict y_{t+h} for all feasible t given y_lf and x_hf, using stored parameters
#         (or user-provided theta/beta).

#         Returns predictions aligned with the internal estimation sample (after trimming lags),
#         not a full-length vector with NaNs.
#         """
#         if theta is None:
#             if self.theta_ is None:
#                 raise RuntimeError("Model not fitted and no theta provided.")
#             theta = (float(self.theta_[0]), float(self.theta_[1]))
#         if beta is None:
#             if self.beta_ is None:
#                 raise RuntimeError("Model not fitted and no beta provided.")
#             beta = self.beta_

#         Y, Xlags = self._build_hf_lag_matrix(y_lf, x_hf)  # Y just for alignment
#         w = self.weights_expalmon(theta[0], theta[1])
#         z = Xlags @ w

#         if self.include_intercept:
#             c, b = float(beta[0]), float(beta[1])
#             return c + b * z
#         else:
#             b = float(beta[0])
#             return b * z

#     # --- Helpers to connect with eq (2.18) ---
#     @staticmethod
#     def geometric_weights_from_kalman(rho: float, kappa: float, K: int) -> np.ndarray:
#         """
#         Eq (2.18) implies weights proportional to (rho - rho*kappa)^j (geometric).
#         This returns normalized weights on j=0..K. :
#         """
#         a = float(rho - rho * kappa)
#         j = np.arange(K + 1, dtype=float)
#         w = a ** j
#         s = np.sum(w)
#         return w / s if s != 0 else np.ones(K + 1) / (K + 1)