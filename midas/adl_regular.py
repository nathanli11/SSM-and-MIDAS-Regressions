from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.optimize import minimize

from midas.midas_base import MixedFreqIndexer, exp_almon_weights, ols, MIDASFit

class ADLRegularMIDAS:
    """Table 7 - Regular ADL-MIDAS (m1).

    Paper-aligned design (monthly→quarterly; info set: j_obs=2 for m=3):
      y_{t+h} = c + phi*y_t + beta * (sum_{k=1..(m*Kx)} w_k(theta_x) x_{t,cut-k+1}) + e_{t+h}

    Defaults: Ky=1 via y_t, Kx_LF up to 6, m=3, j_obs=2.
    """

    def __init__(self, Kx_LF: int = 6, m: int = 3, j_obs: int = 2, include_intercept: bool = True):
        self.Kx_LF = int(Kx_LF)
        self.m = int(m)
        self.j_obs = int(j_obs)
        self.include_intercept = bool(include_intercept)

        self.idx = MixedFreqIndexer(m=self.m)
        self.fit_: Optional[MIDASFit] = None

    def _build_train(self, y_q: pd.Series, x_m: pd.Series, h: int, origin: pd.Period):
        y_q = self.idx.ensure_quarterly_period(y_q)
        df = self.idx.monthly_panel(x_m)

        # Use only (t, t+h) pairs available by origin (paper: expanding window).
        limit = origin - int(h)
        metas = [t for t in y_q.index if (t <= limit) and ((t + h) in y_q.index)]

        Y, ylag, Xlags = [], [], []
        for t in metas:
            if pd.isna(y_q.loc[t]) or pd.isna(y_q.loc[t + h]):
                continue
            try:
                xlags = self.idx.stacked_hf_lags(df, t=t, j_obs=self.j_obs, Kx_LF=self.Kx_LF)
            except Exception:
                continue
            Y.append(float(y_q.loc[t + h]))
            ylag.append(float(y_q.loc[t]))
            Xlags.append(xlags)

        if len(Y) == 0:
            raise ValueError("No usable training observations.")
        return np.asarray(Y, float), np.asarray(ylag, float), np.asarray(Xlags, float), df

    def fit(self, y_q: pd.Series, x_m: pd.Series, h: int, origin: pd.Period,
            theta_init: Tuple[float, float] = (0.0, 0.0),
            maxiter: int = 8000) -> MIDASFit:
        Y, ylag, Xlags, _ = self._build_train(y_q, x_m, h=h, origin=origin)

        def obj(theta):
            w = exp_almon_weights(Xlags.shape[1], float(theta[0]), float(theta[1]))
            z = Xlags @ w
            if self.include_intercept:
                X = np.column_stack([np.ones(len(Y)), ylag, z])
            else:
                X = np.column_stack([ylag, z])
            b = ols(Y, X)
            resid = Y - X @ b
            sse = float(np.sum(resid**2))
            return sse if np.isfinite(sse) else 1e50

        res = minimize(
            obj,
            x0=np.array(theta_init, float),
            method="Nelder-Mead",
            options={"maxiter": int(maxiter), "disp": False},
        )

        theta = np.array(res.x, float)
        w = exp_almon_weights(Xlags.shape[1], theta[0], theta[1])
        z = Xlags @ w

        if self.include_intercept:
            X = np.column_stack([np.ones(len(Y)), ylag, z])
        else:
            X = np.column_stack([ylag, z])
        b = ols(Y, X)

        out = MIDASFit(
            params={"model": "adl_regular", "Ky": 1, "Kx_LF": self.Kx_LF, "m": self.m, "j_obs": self.j_obs, "h": int(h)},
            coef=b,
            theta=theta,
            success=bool(res.success),
            message=str(res.message),
        )
        self.fit_ = out
        return out

    def predict_one(self, y_q: pd.Series, x_m: pd.Series, origin: pd.Period) -> float:
        if self.fit_ is None:
            raise ValueError("Model is not fitted.")
        y_q = self.idx.ensure_quarterly_period(y_q)
        df = self.idx.monthly_panel(x_m)

        ylag = float(y_q.loc[origin]) # type: ignore
        xlags = self.idx.stacked_hf_lags(df, t=origin, j_obs=self.j_obs, Kx_LF=self.Kx_LF)
        w = exp_almon_weights(len(xlags), float(self.fit_.theta[0]), float(self.fit_.theta[1]))
        z = float(xlags @ w)

        if self.include_intercept:
            c, phi, beta = self.fit_.coef[0], self.fit_.coef[1], self.fit_.coef[2]
            return float(c + phi*ylag + beta*z)
        else:
            phi, beta = self.fit_.coef[0], self.fit_.coef[1]
            return float(phi*ylag + beta*z)
