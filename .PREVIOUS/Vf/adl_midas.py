from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.optimize import minimize

from midas_base import MixedFreqIndexer, exp_almon_weights, ols, MIDASFit

class ADLRegularMIDAS:
    """Regular ADL-MIDAS (m1).

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


class ADLMultiplicativeMIDAS:
    """Multiplicative ADL-MIDAS (m2), paper Eq. (2.25)-(2.26).

    Design:
      x_agg(t; theta_in)  = sum_{k=1..K_intra} w_k(theta_in) x_{t,k}   (intra-quarter)
      z_x(t; theta_out)   = sum_{j=0..Kx} w_j(theta_out) x_agg(t-j)
      y_{t+h}             = c + phi*y_t + beta*z_x(t) + e_{t+h}

    For monthly data:
      m=3, j_obs=2 => K_intra=2 (months available inside quarter at the forecast origin).
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

        limit = origin - int(h)
        metas = [t for t in y_q.index if (t <= limit) and ((t + h) in y_q.index)]

        Y, ylag, X_intra = [], [], []
        Kintra = None

        for t in metas:
            if pd.isna(y_q.loc[t]) or pd.isna(y_q.loc[t + h]):
                continue

            try:
                blocks = []
                for j in range(0, self.Kx_LF + 1):
                    block = self.idx.intra_block(df, t=t - j, j_obs=self.j_obs)  # recent→old
                    blocks.append(block)
            except Exception:
                continue

            lens = [len(b) for b in blocks]
            if Kintra is None:
                Kintra = min(lens)
            if any(l != Kintra for l in lens):
                # missing monthly obs => skip (paper datasets are usually complete after 1959)
                continue

            Y.append(float(y_q.loc[t + h]))
            ylag.append(float(y_q.loc[t]))
            X_intra.append(np.stack(blocks, axis=0))  # (Kx+1, Kintra)

        if len(Y) == 0:
            raise ValueError("No usable training observations.")
        return np.asarray(Y, float), np.asarray(ylag, float), np.asarray(X_intra, float), df

    def fit(self, y_q: pd.Series, x_m: pd.Series, h: int, origin: pd.Period,
            theta_out_init: Tuple[float, float] = (0.0, 0.0),
            theta_in_init: Tuple[float, float] = (0.0, 0.0),
            maxiter: int = 10000) -> MIDASFit:
        Y, ylag, X_intra, _ = self._build_train(y_q, x_m, h=h, origin=origin)

        # theta = [out1,out2,in1,in2]
        x0 = np.array([theta_out_init[0], theta_out_init[1], theta_in_init[0], theta_in_init[1]], float)

        def obj(theta):
            w_in = exp_almon_weights(X_intra.shape[2], float(theta[2]), float(theta[3]))
            Xagg = np.tensordot(X_intra, w_in, axes=([2], [0]))      # (n, Kx+1)
            w_out = exp_almon_weights(X_intra.shape[1], float(theta[0]), float(theta[1]))
            z = Xagg @ w_out

            if self.include_intercept:
                X = np.column_stack([np.ones(len(Y)), ylag, z])
            else:
                X = np.column_stack([ylag, z])

            b = ols(Y, X)
            resid = Y - X @ b
            sse = float(np.sum(resid**2))
            return sse if np.isfinite(sse) else 1e50

        res = minimize(obj, x0=x0, method="Nelder-Mead", options={"maxiter": int(maxiter), "disp": False})
        theta = np.array(res.x, float)

        # store only what you need for prediction
        out = MIDASFit(
            params={"model": "adl_multiplicative", "Ky": 1, "Kx_LF": self.Kx_LF, "m": self.m, "j_obs": self.j_obs, "h": int(h)},
            coef=np.nan*np.ones(3),  # filled below
            theta=theta,
            success=bool(res.success),
            message=str(res.message),
        )

        # rebuild coefs
        w_in = exp_almon_weights(X_intra.shape[2], float(theta[2]), float(theta[3]))
        Xagg = np.tensordot(X_intra, w_in, axes=([2], [0]))
        w_out = exp_almon_weights(X_intra.shape[1], float(theta[0]), float(theta[1]))
        z = Xagg @ w_out

        if self.include_intercept:
            X = np.column_stack([np.ones(len(Y)), ylag, z])
        else:
            X = np.column_stack([ylag, z])

        b = ols(Y, X)
        out.coef = b

        self.fit_ = out
        return out

    def predict_one(self, y_q: pd.Series, x_m: pd.Series, origin: pd.Period) -> float:
        if self.fit_ is None:
            raise ValueError("Model is not fitted.")
        y_q = self.idx.ensure_quarterly_period(y_q)
        df = self.idx.monthly_panel(x_m)

        ylag = float(y_q.loc[origin]) # type: ignore

        # Build blocks for j=0..Kx
        blocks = []
        for j in range(0, self.Kx_LF + 1):
            block = self.idx.intra_block(df, t=origin - j, j_obs=self.j_obs)
            blocks.append(block)

        Kintra = len(blocks[0])
        if any(len(b) != Kintra for b in blocks):
            raise ValueError("Inconsistent intra-quarter blocks (missing HF months?).")

        X_intra = np.stack(blocks, axis=0)  # (Kx+1, Kintra)

        # theta = [out1,out2,in1,in2]
        theta = self.fit_.theta
        w_in = exp_almon_weights(Kintra, float(theta[2]), float(theta[3]))
        xagg = X_intra @ w_in  # (Kx+1,)

        w_out = exp_almon_weights(self.Kx_LF + 1, float(theta[0]), float(theta[1]))
        z = float(xagg @ w_out)

        if self.include_intercept:
            c, phi, beta = self.fit_.coef[0], self.fit_.coef[1], self.fit_.coef[2]
            return float(c + phi*ylag + beta*z)
        else:
            phi, beta = self.fit_.coef[0], self.fit_.coef[1]
            return float(phi*ylag + beta*z)
