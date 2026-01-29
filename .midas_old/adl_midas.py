# adl_midas_paper.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from scipy.optimize import minimize

from midas_old.midas import Midas, MIDASResult


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return OLS coef b solving min ||y - Xb||^2."""
    y = np.asarray(y, float).reshape(-1, 1)
    X = np.asarray(X, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b.reshape(-1)


# ============================================================
# 1) ADL "standard" (single MIDAS polynomial on HF x)
#    - Option A: y-lags unrestricted AR(p) (like your current code)
#    - Option B: y enters via MIDAS weights w_j(theta_y) (paper Eq. 2.25 side y)
# ============================================================

class ADLStandardMIDAS(Midas):
    """
    ADL-MIDAS "standard" (one MIDAS polynomial for x at HF lags).

    Version AR(p) on y:
        y_{t+h} = c + sum_{i=1..p} phi_i y_{t-i} + beta_x * (Xlags_t @ w_x(theta_x)) + e

    Version MIDAS on y (paper-like on y):
        y_{t+h} = c + beta_y * sum_{j=0..Ky} w_j(theta_y) y_{t-j}
                    + beta_x * (Xlags_t @ w_x(theta_x)) + e

    Notes:
    - HF design Xlags_t is built at quarterly origin t using your helper build_midas_xy_generic.
    - If you want the exact multiplicative x-aggregation scheme, use ADLMultiplicativeMIDAS below.
    """

    def __init__(
        self,
        Kx_LF: int,
        m: int,
        p: int = 1,
        include_intercept: bool = True,
        y_mode: str = "ar",          # "ar" or "midas"
        Ky: Optional[int] = None,    # only used if y_mode="midas"
        j_obs: Optional[int] = None,
    ):
        super().__init__(Kx_LF, m, include_intercept=include_intercept)
        if p < 0 or not isinstance(p, int):
            raise ValueError("p must be an integer >= 0")
        if y_mode not in ("ar", "midas"):
            raise ValueError('y_mode must be "ar" or "midas".')
        self.p = p
        self.y_mode = y_mode
        self.Ky = Ky if Ky is not None else (p if y_mode == "midas" else None)
        if self.y_mode == "midas" and (self.Ky is None or self.Ky < 0):
            raise ValueError("Ky must be provided (>=0) when y_mode='midas'.")
        self.j_obs = j_obs

        self.theta_x_: Optional[np.ndarray] = None
        self.theta_y_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.result_: Optional[MIDASResult] = None

    # ---------- design builders ----------
    def _build_y_regressor(self, y_q: pd.Series, meta: pd.PeriodIndex) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          y_target: aligned y_{t+h}
          y_reg   : (n, p) if AR, else (n, Ky+1) raw y-lags for MIDAS weighting
        """
        rows = []
        keep = []
        for i, t in enumerate(meta):
            # For origin t, assume y known up to t-1 (common in nowcasting).
            if self.y_mode == "ar":
                vals = []
                ok = True
                for lag in range(1, self.p + 1):
                    tt = t - lag
                    if tt not in y_q.index or pd.isna(y_q.loc[tt]):
                        ok = False
                        break
                    vals.append(float(y_q.loc[tt]))
                if ok:
                    rows.append(vals)
                    keep.append(i)
            else:
                # MIDAS-on-y uses raw lags y_{t-1}, y_{t-2}, ..., y_{t-(Ky+1)}
                # (exclude y_t for the same "info up to t-1" convention)
                vals = []
                ok = True
                for j in range(0, self.Ky + 1):
                    tt = t - (j + 1)
                    if tt not in y_q.index or pd.isna(y_q.loc[tt]):
                        ok = False
                        break
                    vals.append(float(y_q.loc[tt]))
                if ok:
                    rows.append(vals)
                    keep.append(i)

        if len(keep) == 0:
            raise ValueError("No usable observations after adding y regressors.")

        keep = np.asarray(keep, int)
        return keep, np.asarray(rows, float)

    def _assemble_X(self, n: int, y_part: np.ndarray, z_x: np.ndarray, z_y: Optional[np.ndarray] = None) -> np.ndarray:
        parts = []
        if self.include_intercept:
            parts.append(np.ones((n, 1), float))

        if self.y_mode == "ar":
            if self.p > 0:
                parts.append(y_part)        # (n,p)
        else:
            # y enters as single aggregated scalar z_y (n,)
            parts.append(z_y.reshape(-1, 1))

        parts.append(z_x.reshape(-1, 1))
        return np.hstack(parts)

    # ---------- data build ----------
    def build_xy(
        self,
        y: pd.Series,  # quarterly
        x: pd.Series,  # HF
        h: int = 1,
        j_obs: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.PeriodIndex]:
        """
        Returns aligned (Y, Xlags_HF, y_part, meta).
        - y_part is (n,p) if y_mode="ar", else raw y-lags matrix (n, Ky+1).
        """
        Y, Xlags, meta = self.build_midas_xy_generic(y=y, x=x, h=h, j_obs=j_obs)

        y_q = y.copy()
        if not isinstance(y_q.index, pd.PeriodIndex):
            y_q.index = pd.PeriodIndex(y_q.index, freq="Q")
        y_q = y_q.sort_index().asfreq("Q")

        keep, y_part = self._build_y_regressor(y_q, meta)

        return Y[keep], Xlags[keep, :], y_part, meta[keep]

    # ---------- objective ----------
    def objective(self, theta: np.ndarray, Y: np.ndarray, Xlags: np.ndarray, y_part: np.ndarray) -> float:
        # parameter split
        theta = np.asarray(theta, float)
        theta_x = theta[0:2]
        theta_y = theta[2:4] if self.y_mode == "midas" else None

        # x aggregation (single polynomial on HF stacked lags)
        w_x = self.weights(float(theta_x[0]), float(theta_x[1]))
        if Xlags.shape[1] != w_x.shape[0]:
            return 1e50
        z_x = Xlags @ w_x

        # y aggregation
        z_y = None
        if self.y_mode == "midas":
            # weights over 0..Ky (length Ky+1)
            w_y = self._expalmon_weights(self.Ky + 1, float(theta_y[0]), float(theta_y[1]))
            z_y = y_part @ w_y  # (n,)

        X = self._assemble_X(len(Y), y_part, z_x, z_y=z_y)
        b = _ols(X, Y)
        resid = Y - X @ b
        sse = float(np.sum(resid ** 2))
        return sse if np.isfinite(sse) else 1e50

    @staticmethod
    def _expalmon_weights(K: int, a: float, b: float) -> np.ndarray:
        j = np.arange(1, K + 1, dtype=float)
        x = a * j + b * j ** 2
        w = np.exp(x - x.max())
        return w / w.sum()

    # ---------- fit ----------
    def fit(
        self,
        Y: np.ndarray,
        Xlags: np.ndarray,
        y_part: np.ndarray,
        theta_x_init: Tuple[float, float] = (0.0, 0.0),
        theta_y_init: Tuple[float, float] = (0.0, 0.0),
        opt_method: str = "Nelder-Mead",
        opt_options: Optional[Dict[str, Any]] = None,
    ) -> MIDASResult:
        if self.y_mode == "midas":
            x0 = np.array([theta_x_init[0], theta_x_init[1], theta_y_init[0], theta_y_init[1]], float)
        else:
            x0 = np.array([theta_x_init[0], theta_x_init[1]], float)

        res = minimize(
            self.objective,
            x0=x0,
            args=(Y, Xlags, y_part),
            method=opt_method,
            options=(opt_options or {"maxiter": 5000, "disp": False}),
        )

        theta = np.array(res.x, float)
        self.theta_x_ = theta[0:2].copy()
        if self.y_mode == "midas":
            self.theta_y_ = theta[2:4].copy()

        # rebuild fitted
        w_x = self.weights(float(self.theta_x_[0]), float(self.theta_x_[1]))
        z_x = Xlags @ w_x

        z_y = None
        if self.y_mode == "midas":
            w_y = self._expalmon_weights(self.Ky + 1, float(self.theta_y_[0]), float(self.theta_y_[1]))
            z_y = y_part @ w_y

        X = self._assemble_X(len(Y), y_part, z_x, z_y=z_y)
        b_hat = _ols(X, Y)
        fitted = X @ b_hat
        resid = Y - fitted
        sse = float(np.sum(resid ** 2))
        self.beta_ = b_hat

        params = {
            "Kx_LF": int(self.Kx_LF),
            "Kx_HF": int(self.Kx_LF * self.m),
            "m": int(self.m),
            "y_mode": self.y_mode,
            "p": int(self.p),
            "Ky": (None if self.Ky is None else int(self.Ky)),
            "theta_x1": float(self.theta_x_[0]),
            "theta_x2": float(self.theta_x_[1]),
        }
        if self.y_mode == "midas":
            params.update({"theta_y1": float(self.theta_y_[0]), "theta_y2": float(self.theta_y_[1])})

        out = MIDASResult(
            params=params,
            weights=w_x,
            fitted_values=fitted,
            residuals=resid,
            sse=sse,
            success=bool(res.success),
            message=str(res.message),
        )
        self.result_ = out
        return out


# ============================================================
# 2) ADL-MIDAS "multiplicative" (paper Eq. 2.25–2.26)
#    - within-period aggregator x(theta_x2)_t using m weights
#    - then MIDAS weights across quarterly lags of that aggregator
# ============================================================

class ADLMultiplicativeMIDAS(Midas):
    """
    Multiplicative ADL-MIDAS (paper Eq. 2.25–2.26):
        y_{t+h} = c + beta_y * sum_{j=0..Ky} w_j(theta_y) y_{t-j}
                    + beta_x * sum_{j=0..Kx} w_j(theta_x1) x_agg(t-j; theta_x2) + e
    where:
        x_agg(t; theta_x2) = sum_{k=0..m-1} w_k(theta_x2) x_{t-k/m}

    Convention (nowcast-style):
    - At origin t, y available only up to t-1 -> we use y_{t-1},...,y_{t-(Ky+1)}.
    - x_agg(t) uses the last m HF observations within quarter t up to j_obs.
    """

    def __init__(
        self,
        Kx_LF: int,     # number of quarterly lags in the outer MIDAS (Kx)
        m: int,
        Ky: int = 0,    # number of quarterly lags for y side
        include_intercept: bool = True,
        j_obs: Optional[int] = None,
    ):
        super().__init__(Kx_LF, m, include_intercept=include_intercept)
        if Ky < 0 or not isinstance(Ky, int):
            raise ValueError("Ky must be an integer >= 0")
        self.Ky = Ky
        self.j_obs = j_obs

        self.theta_y_: Optional[np.ndarray] = None
        self.theta_x1_: Optional[np.ndarray] = None
        self.theta_x2_: Optional[np.ndarray] = None
        self.beta_: Optional[np.ndarray] = None
        self.result_: Optional[MIDASResult] = None

    @staticmethod
    def _expalmon_weights(K: int, a: float, b: float) -> np.ndarray:
        j = np.arange(1, K + 1, dtype=float)
        x = a * j + b * j ** 2
        w = np.exp(x - x.max())
        return w / w.sum()

    def _build_y_rawlags(self, y_q: pd.Series, meta: pd.PeriodIndex) -> Tuple[np.ndarray, np.ndarray]:
        rows, keep = [], []
        for i, t in enumerate(meta):
            vals = []
            ok = True
            for j in range(0, self.Ky + 1):
                tt = t - (j + 1)  # exclude y_t
                if tt not in y_q.index or pd.isna(y_q.loc[tt]):
                    ok = False
                    break
                vals.append(float(y_q.loc[tt]))
            if ok:
                rows.append(vals)
                keep.append(i)
        if len(keep) == 0:
            raise ValueError("No usable observations after adding y raw lags.")
        return np.asarray(keep, int), np.asarray(rows, float)  # (n, Ky+1)

    def _build_x_agg_and_outer_lags(
        self,
        x: pd.Series,
        meta: pd.PeriodIndex,
    ) -> np.ndarray:
        """
        Build matrix Xouter of shape (n, Kx_LF+1) where column j is x_agg(t-j).
        Each x_agg(t) uses m HF obs inside quarter t up to j_obs.
        """
        # Ensure we can query x at high-frequency timestamps used by your helper xlags_row_at_origin.
        # We rely on your base method xlags_row_at_origin to get the m HF obs for ONE quarter.
        # Then x_agg(t) is a weighted sum of those m obs.
        n = len(meta)
        Xouter = np.zeros((n, self.Kx_LF + 1), float)

        for i, t in enumerate(meta):
            for j in range(0, self.Kx_LF + 1):
                tj = t - j
                hf_block = self.xlags_row_at_origin(x=x, t=tj, j_obs=self.j_obs)  # expected shape (m,) or (m,1)
                hf_block = np.asarray(hf_block, float).reshape(-1)
                if hf_block.shape[0] != self.m:
                    raise ValueError(f"Expected m={self.m} HF obs for quarter {tj}, got {hf_block.shape[0]}.")
                Xouter[i, j] = np.nan  # placeholder; filled after theta_x2 known
        return Xouter  # placeholder container

    def build_xy(
        self,
        y: pd.Series,
        x: pd.Series,
        h: int = 1,
    ) -> Tuple[np.ndarray, pd.PeriodIndex, np.ndarray, np.ndarray]:
        """
        Returns:
          Y     : (n,)
          meta  : origins t
          Yraw  : (n, Ky+1) raw y-lags
          Xhf   : (n, Kx_LF+1, m) raw HF blocks for each (t-j)
        """
        # Use your generic builder just to get aligned Y and meta (origins).
        Y, _, meta = self.build_midas_xy_generic(y=y, x=x, h=h, j_obs=self.j_obs)

        y_q = y.copy()
        if not isinstance(y_q.index, pd.PeriodIndex):
            y_q.index = pd.PeriodIndex(y_q.index, freq="Q")
        y_q = y_q.sort_index().asfreq("Q")

        keep, Yraw = self._build_y_rawlags(y_q, meta)

        Y2 = Y[keep]
        meta2 = meta[keep]

        # Build raw HF blocks for x_agg(t-j): shape (n, Kx_LF+1, m)
        Xhf = np.zeros((len(meta2), self.Kx_LF + 1, self.m), float)
        for i, t in enumerate(meta2):
            for j in range(0, self.Kx_LF + 1):
                tj = t - j
                hf_block = self.xlags_row_at_origin(x=x, t=tj, j_obs=self.j_obs)
                hf_block = np.asarray(hf_block, float).reshape(-1)
                if hf_block.shape[0] != self.m:
                    raise ValueError(f"Expected m={self.m} HF obs for quarter {tj}, got {hf_block.shape[0]}.")
                Xhf[i, j, :] = hf_block

        return Y2, meta2, Yraw, Xhf

    def objective(self, theta: np.ndarray, Y: np.ndarray, Yraw: np.ndarray, Xhf: np.ndarray) -> float:
        theta = np.asarray(theta, float)
        theta_y = theta[0:2]
        theta_x1 = theta[2:4]
        theta_x2 = theta[4:6]

        # y weighted sum
        w_y = self._expalmon_weights(self.Ky + 1, float(theta_y[0]), float(theta_y[1]))
        z_y = Yraw @ w_y  # (n,)

        # within-quarter weights (length m)
        w_intra = self._expalmon_weights(self.m, float(theta_x2[0]), float(theta_x2[1]))  # (m,)

        # x_agg(t-j) for each j: (n, Kx_LF+1)
        Xagg = np.tensordot(Xhf, w_intra, axes=([2], [0]))  # (n, Kx_LF+1)

        # outer MIDAS weights across quarterly lags of x_agg
        w_outer = self._expalmon_weights(self.Kx_LF + 1, float(theta_x1[0]), float(theta_x1[1]))  # (Kx_LF+1,)
        z_x = Xagg @ w_outer  # (n,)

        # assemble regression
        parts = []
        if self.include_intercept:
            parts.append(np.ones((len(Y), 1), float))
        parts.append(z_y.reshape(-1, 1))
        parts.append(z_x.reshape(-1, 1))
        X = np.hstack(parts)

        b = _ols(X, Y)
        resid = Y - X @ b
        sse = float(np.sum(resid ** 2))
        return sse if np.isfinite(sse) else 1e50

    def fit(
        self,
        Y: np.ndarray,
        Yraw: np.ndarray,
        Xhf: np.ndarray,
        theta_init: Tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0),
        opt_method: str = "Nelder-Mead",
        opt_options: Optional[Dict[str, Any]] = None,
    ) -> MIDASResult:
        x0 = np.array(theta_init, float)
        res = minimize(
            self.objective,
            x0=x0,
            args=(Y, Yraw, Xhf),
            method=opt_method,
            options=(opt_options or {"maxiter": 8000, "disp": False}),
        )

        theta = np.array(res.x, float)
        self.theta_y_ = theta[0:2].copy()
        self.theta_x1_ = theta[2:4].copy()
        self.theta_x2_ = theta[4:6].copy()

        # rebuild fitted
        w_y = self._expalmon_weights(self.Ky + 1, float(self.theta_y_[0]), float(self.theta_y_[1]))
        z_y = Yraw @ w_y

        w_intra = self._expalmon_weights(self.m, float(self.theta_x2_[0]), float(self.theta_x2_[1]))
        Xagg = np.tensordot(Xhf, w_intra, axes=([2], [0]))

        w_outer = self._expalmon_weights(self.Kx_LF + 1, float(self.theta_x1_[0]), float(self.theta_x1_[1]))
        z_x = Xagg @ w_outer

        parts = []
        if self.include_intercept:
            parts.append(np.ones((len(Y), 1), float))
        parts.append(z_y.reshape(-1, 1))
        parts.append(z_x.reshape(-1, 1))
        X = np.hstack(parts)

        b_hat = _ols(X, Y)
        fitted = X @ b_hat
        resid = Y - fitted
        sse = float(np.sum(resid ** 2))

        self.beta_ = b_hat

        out = MIDASResult(
            params={
                "model": "multiplicative_adl_midas",
                "Ky": int(self.Ky),
                "Kx_LF": int(self.Kx_LF),
                "m": int(self.m),
                "theta_y1": float(self.theta_y_[0]),
                "theta_y2": float(self.theta_y_[1]),
                "theta_x1_1": float(self.theta_x1_[0]),
                "theta_x1_2": float(self.theta_x1_[1]),
                "theta_x2_1": float(self.theta_x2_[0]),
                "theta_x2_2": float(self.theta_x2_[1]),
            },
            weights=w_outer,  # outer weights (most interpretable at LF)
            fitted_values=fitted,
            residuals=resid,
            sse=sse,
            success=bool(res.success),
            message=str(res.message),
        )
        self.result_ = out
        return out
