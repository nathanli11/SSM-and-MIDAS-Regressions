from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

from adl_midas import ADLRegularMIDAS, ADLMultiplicativeMIDAS

def recursive_rmse(
    y_q: pd.Series,
    x_m: pd.Series,
    h: int,
    model: str,
    Kx_LF: int = 6,
    m: int = 3,
    j_obs: int = 2,
    train_end0: str = "1978Q4",
    eval_start: str = "1979Q1",
    eval_end: str = "2009Q1",
    warm_start: bool = True,
    maxiter: int = 3000,
) -> Tuple[float, int]:
    """Compute expanding-window RMSE for a single regressor and horizon.

    Paper settings (Table 7):
      - initial estimation sample ends 1978Q4
      - evaluation 1979Q1..2009Q1
      - horizons h=1..8
      - info set: monthly data available up to 2nd month of quarter (j_obs=2)
    """
    y_q = y_q.copy()
    if not isinstance(y_q.index, pd.PeriodIndex):
        y_q.index = pd.PeriodIndex(pd.to_datetime(y_q.index), freq="Q")
    y_q = y_q.sort_index().asfreq("Q")

    train_end0 = pd.Period(train_end0, freq="Q") # type: ignore
    eval_start = pd.Period(eval_start, freq="Q") # type: ignore
    eval_end = pd.Period(eval_end, freq="Q") # type: ignore

    if model == "regular":
        M = ADLRegularMIDAS(Kx_LF=Kx_LF, m=m, j_obs=j_obs, include_intercept=True)
        theta_init = (0.0, 0.0)
    elif model == "multiplicative":
        M = ADLMultiplicativeMIDAS(Kx_LF=Kx_LF, m=m, j_obs=j_obs, include_intercept=True)
        theta_init = (0.0, 0.0, 0.0, 0.0)
    else:
        raise ValueError("model must be 'regular' or 'multiplicative'")

    preds, actual = [], []

    for target in pd.period_range(eval_start, eval_end, freq="Q"):
        origin = target - int(h)
        if origin < train_end0: # type: ignore
            continue
        if target not in y_q.index or pd.isna(y_q.loc[target]): # type: ignore
            continue

        # expanding window: fit using data available up to origin (y up to origin; x handled by j_obs inside the model)
        if model == "regular":
            fit = M.fit(y_q=y_q[y_q.index <= origin], x_m=x_m, h=h, origin=origin, theta_init=theta_init, maxiter=maxiter) # type: ignore
            yhat = M.predict_one(y_q=y_q, x_m=x_m, origin=origin)
            if warm_start:
                theta_init = (float(fit.theta[0]), float(fit.theta[1]))
        else:
            fit = M.fit(
                y_q=y_q[y_q.index <= origin],
                x_m=x_m,
                h=h,
                origin=origin,
                theta_out_init=(theta_init[0], theta_init[1]), # type: ignore
                theta_in_init=(theta_init[2], theta_init[3]), # type: ignore
                maxiter=maxiter,
            )
            yhat = M.predict_one(y_q=y_q, x_m=x_m, origin=origin)
            if warm_start:
                theta_init = (float(fit.theta[0]), float(fit.theta[1]), float(fit.theta[2]), float(fit.theta[3]))

        preds.append(float(yhat))
        actual.append(float(y_q.loc[target])) # type: ignore

    preds = np.asarray(preds, float)
    actual = np.asarray(actual, float)
    rmse = float(np.sqrt(np.mean((preds - actual) ** 2)))
    return rmse, int(len(preds))

def table7_rmse_grid(
    y_q: pd.Series,
    X: pd.DataFrame,
    horizons=range(1, 9),
    regressors=None,
    **kwargs,
) -> pd.DataFrame:
    """Return a DataFrame with MultiIndex (regressor, h) and RMSE columns."""
    if regressors is None:
        regressors = list(X.columns)

    rows = []
    for name in regressors:
        x_m = X[name].dropna()
        for h in horizons:
            rm1, n1 = recursive_rmse(y_q, x_m, h=h, model="regular", **kwargs)
            rm2, n2 = recursive_rmse(y_q, x_m, h=h, model="multiplicative", **kwargs)
            rows.append((name, h, rm1, rm2, n1))

    out = pd.DataFrame(rows, columns=["regressor", "h", "rmse_regular", "rmse_multiplicative", "n"])
    return out.set_index(["regressor", "h"]).sort_index()
