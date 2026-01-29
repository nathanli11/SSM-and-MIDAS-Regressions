import numpy as np
from midas_base import exp_almon_weights, hf_lag_at_low_t
from numpy.linalg import lstsq
from scipy.optimize import minimize

def regular_midas_forecast(y, x, h=1, m=3, Ky=4, Kx=12):
    """
    Regular MIDAS (Eq. 3.5 in the paper):
      y_{t+h} = b0 + b_y * sum_{j=0..Ky} w_y(j;theta_y) y_{t-j}
                    + b_x * sum_{j=0..Kx} w_x(j;theta_x) x_{t - j/m}
                    + e_{t+h}
    No aggregator scheme for x: we directly use high-frequency lags x_{t - j/m}.
    """
    y = np.asarray(y).astype(float)
    x = np.asarray(x).astype(float)
    T = len(y)
    assert x.shape[0] == T * m

    # Need y lags up to Ky and HF x lags up to Kx at low time t
    t_min = max(Ky, int(np.ceil((Kx + 1) / m)))
    t_idx = np.arange(t_min, T - h)
    if len(t_idx) < 5:
        return np.array([]), np.array([])

    Y = y[t_idx + h]

    def build_terms(theta_y1, theta_y2, theta_x1, theta_x2):
        w_y = exp_almon_weights(Ky, theta_y1, theta_y2)   # Ky+1
        w_x = exp_almon_weights(Kx, theta_x1, theta_x2)   # Kx+1

        Yterm = np.zeros(len(t_idx))
        Xterm = np.zeros(len(t_idx))

        for ii, t in enumerate(t_idx):
            # MIDAS on y (low-frequency lags)
            yt = 0.0
            for j in range(Ky + 1):
                yt += w_y[j] * y[t - j]
            Yterm[ii] = yt

            # MIDAS on x (high-frequency lags) at low-frequency time t
            xt = 0.0
            for j in range(Kx + 1):
                val = hf_lag_at_low_t(x, t, m, j)  # HF lag j from end of period t
                if np.isnan(val):
                    xt = np.nan
                    break
                xt += w_x[j] * val
            Xterm[ii] = xt

        return Yterm, Xterm

    def mse(theta):
        th_y1, th_y2, th_x1, th_x2 = map(float, theta)
        Yterm, Xterm = build_terms(th_y1, th_y2, th_x1, th_x2)
        if np.any(np.isnan(Xterm)) or np.any(np.isnan(Yterm)):
            return 1e18

        # Profile out betas by OLS: Y ≈ b0 + b_y*Yterm + b_x*Xterm
        Xreg = np.column_stack([np.ones(len(t_idx)), Yterm, Xterm])
        beta = lstsq(Xreg, Y, rcond=None)[0]
        resid = Y - Xreg @ beta
        return float(np.mean(resid ** 2))

    theta0 = np.array([-0.1, -0.01,  -0.1, -0.01])
    bnds = [(-10, 10)] * 4
    res = minimize(mse, theta0, method="L-BFGS-B", bounds=bnds)

    th_y1, th_y2, th_x1, th_x2 = map(float, res.x)
    Yterm, Xterm = build_terms(th_y1, th_y2, th_x1, th_x2)
    Xreg = np.column_stack([np.ones(len(t_idx)), Yterm, Xterm])
    beta = lstsq(Xreg, Y, rcond=None)[0]

    forecasts = Xreg @ beta
    actuals = Y
    return np.asarray(forecasts), np.asarray(actuals)

def multiplicative_midas_forecast(y, x, h=1, m=3, Ky=4, Kx=4):
    """
    Paper-style Multiplicative MIDAS (ADL-MIDAS):
      y_{t+h} = b_y * sum_{j=0..Ky} w_y(j;theta_y) y_{t-j}
              + b_x * sum_{j=0..Kx} w_x_inter(j;theta_x_inter) x_agg(t-j;theta_x_intra)
              + e

    where:
      x_agg(t;theta_x_intra) = sum_{k=0..m-1} w_x_intra(k;theta_x_intra) x_{t - k/m}
      and w_* are exponential Almon weights.
    """
    y = np.asarray(y).astype(float)
    x = np.asarray(x).astype(float)
    T = len(y)

    # Need y-lags up to Ky and x inter-lags up to Kx plus intra m-1
    t_min = max(Ky, Kx) + 1
    t_idx = np.arange(t_min, T - h)
    if len(t_idx) < 5:
        return np.array([]), np.array([])

    Y = y[t_idx + h]

    def x_intra_agg(t, th_intra1, th_intra2):
        w_intra = exp_almon_weights(m - 1, th_intra1, th_intra2)  # length m
        s = 0.0
        for k in range(m):
            val = hf_lag_at_low_t(x, t, m, k)  # lag_hf=k within the quarter
            if np.isnan(val):
                return np.nan
            s += w_intra[k] * val
        return s

    def build_terms(theta):
        th_y1, th_y2, th_xi1, th_xi2, th_xa1, th_xa2 = map(float, theta)

        w_y = exp_almon_weights(Ky, th_y1, th_y2)       # Ky+1
        w_xi = exp_almon_weights(Kx, th_xi1, th_xi2)    # inter Kx+1

        Yterm = np.zeros(len(t_idx))
        Xterm = np.zeros(len(t_idx))

        for ii, t in enumerate(t_idx):
            # MIDAS on y (low-frequency lags)
            yt = 0.0
            for j in range(Ky + 1):
                yt += w_y[j] * y[t - j]
            Yterm[ii] = yt

            # double MIDAS on x: intra -> inter
            xt = 0.0
            for j in range(Kx + 1):
                xa = x_intra_agg(t - j, th_xa1, th_xa2)
                if np.isnan(xa):
                    xt = np.nan
                    break
                xt += w_xi[j] * xa
            Xterm[ii] = xt

        return Yterm, Xterm

    def mse(theta):
        Yterm, Xterm = build_terms(theta)
        if np.any(np.isnan(Xterm)) or np.any(np.isnan(Yterm)):
            return 1e18

        # Profile out betas by OLS: Y ≈ b_y*Yterm + b_x*Xterm
        Xreg = np.column_stack([Yterm, Xterm])
        beta = lstsq(Xreg, Y, rcond=None)[0]
        resid = Y - Xreg @ beta
        return float(np.mean(resid ** 2))

    theta0 = np.array([-0.1, -0.01,  -0.1, -0.01,  -0.1, -0.01])
    bnds = [(-10, 10)] * 6
    res = minimize(mse, theta0, method="L-BFGS-B", bounds=bnds)

    Yterm, Xterm = build_terms(res.x)
    Xreg = np.column_stack([Yterm, Xterm])
    beta = lstsq(Xreg, Y, rcond=None)[0]

    forecasts = Xreg @ beta
    actuals = Y
    return np.asarray(forecasts), np.asarray(actuals)
