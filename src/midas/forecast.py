import numpy as np
from src.midas.midas_base import exp_almon_weights, hf_lag_at_low_t
from numpy.linalg import lstsq
from scipy.optimize import minimize

def regular_midas_forecast(y, x, h=1, m=3, Ky=4, Kx=12):
    
    y = np.asarray(y).astype(float)
    x = np.asarray(x).astype(float)
    T = len(y)
    assert x.shape[0] == T * m

    
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
            # MIDAS sur y (LF lags)
            yt = 0.0
            for j in range(Ky + 1):
                yt += w_y[j] * y[t - j]
            Yterm[ii] = yt

            # MIDAS sur x (HF lags) sur LF t
            xt = 0.0
            for j in range(Kx + 1):
                val = hf_lag_at_low_t(x, t, m, j)  
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

        # OLS: Y ≈ b0 + b_y*Yterm + b_x*Xterm
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
    """ Prev MIDAS mutliplicative avec ponderations exp d'Almon"""

    y = np.asarray(y).astype(float)
    x = np.asarray(x).astype(float)
    T = len(y)

    #Indice de depart
    t_min = max(Ky, Kx) + 1
    t_idx = np.arange(t_min, T - h)
    if len(t_idx) < 5:
        return np.array([]), np.array([])

    # Variable dependante à horizon h
    Y = y[t_idx + h]

    # Aggregation intra periode
    def x_intra_agg(t, th_intra1, th_intra2):
        w_intra = exp_almon_weights(m - 1, th_intra1, th_intra2)  # length m
        s = 0.0
        for k in range(m):
            val = hf_lag_at_low_t(x, t, m, k)  
            if np.isnan(val):
                return np.nan
            s += w_intra[k] * val
        return s

    def build_terms(theta):
        """Construction des composantes MIDAS de y et x"""
        th_y1, th_y2, th_xi1, th_xi2, th_xa1, th_xa2 = map(float, theta)
        # poids retard bf de y et x
        w_y = exp_almon_weights(Ky, th_y1, th_y2)       
        w_xi = exp_almon_weights(Kx, th_xi1, th_xi2)    

        Yterm = np.zeros(len(t_idx))
        Xterm = np.zeros(len(t_idx))

        for ii, t in enumerate(t_idx):

            # composante autoregressive sur y
            yt = 0.0
            for j in range(Ky + 1):
                yt += w_y[j] * y[t - j]
            Yterm[ii] = yt
            # composante autoregressive sur x
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
        # fonction objectif
        Yterm, Xterm = build_terms(theta)
        if np.any(np.isnan(Xterm)) or np.any(np.isnan(Yterm)):
            return 1e18

        # OLS: Y ≈ b_y*Yterm + b_x*Xterm
        Xreg = np.column_stack([Yterm, Xterm])
        beta = lstsq(Xreg, Y, rcond=None)[0]
        resid = Y - Xreg @ beta
        return float(np.mean(resid ** 2))

    # valeurs initiales
    theta0 = np.array([-0.1, -0.01,  -0.1, -0.01,  -0.1, -0.01])
    bnds = [(-10, 10)] * 6
    res = minimize(mse, theta0, method="L-BFGS-B", bounds=bnds)

    Yterm, Xterm = build_terms(res.x)
    Xreg = np.column_stack([Yterm, Xterm])
    beta = lstsq(Xreg, Y, rcond=None)[0]

    forecasts = Xreg @ beta
    actuals = Y
    return np.asarray(forecasts), np.asarray(actuals)
