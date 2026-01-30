#============================
# Imports
#============================
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.optimize import minimize
from dataclasses import dataclass
from utils import *

# Constantes
RICCATI_MAX_ITERS = 50_000
RICCATI_TOL = 1e-12
RHO_GRID = [-0.9, -0.5, 0.5, 0.95]   
D_GRID    = [-0.9, -0.5, 0.0, 0.5, 0.95]

#============================
# Kalman filter
#============================
@dataclass
class OneFactorParams:
    m: int = 3
    n_x: int = 1

    # Coef (gamma1=gamma2=1)
    lam_y: float = 1.0
    lam_x: np.ndarray = None

    # AR parametres
    rho:float = 0.5
    d: float = 0.0

    # variances des innov
    sig2_f: float = 1.0
    sig2_uy: float = 1.0
    sig2_ux: np.ndarray = None 

    @property
    def dim_state(self) -> int:
        # Dimension state = [f, u_y, u_x1,...,u_xn]
        return 2 + self.n_x

@dataclass
class TwoFactorParams:
    m: int = 3
    n_x: int = 1

    rho1: float = 0.9
    rho2: float = 0.3
    d: float = 0.0

    sig2_f1: float = 1.0
    sig2_f2: float = 1.0
    sig2_uy: float = 1.0
    sig2_ux: float = 1.0

    @property
    def dim_state(self):
        # [f1, f2, u_y, u_x]
        return 4

def kalman_loglike_full(p: OneFactorParams, y: np.ndarray, x: np.ndarray) -> float:
    """
    Log vraisemblance de Kalman format matriciel
    y: (T_BF,) variable basse frequence
    x: (T_HF, n_x) avec T_HF = T_BF*m variables haute frequence
    """

    # Verification dimension
    assert x.shape[0] == y.shape[0] * p.m
    assert x.shape[1] == p.n_x

    # Construction des matrices (mesure et variance d'erreur de mesure)
    Z_list, H_list = build_measurement_mats(p)
    # Matrice de transition de l etat
    G = np.diag([p.rho] + [p.d] * (1 + p.n_x))
    # Matrice de variance des innovations
    Q = np.diag([p.sig2_f, p.sig2_uy] + list(p.sig2_ux))

    # Initialisation
    dim = p.dim_state
    a = np.zeros(dim)
    P = np.eye(dim) * 10.0  

    ll = 0.0
    low_idx = 0
    two_pi = np.log(2.0 * np.pi)

    # Boucle sur les observations
    for t_high in range(x.shape[0]):
        j = (t_high % p.m) + 1      # 1..m
        jj = j - 1                  # 0..m-1

        # Prediction
        a = G @ a
        P = G @ P @ G.T + Q

        # Mesure
        if j < p.m:
            # (n_x,)
            y_obs = x[t_high, :]  
            # (n_x, dim)                 
            Z = Z_list[jj]     
            # (n_x, n_x)                    
            H = H_list[jj]                         
        else:
            # (1+n_x,)
            y_obs = np.concatenate([[y[low_idx]], x[t_high, :]])  
            # (1+n_x, dim)
            Z = Z_list[jj]           
            # (1+n_x, 1+n_x)              
            H = H_list[jj]                         
            low_idx += 1

        # residu
        v = y_obs - (Z @ a)
        # variance de l innovation
        S = Z @ P @ Z.T + H

        # Stabilite numerique pour que ca run correctement matrice inverse
        try:
            # Cholesky
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return -np.inf

        # inversion de matrice
        tmp = np.linalg.solve(L, v)
        Sinv_v = np.linalg.solve(L.T, tmp)
        quad = float(v.T @ Sinv_v)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        # Dim de l observation
        k = len(y_obs)

        ll += -0.5 * (logdet + quad + k * two_pi)

        # MAJ
        # Gain de Kalman : K = P Z' S^{-1} 
        
        PZt = P @ Z.T
        #  S^{-1} * (Z P)' = S^{-1} * (PZt)'
        W = np.linalg.solve(L, PZt.T)
        U = np.linalg.solve(L.T, W)
        K = U.T 
        # MAJ de l etat
        a = a + K @ v
        # MAJ de la variance
        P = P - K @ Z @ P
    # Retourne la log vraisemblance
    return float(ll)

def kalman_loglike_2f(p: TwoFactorParams, y, x):
    '''Calcule la log vraimensemblance pour 2 facteurs'''
    m = p.m
    T_low = len(y)
    T_high = T_low * m

    G = np.diag([p.rho1, p.rho2, p.d, p.d])
    Q = np.diag([p.sig2_f1, p.sig2_f2, p.sig2_uy, p.sig2_ux])

    a = np.zeros(4)
    P = np.eye(4) * 10.0
    ll = 0.0
    low_idx = 0
    two_pi = np.log(2.0*np.pi)

    for t in range(T_high):
        j = (t % m) + 1

        # predict
        a = G @ a
        P = G @ P @ G.T + Q

        # measurement
        if j < m:
            Z = np.array([[1, 0, 0, 1]])
            y_obs = np.array([x[t, 0]])
        else:
            Z = np.array([[1, 1, 1, 0],
                          [1, 0, 0, 1]])
            y_obs = np.array([y[low_idx], x[t, 0]])
            low_idx += 1

        v = y_obs - Z @ a
        S = Z @ P @ Z.T  # H=0 ici

        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return -np.inf

        tmp = np.linalg.solve(L, v)
        Sinv_v = np.linalg.solve(L.T, tmp)
        quad = float(v.T @ Sinv_v)
        logdet = 2.0*np.sum(np.log(np.diag(L)))
        k = len(y_obs)

        ll += -0.5*(logdet + quad + k*two_pi)

        # update
        PZt = P @ Z.T
        W = np.linalg.solve(L, PZt.T)
        U = np.linalg.solve(L.T, W)
        K = U.T
        a = a + K @ v
        P = P - K @ Z @ P

    return float(ll)


def fit_kalman_mle(y: np.ndarray, x: np.ndarray, m=3) -> OneFactorParams:
    '''Estime les parametres d un model a un facteur 
    par maximum de vraisemblance via le filtre de Kalman'''
    n_x = x.shape[1]

    # Fonction objectif
    def neg_ll(theta):
        eps = 1e-8 # securite numerique
        rho = np.tanh(theta[0])
        d = np.tanh(theta[1])
        sig2_f = np.exp(theta[2]) + eps
        sig2_uy = np.exp(theta[3]) + eps
        sig2_ux = np.exp(theta[4:4+n_x]) + eps

        # objet parametre
        p = OneFactorParams(
            m=m,
            n_x=n_x,
            lam_y=1.0,
            lam_x=np.ones(n_x),
            rho=rho,
            d=d,
            sig2_f=sig2_f,
            sig2_uy=sig2_uy,
            sig2_ux=sig2_ux
        )

        return -kalman_loglike_full(p, y, x)

    # valeur init
    theta0 = np.array(
        [np.arctanh(0.2), np.arctanh(0.1),
         np.log(1.0), np.log(1.0)] + [np.log(1.0)] * n_x
    )

    # optimisation numerique
    res = minimize(neg_ll, theta0, method="L-BFGS-B")

    # parametres estimes
    rho = np.tanh(res.x[0])
    d = np.tanh(res.x[1])
    sig2_f = np.exp(res.x[2])
    sig2_uy = np.exp(res.x[3])
    sig2_ux = np.exp(res.x[4:4+n_x])

    return OneFactorParams(
        m=m,
        n_x=n_x,
        lam_y=1.0,
        lam_x=np.ones(n_x),
        rho=rho,
        d=d,
        sig2_f=sig2_f,
        sig2_uy=sig2_uy,
        sig2_ux=sig2_ux
    )

def fit_kalman_mle_2f(y: np.ndarray, x: np.ndarray, m=3) -> TwoFactorParams:
    def neg_ll(theta):
        eps = 1e-8
        rho1 = np.tanh(theta[0])
        rho2 = np.tanh(theta[1])
        d    = np.tanh(theta[2])

        sig2_f1 = np.exp(theta[3]) + eps
        sig2_f2 = np.exp(theta[4]) + eps
        sig2_uy = np.exp(theta[5]) + eps
        sig2_ux = np.exp(theta[6]) + eps

        p = TwoFactorParams(
            m=m, rho1=rho1, rho2=rho2, d=d,
            sig2_f1=sig2_f1, sig2_f2=sig2_f2,
            sig2_uy=sig2_uy, sig2_ux=sig2_ux
        )
        return -kalman_loglike_2f(p, y, x)

    theta0 = np.array([
        np.arctanh(0.5), np.arctanh(0.2), np.arctanh(0.1),
        np.log(1.0), np.log(1.0), np.log(1.0), np.log(1.0)
    ])

    res = minimize(neg_ll, theta0, method="L-BFGS-B")

    return TwoFactorParams(
        m=m,
        rho1=np.tanh(res.x[0]),
        rho2=np.tanh(res.x[1]),
        d=np.tanh(res.x[2]),
        sig2_f1=np.exp(res.x[3]),
        sig2_f2=np.exp(res.x[4]),
        sig2_uy=np.exp(res.x[5]),
        sig2_ux=np.exp(res.x[6]),
    )


def periodic_steady_state_kf_2f(p: TwoFactorParams):
    """
    Calcule les gains de Kalman périodiques (steady-state)
    pour le modèle à deux facteurs
    """
    m = p.m
    dim = p.dim_state

    # Matrice de transition
    G = np.diag([p.rho1, p.rho2, p.d, p.d])

    # Variance des innovations
    Q = np.diag([p.sig2_f1, p.sig2_f2, p.sig2_uy, p.sig2_ux])

    # Matrices de mesure selon la sous-période
    Z_list = []
    H_list = []

    for j in range(1, m + 1):
        if j < m:
            # x = f1 + u_x
            Z = np.array([[1, 0, 0, 1]])
            H = np.zeros((1, 1))
        else:
            # y = f1 + f2 + u_y
            # x = f1 + u_x
            Z = np.array([
                [1, 1, 1, 0],
                [1, 0, 0, 1]
            ])
            H = np.zeros((2, 2))

        Z_list.append(Z)
        H_list.append(H)

    # Initialisation Riccati
    P = np.eye(dim) * 10.0

    for _ in range(RICCATI_MAX_ITERS):
        P_old = P.copy()
        for j in range(m):
            Z = Z_list[j]
            H = H_list[j]

            S = Z @ P @ Z.T + H
            K = P @ Z.T @ np.linalg.inv(S)
            P = P - K @ Z @ P
            P = G @ P @ G.T + Q

        if np.max(np.abs(P - P_old)) < RICCATI_TOL:
            break

    # Gains de Kalman steady-state
    K_list = []
    for j in range(m):
        Z = Z_list[j]
        H = H_list[j]
        S = Z @ P @ Z.T + H
        K = P @ Z.T @ np.linalg.inv(S)
        K_list.append(K)

    return {
        "G": G,
        "Z_list": Z_list,
        "K_list": K_list,
    }


def run_periodic_kf_filter_2f(kf, y, x):
    """
    Applique le filtre de Kalman périodique à deux facteurs
    et retourne les états filtrés aux dates basse fréquence
    """
    G = kf["G"]
    Z_list = kf["Z_list"]
    K_list = kf["K_list"]

    m = len(Z_list)
    T_low = len(y)
    T_high = T_low * m

    a = np.zeros(G.shape[0])
    states_low = []

    low_idx = 0

    for t in range(T_high):
        j = t % m

        # Prediction
        a = G @ a

        # Observation
        if j < m - 1:
            y_obs = np.array([x[t, 0]])
        else:
            y_obs = np.array([y[low_idx], x[t, 0]])
            low_idx += 1

        Z = Z_list[j]
        K = K_list[j]

        # Innovation
        v = y_obs - Z @ a

        # Mise à jour
        a = a + K @ v

        # Sauvegarde à la fin de chaque période LF
        if j == m - 1:
            states_low.append(a.copy())

    return np.array(states_low)


def kalman_filter_forecast(y, x, h=1, m=3):
    '''
    Calcule des prev à horizon h  via le filtre de Kalman
    y: (T_BF,) variable basse frequence
    x: (T_HF, n_x) avec T_HF = T_BF*m variables haute frequence

    '''
    # Estimation des parametres
    p_hat = fit_kalman_mle(y, x, m=m)
    # Filtre de Kalman
    kf = periodic_steady_state_kf(p_hat)

    _, states_low = run_periodic_kf_filter(kf, y, x)


    forecasts = []
    actuals = []
    # Prevision recursive
    for t in range(len(y) - h):
        forecasts.append(forecast_y_from_state(p_hat, states_low[t], h))
        actuals.append(y[t + h])

    return np.array(forecasts), np.array(actuals)

def kalman_ic_1f(y, x, m=3):
    # Critères pour aic ou bic 1 facteur
    p = fit_kalman_mle(y, x, m=m)
    ll = kalman_loglike_full(p, y, x)
    k = 5   # rho, d, sig2_f, sig2_uy, sig2_ux
    return ll, k, p


def kalman_ic_2f(y, x, m=3):
    p = fit_kalman_mle_2f(y, x, m=m)
    ll = kalman_loglike_2f(p, y, x)
    k = 7  # rho1,rho2,d + 4 variances
    return ll, k, p

def forecast_y_from_state_2f(p: TwoFactorParams, state, h):
    f1, f2, uy, _ = state
    return (p.rho1**(p.m*h))*f1 + (p.rho2**(p.m*h))*f2 + (p.d**(p.m*h))*uy


#============================
# MIDAS & ADL-MIDAS
#============================
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

#============================
# DGP
#============================
def simulate_one_factor_dgp(T=40, m=3, rho=0.9, d=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # HF taille
    Th = T*m

    # Innovations (loi normale)
    eta = np.random.normal(size=Th)
    eps1 = np.random.normal(size=Th)
    eps2 = np.random.normal(size=Th)

    # Facteur Latent
    f = np.zeros(Th)
    for t in range(1, Th):
        f[t] = rho * f[t-1] + eta[t]
    
    # Erreurs
    u1 = np.zeros(Th)
    u2 = np.zeros(Th)
    # Simulation
    for t in range(1, Th):
        u1[t] = d * u1[t-1] + eps1[t]
        u2[t] = d * u2[t-1] + eps2[t]
    
    # Observations
    y_star = f + u1
    x = f + u2

    # LF aggregation 
    y = y_star[m-1::m]

    return y, x.reshape(-1,1), f

def simulate_two_factor_dgp(
        T=40,
        m=3,
        rho=0.9,
        d=0.5,
        seed=None
):
    if seed is not None:
        np.random.seed(seed)
    Th = T * m

    # Innovations
    eta1 = np.random.normal(size=Th)
    eta2 = np.random.normal(size=Th)
    epsy = np.random.normal(size=Th)
    epsx = np.random.normal(size=Th)

    # Factors
    f1 = np.zeros(Th)
    f2 = np.zeros(Th)
    for t in range(1, Th):
        f1[t] = rho * f1[t - 1] + eta1[t]
        f2[t] = rho * f2[t - 1] + eta2[t]
    
    # Erreurs
    uy = np.zeros(Th)
    ux = np.zeros(Th)
    for t in range(1, Th):
        uy[t] = d * uy[t - 1] + epsy[t]
        ux[t] = d * ux[t - 1] + epsx[t]
    
    # Observations
    y_star = f1 + f2 + uy
    x = f1 + ux

    # LF
    y = y_star[m-1::m]

    return y, x.reshape(-1,1), f1, f2

#============================
# Utilities
#============================
def rmspe(forecast, actual):
    """Retourne l erreur de prevision quadratique moyenne"""
    return np.sqrt(np.mean(((forecast - actual)) ** 2))

def gaussian_loglike(residuals):
    """Log-vraisemblance sous hypothese d erreurs gaussiennes"""
    residuals = np.asarray(residuals)
    # clip pour éviter explosions numériques
    residuals = np.clip(residuals, -1e6, 1e6)
    T = len(residuals)
    sigma2 = np.mean(residuals**2)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return -np.inf
    return -0.5 * T * (np.log(2*np.pi*sigma2) + 1)

def aic(loglike: float, k: int) -> float:
    """AIC critère"""
    return -2 * loglike + 2 * k


def bic(loglike: float, k: int, T_low: int, m: int, n_x: int = 1) -> float:
    n_obs = T_low + (T_low * m) * n_x
    return -2 * loglike + k * np.log(n_obs)



def kalman_ic(y, x, m=3):
    """Critères pour Kalman un facteur"""
    p_hat = fit_kalman_mle(y, x, m=m)
    ll = kalman_loglike_full(p_hat, y, x)
    k = 2 + 3   # rho, d + 3 variances
    return ll, k, p_hat

def hf_lag_at_low_t(x: np.ndarray, t: int, m: int, lag_hf: int) -> float:
    
    if x.ndim == 2:
        x1 = x[:, 0]
    else:
        x1 = x
    idx = t * m - 1 - lag_hf
    if idx < 0 or idx >= len(x1):
        return np.nan
    return float(x1[idx])

#============================
# Monte Carlo Simulation
#============================
def monte_carlo_simulation_1(
        N=500, T=40, m=3, rho=0.9, d=0.5, h=1
):
    rmspe_midas = []
    rmspe_adl_midas = []
    rmspe_kf = []

    " Boucle sur le nombre de simulations "
    for i in range(N):
        y, x, _= simulate_one_factor_dgp(T=T, m=m, rho=rho, d=d, seed=i)
        # MIDAS Forecast
        midas_forecast, midas_actual = regular_midas_forecast(y, x, h=h, m=m)
        rmspe_midas.append(rmspe(midas_forecast, midas_actual))

        # ADL-MIDAS Forecast
        adl_forecast, adl_actual = multiplicative_midas_forecast(y, x, h=h, m=m)
        rmspe_adl_midas.append(rmspe(adl_forecast, adl_actual))

        kf_forecast, kf_actual = kalman_filter_forecast(y, x, h=h, m=m)
        rmspe_kf.append(rmspe(kf_forecast, kf_actual))

    return {
        "KF / MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_midas),
        "KF / ADL-MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_adl_midas),
    }

def monte_carlo_simulation_2(
        N=500,
        T=40,
        m=3,
        rho=0.9,
        d=0.5,
        h=1
):
    rmspe_midas = []
    rmspe_adl   = []
    rmspe_kf    = []

    for i in range(N):
        y, x, _, _ = simulate_two_factor_dgp(
            T=T,
            m=m,
            rho=rho,
            d=d,
            seed=i
        )

        # MIDAS Forecast
        midas_forecast, midas_actual = regular_midas_forecast(y, x, h=h, m=m)
        rmspe_midas.append(rmspe(midas_forecast, midas_actual))

        # ADL-MIDAS Forecast
        adl_forecast, adl_actual = multiplicative_midas_forecast(y, x, h=h, m=m)
        rmspe_adl.append(rmspe(adl_forecast, adl_actual))

        # Kalman Filter Forecast (MISSPECIFIED)
        try:    
            kf_forecast, kf_actual = kalman_filter_forecast(y, x, h=h, m=m)
            rmspe_kf.append(rmspe(kf_forecast, kf_actual))
        except RuntimeError:
            # Non convergence de Riccati
            continue
    return {
        "KF / MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_midas),
        "KF / ADL-MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_adl),
    }


def monte_carlo_simulation_3(
    N=500,
    T=40,
    m=3,
    rho=0.9,
    d=0.5,
    h=1,
    criterion="AIC"
):
    rmspe_kf = []
    rmspe_midas = []
    rmspe_adl = []

    n2 = 0
    for i in range(N):
        # DGP : 1 facteur
        y, x, _ = simulate_one_factor_dgp(T=T, m=m, rho=rho, d=d, seed=i)

        # MIDAS
        f_m, a_m = regular_midas_forecast(y, x, h=h, m=m)
        rmspe_midas.append(rmspe(f_m, a_m))

        # ADL-MIDAS
        f_a, a_a = multiplicative_midas_forecast(y, x, h=h, m=m)
        rmspe_adl.append(rmspe(f_a, a_a))

        # Kalman: 
        ll1, k1, p1 = kalman_ic_1f(y, x, m)
        ll2, k2, p2 = kalman_ic_2f(y, x, m)

        if criterion == "AIC":
            ic1 = aic(ll1, k1)
            ic2 = aic(ll2, k2)
        else:
            ic1 = bic(ll1, k1, T_low=len(y), m=m, n_x=x.shape[1])
            ic2 = bic(ll2, k2, T_low=len(y), m=m, n_x=x.shape[1])


        if ic1 <= ic2:
            model_type = "1f"
            p_hat = p1
            kf = periodic_steady_state_kf(p_hat)
            _, states_low = run_periodic_kf_filter(kf, y, x)

        else:
            model_type = "2f"
            p_hat = p2
            kf = periodic_steady_state_kf_2f(p_hat)
            states_low = run_periodic_kf_filter_2f(kf, y, x)
            n2 += 1

        # Prévisions
        fcast = []
        actual = []

        for t in range(len(y) - h):

            if model_type =="1f":
                fcast.append(forecast_y_from_state(p_hat, states_low[t], h))
            else:
                # Prevision 2 facteurs
                fcast.append(forecast_y_from_state_2f(p_hat, states_low[t], h))
            
            actual.append(y[t + h])

        rmspe_kf.append(rmspe(np.array(fcast), np.array(actual)))
    print("share 2-factor selected =", n2/N)
    return {
        "KF / MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_midas),
        "KF / ADL-MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_adl),
    }

#===========================
# Panels
#===========================

def run_panel_simulation_1(
    h: int,
    N: int = 500,
    T: int = 40,
    m: int = 3
):
    """
    Le but de cette fonction est de reproduire les tables du papier
    ie les ratios de RMSPE entre Kalman et MIDAS / ADL-MIDAS
    """
    res_midas = pd.DataFrame(index=D_GRID, columns=RHO_GRID, dtype=float)
    res_adl   = pd.DataFrame(index=D_GRID, columns=RHO_GRID, dtype=float)

    for d in D_GRID:
        for rho in RHO_GRID:
            out = monte_carlo_simulation_1(
                N=N,
                T=T,
                m=m,
                rho=rho,
                d=d,
                h=h
            )

            res_midas.loc[d, rho] = out["KF / MIDAS"]
            res_adl.loc[d, rho]   = out["KF / ADL-MIDAS"]

            print(f"h={h} | d={d:>4} | rho={rho:>5} | done")

    return res_midas, res_adl

def run_panel_simulation_2(
    h: int,
    N: int = 500,
    T: int = 40,
    m: int = 3
):
    res_midas = pd.DataFrame(index=D_GRID, columns=RHO_GRID)
    res_adl   = pd.DataFrame(index=D_GRID, columns=RHO_GRID)

    for d in D_GRID:
        for rho in RHO_GRID:
            out = monte_carlo_simulation_2(
                N=N,
                T=T,
                m=m,
                rho=rho,
                d=d,
                h=h
            )

            res_midas.loc[d, rho] = out["KF / MIDAS"]
            res_adl.loc[d, rho]   = out["KF / ADL-MIDAS"]

            print(f"h={h} | d={d:>4} | rho={rho:>5} | done")

    return res_midas, res_adl

def run_panel_simulation_3(
    h: int,
    criterion: str,
    N: int = 500,
    T: int = 40,
    m: int = 3
):
    
    res_midas = pd.DataFrame(index=D_GRID, columns=RHO_GRID, dtype=float)
    res_adl   = pd.DataFrame(index=D_GRID, columns=RHO_GRID, dtype=float)

    for d in D_GRID:
        for rho in RHO_GRID:
            out = monte_carlo_simulation_3(
                N=N,
                T=T,
                m=m,
                rho=rho,
                d=d,
                h=h,
                criterion=criterion
            )

            res_midas.loc[d, rho] = out["KF / MIDAS"]
            res_adl.loc[d, rho]   = out["KF / ADL-MIDAS"]

            print(f"{criterion} | h={h} | d={d:>4} | rho={rho:>5} | done")

    return res_midas, res_adl


#===========================
# Tables
#===========================
def generate_table_4A(N=500):
    print("=== Panel A: One-Factor DGP, h = 1 ===")
    A_midas, A_adl = run_panel_simulation_1(h=1, N=N)

    print("=== Panel B: One-Factor DGP, h = 4 ===")
    B_midas, B_adl = run_panel_simulation_1(h=4, N=N)

    return {
        "Panel A (h=1) - Regular MIDAS": A_midas,
        "Panel A (h=1) - Multiplicative MIDAS": A_adl,
        "Panel B (h=4) - Regular MIDAS": B_midas,
        "Panel B (h=4) - Multiplicative MIDAS": B_adl,
    }

def generate_table_4B(N=500):
    print("=== Table 4 – Panel C (Two-Factor DGP, h=1) ===")
    C_midas, C_adl = run_panel_simulation_2(h=1, N=N)

    print("=== Table 4 – Panel D (Two-Factor DGP, h=4) ===")
    D_midas, D_adl = run_panel_simulation_2(h=4, N=N)

    return {
        "Panel C (h=1) - Regular MIDAS": C_midas,
        "Panel C (h=1) - Multiplicative MIDAS": C_adl,
        "Panel D (h=4) - Regular MIDAS": D_midas,
        "Panel D (h=4) - Multiplicative MIDAS": D_adl,
    }

def generate_table_5(N: int = 500):
    

    print("=== Table 5, Panel A: AIC, h = 1 ===")
    A_midas, A_adl = run_panel_simulation_3(
        h=1, criterion="AIC", N=N
    )

    print("=== Table 5, Panel B: BIC, h = 1 ===")
    B_midas, B_adl = run_panel_simulation_3(
        h=1, criterion="BIC", N=N
    )

    print("=== Table 5, Panel C: AIC, h = 4 ===")
    C_midas, C_adl = run_panel_simulation_3(
        h=4, criterion="AIC", N=N
    )

    print("=== Table 5, Panel D: BIC, h = 4 ===")
    D_midas, D_adl = run_panel_simulation_3(
        h=4, criterion="BIC", N=N
    )

    return {
        "Panel A (AIC, h=1) - Regular MIDAS": A_midas,
        "Panel A (AIC, h=1) - Multiplicative MIDAS": A_adl,
        "Panel B (BIC, h=1) - Regular MIDAS": B_midas,
        "Panel B (BIC, h=1) - Multiplicative MIDAS": B_adl,
        "Panel C (AIC, h=4) - Regular MIDAS": C_midas,
        "Panel C (AIC, h=4) - Multiplicative MIDAS": C_adl,
        "Panel D (BIC, h=4) - Regular MIDAS": D_midas,
        "Panel D (BIC, h=4) - Multiplicative MIDAS": D_adl,
    }


tables_4A = generate_table_4A(N=500)
tables_4A["Panel A (h=1) - Regular MIDAS"].to_excel("Table_4A_PanelA_MIDAS.xlsx")
tables_4A["Panel A (h=1) - Multiplicative MIDAS"].to_excel("Table_4A_PanelA_Multiplicative_MIDAS.xlsx")
tables_4A["Panel B (h=4) - Regular MIDAS"].to_excel("Table_4A_PanelB_MIDAS.xlsx")
tables_4A["Panel B (h=4) - Multiplicative MIDAS"].to_excel("Table_4A_PanelB_Multiplicative_MIDAS.xlsx")


tables_4B = generate_table_4B(N=500)
tables_4B["Panel C (h=1) - Regular MIDAS"].to_excel("Table_4_PanelC_MIDAS.xlsx")
tables_4B["Panel C (h=1) - Multiplicative MIDAS"].to_excel("Table_4_PanelC_ADL_MIDAS.xlsx")
tables_4B["Panel D (h=4) - Regular MIDAS"].to_excel("Table_4_PanelD_MIDAS.xlsx")
tables_4B["Panel D (h=4) - Multiplicative MIDAS"].to_excel("Table_4_PanelD_ADL_MIDAS.xlsx")


tables_5 = generate_table_5(N=500)
tables_5["Panel A (AIC, h=1) - Regular MIDAS"].to_excel("Table_5_PanelA_MIDAS.xlsx")
tables_5["Panel A (AIC, h=1) - Multiplicative MIDAS"].to_excel("Table_5_PanelA_Multiplicative_MIDAS.xlsx")
tables_5["Panel B (BIC, h=1) - Regular MIDAS"].to_excel("Table_5_PanelB_MIDAS.xlsx")
tables_5["Panel B (BIC, h=1) - Multiplicative MIDAS"].to_excel("Table_5_PanelB_Multiplicative_MIDAS.xlsx")
tables_5["Panel C (AIC, h=4) - Regular MIDAS"].to_excel("Table_5_PanelC_MIDAS.xlsx")
tables_5["Panel C (AIC, h=4) - Multiplicative MIDAS"].to_excel("Table_5_PanelC_Multiplicative_MIDAS.xlsx")
tables_5["Panel D (BIC, h=4) - Regular MIDAS"].to_excel("Table_5_PanelD_MIDAS.xlsx")
tables_5["Panel D (BIC, h=4) - Multiplicative MIDAS"].to_excel("Table_5_PanelD_Multiplicative_MIDAS.xlsx")
