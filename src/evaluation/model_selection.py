import numpy as np
from src.ssm.likelihood import kalman_loglike_full, kalman_loglike_2f, fit_kalman_mle
from src.ssm.params import TwoFactorParams

# ----------------------------------------------------
# GRID as in the paper
# ----------------------------------------------------
RHO_GRID = [-0.9, -0.5, 0.5, 0.95]
D_GRID   = [-0.9, -0.5, 0.0, 0.5, 0.95]

RICCATI_MAX_ITERS = 50_000
RICCATI_TOL = 1e-12

K_BAR = 40                 # lag truncation K̄ used in Eqs (3.4)-(3.5) and Appendix weight vectors
KF_WARMUP_PERIODS = 400    # warmup in low-freq periods when extracting impulse-response weights
OPT_TOL = 1e-10

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

def bic(loglike: float, k: int, T: int) -> float:
    return -2 * loglike + k * np.log(T)

def rmspe(forecast, actual):
    """Retourne l erreur de prevision quadratique moyenne"""
    return np.sqrt(np.mean(((forecast - actual)) ** 2))


def kalman_ic_1f(y, x, m=3):
    # Critères pour aic ou bic 1 facteur
    p = fit_kalman_mle(y, x, m=m)
    ll = kalman_loglike_full(p, y, x)
    k = 5   # rho, d, sig2_f, sig2_uy, sig2_ux
    return ll, k, p

def kalman_ic_2f(y, x, m=3):
    # critères pour aic ou bic 2 facteurs
    p = TwoFactorParams(m=m)
    ll = kalman_loglike_2f(p, y, x)
    k = 6   # rho1, rho2, sig2_f1, sig2_f2, sig2_uy, sig2_ux
    return ll, k, p

