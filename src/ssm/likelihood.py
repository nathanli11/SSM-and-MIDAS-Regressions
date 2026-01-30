import numpy as np
from scipy.optimize import minimize
from src.ssm.params import OneFactorParams, TwoFactorParams
from src.ssm.measurement import build_measurement_mats

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
        p = OneFactorParams.from_nx(
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

    return OneFactorParams.from_nx(
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

