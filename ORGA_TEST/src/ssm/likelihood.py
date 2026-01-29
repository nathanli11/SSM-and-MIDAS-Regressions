import numpy as np
from scipy.optimize import minimize
from params import OneFactorParams, TwoFactorParams
from measurement import build_measurement_mats

def kalman_loglike_full(p: OneFactorParams, y: np.ndarray, x: np.ndarray) -> float:
    """
    Full (time-varying P) KF loglik with periodic measurement matrices.
    y: (T_low,)
    x: (T_high, n_x) with T_high = T_low*m
    """
    assert x.shape[0] == y.shape[0] * p.m
    assert x.shape[1] == p.n_x

    # build periodic mats
    Z_list, H_list = build_measurement_mats(p)
    G = np.diag([p.rho] + [p.d] * (1 + p.n_x))
    Q = np.diag([p.sig2_f, p.sig2_uy] + list(p.sig2_ux))

    dim = p.dim_state
    a = np.zeros(dim)
    P = np.eye(dim) * 10.0  # diffuse-ish init

    ll = 0.0
    low_idx = 0
    two_pi = np.log(2.0 * np.pi)

    for t_high in range(x.shape[0]):
        j = (t_high % p.m) + 1      # 1..m
        jj = j - 1                  # 0..m-1

        # predict
        a = G @ a
        P = G @ P @ G.T + Q

        # measurement
        if j < p.m:
            y_obs = x[t_high, :]                   # (n_x,)
            Z = Z_list[jj]                         # (n_x, dim)
            H = H_list[jj]                         # (n_x, n_x)
        else:
            y_obs = np.concatenate([[y[low_idx]], x[t_high, :]])  # (1+n_x,)
            Z = Z_list[jj]                         # (1+n_x, dim)
            H = H_list[jj]                         # (1+n_x, 1+n_x)
            low_idx += 1

        v = y_obs - (Z @ a)
        S = Z @ P @ Z.T + H

        # numerical stability
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return -np.inf

        # solve S^{-1}v using chol
        tmp = np.linalg.solve(L, v)
        Sinv_v = np.linalg.solve(L.T, tmp)
        quad = float(v.T @ Sinv_v)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        k = len(y_obs)

        ll += -0.5 * (logdet + quad + k * two_pi)

        # update
        # K = P Z' S^{-1} via chol solves
        # compute PZ' then solve for each column
        PZt = P @ Z.T
        # solve S^{-1} * (Z P)' = S^{-1} * (PZt)'
        # using chol: solve L w = (PZt)' then L.T u = w
        W = np.linalg.solve(L, PZt.T)
        U = np.linalg.solve(L.T, W)
        K = U.T  # (dim, k)

        a = a + K @ v
        P = P - K @ Z @ P

    return float(ll)

def kalman_loglike_2f(p: TwoFactorParams, y, x):
    m = p.m
    T_low = len(y)
    T_high = T_low * m

    G = np.diag([p.rho1, p.rho2, p.d, p.d])
    Q = np.diag([p.sig2_f1, p.sig2_f2, p.sig2_uy, p.sig2_ux])

    a = np.zeros(4)
    P = np.eye(4) * 10
    ll = 0.0
    low_idx = 0

    for t in range(T_high):
        j = (t % m) + 1
        a = G @ a
        P = G @ P @ G.T + Q

        if j < m:
            Z = np.array([[1, 0, 0, 1]])   # x = f1 + u_x
            y_obs = np.array([x[t, 0]])
        else:
            Z = np.array([
                [1, 1, 1, 0],             # y = f1 + f2 + u_y
                [1, 0, 0, 1]              # x
            ])
            y_obs = np.array([y[low_idx], x[t, 0]])
            low_idx += 1

        v = y_obs - Z @ a
        S = Z @ P @ Z.T
        ll += -0.5 * (np.log(np.linalg.det(S)) + v.T @ np.linalg.solve(S, v))
        K = P @ Z.T @ np.linalg.inv(S)
        a = a + K @ v
        P = P - K @ Z @ P

    return float(ll)

def fit_kalman_mle(y: np.ndarray, x: np.ndarray, m=3) -> OneFactorParams:
    n_x = x.shape[1]

    def neg_ll(theta):
        eps = 1e-8
        rho = np.tanh(theta[0])
        d = np.tanh(theta[1])
        sig2_f = np.exp(theta[2]) + eps
        sig2_uy = np.exp(theta[3]) + eps
        sig2_ux = np.exp(theta[4:4+n_x]) + eps

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

    theta0 = np.array(
        [np.arctanh(0.2), np.arctanh(0.1),
         np.log(1.0), np.log(1.0)] + [np.log(1.0)] * n_x
    )

    res = minimize(neg_ll, theta0, method="L-BFGS-B")

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
