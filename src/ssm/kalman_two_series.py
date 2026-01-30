import numpy as np
from scipy.optimize import minimize


def kalman_loglik_two_series(df, rho, d1, d2, gamma1, gamma2, sig2_u1, sig2_u2,
                             names, x0=None, P0=None, jitter_R=1e-8):
    """
    Log-vraisemblance (Kalman) du modèle à deux séries mensuelles:
    """
    y = df.loc[:, list(names)].values.astype(float)
    T, k = y.shape
    assert k == 2

    # Matrices
    G = np.array([[rho, 0.0, 0.0],
                  [0.0, d1, 0.0],
                  [0.0, 0.0, d2]], dtype=float)
    
    # sigma_f^2 = 1 (normalisation)
    Q = np.diag([1.0, sig2_u1, sig2_u2]).astype(float)

    Z = np.array([[gamma1, 1.0, 0.0],
                  [gamma2, 0.0, 1.0]], dtype=float)

    R = jitter_R * np.eye(2)

    p = 3
    I = np.eye(p)

    if x0 is None:
        a_t = np.zeros(p)
    else:
        a_t = np.asarray(x0, float).reshape(p)

    if P0 is None:
        P_t = np.eye(p) * 1e2
    else:
        P_t = np.asarray(P0, float).reshape(p, p)

    ll = 0.0

    for t in range(T):
        # Prediction
        a_pred = G @ a_t
        P_pred = G @ P_t @ G.T + Q

        yt = y[t, :]
        mask = ~np.isnan(yt)
        m = int(mask.sum())

        if m == 0:
            a_t, P_t = a_pred, P_pred
            continue

        y_obs = yt[mask]
        Z_obs = Z[mask, :]
        R_obs = R[np.ix_(mask, mask)]

        v = y_obs - (Z_obs @ a_pred)
        S = Z_obs @ P_pred @ Z_obs.T + R_obs

        # contribution log-vraisemblance
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return -np.inf  # S doit être SPD

        try:
            Sinv_v = np.linalg.solve(S, v)
        except np.linalg.LinAlgError:
            S = S + 1e-8 * np.eye(m)
            Sinv_v = np.linalg.solve(S, v)

        ll += -0.5 * (logdet + v @ Sinv_v + m * np.log(2*np.pi))

        # MAJ filtre
        PZt = P_pred @ Z_obs.T
        try:
            K = np.linalg.solve(S, PZt.T).T
        except np.linalg.LinAlgError:
            S = S + 1e-8 * np.eye(m)
            K = np.linalg.solve(S, PZt.T).T

        a_t = a_pred + K @ v
        KH = K @ Z_obs
        P_t = (I - KH) @ P_pred @ (I - KH).T + K @ R_obs @ K.T

    return ll

def fit_ssm_ml(df_hf, names):
    """
    Estime (rho,d1,d2,gamma1,gamma2,sig2_u1,sig2_u2) par ML (Kalman innovations).
    """
    def unpack(theta):
        trho, td1, td2, g1, g2, lsu1, lsu2 = theta
        rho = np.tanh(trho)
        d1  = np.tanh(td1)
        d2  = np.tanh(td2)
        sig2_u1 = np.exp(lsu1)
        sig2_u2 = np.exp(lsu2)
        return rho, d1, d2, g1, g2, sig2_u1, sig2_u2

    def neg_ll(theta):
        rho, d1, d2, g1, g2, s1, s2 = unpack(theta)
        ll = kalman_loglik_two_series(df_hf, rho, d1, d2, g1, g2, s1, s2, names=(names[0], names[1]))
        if not np.isfinite(ll):
            return 1e12
        return -ll

    theta0 = np.array([
        np.arctanh(0.8),
        np.arctanh(0.5),
        np.arctanh(0.5),
        1.0,
        1.0,
        np.log(0.5),
        np.log(0.5),
    ], float)

    res = minimize(neg_ll, theta0, method="L-BFGS-B", options={"maxiter": 2000})
    if not res.success:
        raise RuntimeError(res.message)

    rho, d1, d2, g1, g2, s1, s2 = unpack(res.x)
    return {"rho": rho, "d1": d1, "d2": d2, "gamma1": g1, "gamma2": g2, "sig2_u1": s1, "sig2_u2": s2}

def build_ssm_two_series_ar1_ml(params, jitter_R=1e-8):
    """"
    Construit les matrices SSM à partir des paramètres ML estimés.
    """
    rho = params['rho']
    d1 = params['d1']
    d2 = params['d2']
    gamma1 = params['gamma1']
    gamma2 = params['gamma2']
    sig2_u1 = params['sig2_u1']
    sig2_u2 = params['sig2_u2']


    G = np.array([[rho, 0.0, 0.0],
                  [0.0, d1, 0.0],
                  [0.0, 0.0, d2]], dtype=float)
    Q = np.diag([1.0, sig2_u1, sig2_u2]).astype(float)  # sigma_f^2 fixé à 1
    a = np.zeros(3)
    Z = np.array([[gamma1, 1.0, 0.0],
                  [gamma2, 0.0, 1.0]], dtype=float)
    R = jitter_R * np.eye(2)
    return G, Q, a, Z, R

# Filtre de Kalman pour SSM à deux séries mensuelles
def kalman_filter_states(df_hf, params, names, jitter_R=1e-8):
    """Filtre (renvoie a_pred / a_filt) pour utilisation en forecast."""
    y = df_hf[names].values.astype(float)
    idx = df_hf.index
    T = len(idx)

    rho = params["rho"]; d1=params["d1"]; d2=params["d2"]
    g1 = params["gamma1"]; g2=params["gamma2"]
    s1 = params["sig2_u1"]; s2=params["sig2_u2"]

    G = np.array([[rho, 0.0, 0.0],
                  [0.0, d1, 0.0],
                  [0.0, 0.0, d2]], float)
    Q = np.diag([1.0, s1, s2]).astype(float)
    Z = np.array([[g1, 1.0, 0.0],
                  [g2, 0.0, 1.0]], float)
    R = jitter_R*np.eye(2)

    a = np.zeros(3)
    P = np.eye(3)*1e2
    I = np.eye(3)

    a_pred = np.zeros((T,3))
    a_filt = np.zeros((T,3))

    for t in range(T):
        ap = G @ a
        Pp = G @ P @ G.T + Q

        a_pred[t] = ap

        yt = y[t]
        mask = ~np.isnan(yt)
        if mask.sum() == 0:
            a, P = ap, Pp
        else:
            y_obs = yt[mask]
            Z_obs = Z[mask,:]
            R_obs = R[np.ix_(mask,mask)]
            v = y_obs - Z_obs @ ap
            S = Z_obs @ Pp @ Z_obs.T + R_obs
            PZt = Pp @ Z_obs.T
            try:
                K = np.linalg.solve(S, PZt.T).T
            except np.linalg.LinAlgError:
                S = S + 1e-8*np.eye(S.shape[0])
                K = np.linalg.solve(S, PZt.T).T
            a = ap + K @ v
            KH = K @ Z_obs
            P = (I - KH) @ Pp @ (I - KH).T + K @ R_obs @ K.T

        a_filt[t] = a

    return {"index": idx, "a_pred": a_pred, "a_filt": a_filt, "G": G, "Z": Z}


# def kalman_filter_minimal(df_x, G, Q, a, Z, R, names, x0=None, P0=None, scaler=None):
#     """
#     Filtre de Kalman minimal (prévision et filtrage) pour données avec NaN.
#     """
#     y = df_x.reindex(columns=names).values.astype(float)
#     idx = df_x.index
#     T, k = y.shape
#     p = G.shape[0]
#     I = np.eye(p)

#     if x0 is None:
#         x0 = np.zeros(p, dtype=float)
#     else:
#         x0 = np.asarray(x0, dtype=float).reshape(p)

#     if P0 is None:
#         P0 = np.eye(p, dtype=float) * 1e2
#     else:
#         P0 = np.asarray(P0, dtype=float).reshape(p, p)

#     a = np.asarray(a, dtype=float).reshape(p)

#     a_pred = np.zeros((T, p))
#     P_pred = np.zeros((T, p, p))
#     a_filt = np.zeros((T, p))
#     P_filt = np.zeros((T, p, p))

#     a_t = x0.copy()
#     P_t = P0.copy()

#     if scaler is not None:
#         mu = np.asarray(scaler.mean_, dtype=float)
#         sd = np.asarray(scaler.scale_, dtype=float)
#         if mu.shape[0] != k or sd.shape[0] != k:
#             raise ValueError("Scaler incompatible: il doit être fit sur df[names] dans le même ordre.")
#         sd = np.where(sd == 0.0, 1.0, sd)

#     for t in range(T):
#         # prediction
#         a_t_pred = G @ a_t + a
#         P_t_pred = G @ P_t @ G.T + Q

#         yt = y[t, :]
#         mask = ~np.isnan(yt)

#         if mask.sum() == 0:
#             a_t, P_t = a_t_pred, P_t_pred
#         else:
#             y_obs = yt[mask]
#             if scaler is not None:
#                 y_obs = (y_obs - mu[mask]) / sd[mask]

#             Z_obs = Z[mask, :]
#             R_obs = R[np.ix_(mask, mask)]

#             v = y_obs - (Z_obs @ a_t_pred)              # innovation
#             S = Z_obs @ P_t_pred @ Z_obs.T + R_obs      # cov innovation

#             # gain K
#             PZt = P_t_pred @ Z_obs.T                    # (p, m)
#             try:
#                 K = np.linalg.solve(S, PZt.T).T         # (p, m)
#             except np.linalg.LinAlgError:
#                 S = S + 1e-8 * np.eye(S.shape[0])
#                 K = np.linalg.solve(S, PZt.T).T

#             a_t = a_t_pred + K @ v

#             KH = K @ Z_obs
#             P_t = (I - KH) @ P_t_pred @ (I - KH).T + K @ R_obs @ K.T

#         a_pred[t], P_pred[t] = a_t_pred, P_t_pred
#         a_filt[t], P_filt[t] = a_t, P_t

#     return {"a_pred": a_pred, "P_pred": P_pred,
#             "a_filt": a_filt, "P_filt": P_filt,
#             "index": idx}