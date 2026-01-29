import pandas as pd
import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm

# Data importation
stat_xl = pd.read_excel('stationnary_data.xlsx', index_col=0, parse_dates=True)
y_parse = pd.read_excel('y_sparse.xlsx', index_col=0, parse_dates=True)

series_names = stat_xl.columns.tolist()
gdp_name = y_parse.columns[0]

gdp_diff_df = y_parse.dropna().diff()
gdp_diff_df.columns = [gdp_name]

final_data = (
    stat_xl
    .merge(gdp_diff_df, left_index=True, right_index=True, how="left")
)

# Kalman Filter

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

def kalman_filter_minimal(df_x, G, Q, a, Z, R, names, x0=None, P0=None, scaler=None):
    """
    Filtre de Kalman minimal (prévision et filtrage) pour données avec NaN.
    """
    y = df_x.reindex(columns=names).values.astype(float)
    idx = df_x.index
    T, k = y.shape
    p = G.shape[0]
    I = np.eye(p)

    if x0 is None:
        x0 = np.zeros(p, dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float).reshape(p)

    if P0 is None:
        P0 = np.eye(p, dtype=float) * 1e2
    else:
        P0 = np.asarray(P0, dtype=float).reshape(p, p)

    a = np.asarray(a, dtype=float).reshape(p)

    a_pred = np.zeros((T, p))
    P_pred = np.zeros((T, p, p))
    a_filt = np.zeros((T, p))
    P_filt = np.zeros((T, p, p))

    a_t = x0.copy()
    P_t = P0.copy()

    if scaler is not None:
        mu = np.asarray(scaler.mean_, dtype=float)
        sd = np.asarray(scaler.scale_, dtype=float)
        if mu.shape[0] != k or sd.shape[0] != k:
            raise ValueError("Scaler incompatible: il doit être fit sur df[names] dans le même ordre.")
        sd = np.where(sd == 0.0, 1.0, sd)

    for t in range(T):
        # prediction
        a_t_pred = G @ a_t + a
        P_t_pred = G @ P_t @ G.T + Q

        yt = y[t, :]
        mask = ~np.isnan(yt)

        if mask.sum() == 0:
            a_t, P_t = a_t_pred, P_t_pred
        else:
            y_obs = yt[mask]
            if scaler is not None:
                y_obs = (y_obs - mu[mask]) / sd[mask]

            Z_obs = Z[mask, :]
            R_obs = R[np.ix_(mask, mask)]

            v = y_obs - (Z_obs @ a_t_pred)              # innovation
            S = Z_obs @ P_t_pred @ Z_obs.T + R_obs      # cov innovation

            # gain K
            PZt = P_t_pred @ Z_obs.T                    # (p, m)
            try:
                K = np.linalg.solve(S, PZt.T).T         # (p, m)
            except np.linalg.LinAlgError:
                S = S + 1e-8 * np.eye(S.shape[0])
                K = np.linalg.solve(S, PZt.T).T

            a_t = a_t_pred + K @ v

            KH = K @ Z_obs
            P_t = (I - KH) @ P_t_pred @ (I - KH).T + K @ R_obs @ K.T

        a_pred[t], P_pred[t] = a_t_pred, P_t_pred
        a_filt[t], P_filt[t] = a_t, P_t

    return {"a_pred": a_pred, "P_pred": P_pred,
            "a_filt": a_filt, "P_filt": P_filt,
            "index": idx}

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

# Fonctions utilitaires pour les périodes trimestrielles

def quarter_end_months(q_period: pd.Period) -> pd.Timestamp:
    """Dernier mois (fin de trimestre) en fin de mois."""
    return q_period.asfreq("Q").end_time.to_period("M").to_timestamp("M")

def quarter_second_month(q_period: pd.Period) -> pd.Timestamp:
    """2e mois du trimestre en fin de mois: Q1->Feb, Q2->May, Q3->Aug, Q4->Nov."""
    start = q_period.start_time.to_period("M").to_timestamp("M")
    return (start.to_period("M") + 1).to_timestamp("M")  # +1 mois

def to_quarter_period_index(idx):
    """Index -> PeriodIndex trimestriel."""
    return pd.PeriodIndex(idx, freq="Q")

def normalize_full_sample(df: pd.DataFrame):
    """Normalise une série (DataFrame 1 colonne) par sa moyenne/écart-type full sample."""
    mu = df[df.columns[0]].mean()
    sd = df[df.columns[0]].std(ddof=0)
    sd = 1.0 if sd == 0 else sd
    return (df[df.columns[0]] - mu) / sd, mu, sd

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

def forecast_gdp_quarter_ssm(kf_out, params, origin_month, target_q):
    """
    Prévision du GDP trimestriel target_q à partir du filtre Kalman
    """
    idx = pd.DatetimeIndex(kf_out["index"])
    if origin_month not in idx:
        return np.nan

    t0 = int(np.where(idx == origin_month)[0][0])
    a0 = kf_out["a_filt"][t0].copy()
    G = kf_out["G"]
    g1 = params["gamma1"]

    # dernier mois du trimestre target_q
    m3 = quarter_end_months(target_q)
    if m3 not in idx:
        step = (m3.to_period("M") - origin_month.to_period("M")).n
    else:
        step = int(np.where(idx == m3)[0][0]) - t0

    if step <= 0:
        return np.nan

    # projection d'état jusqu'à m3
    a = a0
    for _ in range(step):
        a = G @ a

    f_m3, u1_m3 = a[0], a[1]
    gdp_hat = g1 * f_m3 + u1_m3
    return float(gdp_hat)

# Previsions récursives SSM avec Kalman

def recursive_forecast_exercise(gdp_q_col : pd.DataFrame,
                                x_m_col : pd.DataFrame,
                                start_est_q: str,
                                end_est_q: str,
                                names,
                                eval_start_q: str,
                                eval_end_q: str,
                                horizons=range(1,9),
                                gdp_lags=1,
                                x_month_lags=6):
    """
    - Normalise (full sample) GDP et x.
    - Estime récursivement:
       * SSM (ML via Kalman)
    - Prévisions faites avec données mensuelles jusqu'au 2e mois du trimestre à prévoir.
    """

    # --- normalisation full sample (comme le papier) ---
    #gdp_q, gdp_mu, gdp_sd = normalize_full_sample(gdp_q_col.astype(float))
    #x_m, x_mu, x_sd = normalize_full_sample(x_m_col.astype(float))
    gdp_norm, gdp_mu, gdp_sd = normalize_full_sample(gdp_q_col.astype(float))
    x_norm, x_mu, x_sd = normalize_full_sample(x_m_col.astype(float))

    # --- indices trimestriels ---
    gdp_q = gdp_norm.dropna()
    gdp_q.index = pd.PeriodIndex(gdp_q.index, freq="Q")
    gdp_q = gdp_q.groupby(level=0).last().sort_index()


    start_est = pd.Period(start_est_q, freq="Q")
    end_est   = pd.Period(end_est_q,   freq="Q")
    eval_start= pd.Period(eval_start_q,freq="Q")
    eval_end  = pd.Period(eval_end_q,  freq="Q")

    # trimestres évalués
    qs_all = gdp_q.index
    eval_qs = [q for q in qs_all if (q >= eval_start and q <= eval_end)]

    # stockage
    out = []
    # pré-calendrier mensuel (index mensuel fin de mois)
    x_m = x_norm.astype(float).copy()
    x_m.index = pd.DatetimeIndex(x_m.index).to_period("M").to_timestamp("M")
    x_m = x_m.sort_index()

    for origin_q in eval_qs:
        print("ORIGIN:", origin_q)

        # fenêtre d'estimation expanding: de start_est à origin_q-1
        est_end_q = origin_q - 1
        if est_end_q < start_est:
            continue
        gdp_est = gdp_q.loc[start_est:est_end_q].dropna()

        # construire x disponible pour l'estimation:
        # on suppose qu'à chaque trimestre, on n'utilise x que jusqu'au 2e mois
        # => pour l'estimation, il suffit d'avoir x sur tout l'historique mensuel.
        x_est = x_m.copy()


        # --------- (B) SSM estimation (ML) ----------
        # construire df_hf mensuel avec gdp observé aux fins de trimestre de la fenêtre d'estimation
        # et NaN ailleurs. Le GDP du trimestre est placé au dernier mois du trimestre.
        # IMPORTANT: on n'inclut pas d'observation GDP pour le trimestre origin_q (à prévoir).
        start_m = quarter_end_months(start_est).to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(2)  # approx
        end_m   = quarter_second_month(origin_q)  # info set: jusqu'au 2e mois du trimestre origin_q (comme le papier)
        months = pd.date_range(start=start_m, end=end_m, freq="M")

        df_hf = pd.DataFrame(index=months, columns=names, dtype=float)
        df_hf[names[1]] = x_m.reindex(months).values

        # GDP observé seulement pour trimestres <= est_end_q, au dernier mois du trimestre
        for q in gdp_est.index:
            m3 = quarter_end_months(q)
            if m3 in df_hf.index:
                df_hf.loc[m3, names[0]] = gdp_est.loc[q]

        # (il y aura des NaN si x manque : on laisse, le Kalman gère)

        # Estimation ML
        params = fit_ssm_ml(df_hf, names)

        # Filtre jusqu'à end_m (2e mois du trimestre origin_q)
        kf_out = kalman_filter_states(df_hf, params, names)

        # --------- Forecasts h = 1..8 ----------
        for h in horizons:
            #target_q = origin_q + (h - 1)  # h=1 => trimestre origin_q
            target_q = origin_q + h #h=0 => trimestre origin_q

            # SSM: on prédit GDP du trimestre target_q à info jusqu'au 2e mois de origin_q
            # (pour h>1, origin_q est plus tôt, mais dans le papier ils font 2..8 quarters ahead pareil
            #  => tu peux remplacer origin_month par 2e mois de origin_q pour "as-of" constant)
            origin_month = quarter_second_month(origin_q)
            yhat_ssm = forecast_gdp_quarter_ssm(
                kf_out, params, origin_month=origin_month, target_q=target_q
            )

            print(origin_q, target_q)
            
            out.append({
                "origin_q": str(origin_q),
                "target_q": str(target_q),
                "h": h,
                "ssm_hat": yhat_ssm,
                "gdp_real": gdp_q.loc[target_q] if target_q in gdp_q.index else np.nan
            })

    return pd.DataFrame(out)

# Boucle à run pour sortir les resultats de forecast et RMSE

for i in range(len(stat_xl.columns)):
  names = [gdp_name, series_names[i]]
  params = fit_ssm_ml(final_data, names)

  G, Q, a, Z, R = build_ssm_two_series_ar1_ml(params)

  x = series_names[i]

  kf = kalman_filter_minimal(
      df_x=final_data,
      G=G, Q=Q, a=a, Z=Z, R=R,
      names=[gdp_name,x],
      scaler=None  
  )

  f_filt = kf["a_filt"][:, 0]
  f_pred = kf["a_pred"][:, 0]

  first_date = [0,1,2,4,5,8]
  second_date = [3] #EXPTN
  third_date = [6]
  fourth_date = [7] #oil

  if not (i in first_date or i in third_date or i in fourth_date):
    continue
  
  elif i in first_date:
    df_forecasts = recursive_forecast_exercise(
      gdp_q_col=final_data[[gdp_name]],
      x_m_col=final_data[[series_names[i]]],
      start_est_q="1958Q1",
      end_est_q="1990Q4",
      names = [gdp_name, series_names[i]],
      eval_start_q="1991Q1",
      eval_end_q="2023Q4",
      horizons=range(1,9),
      gdp_lags=1,
      x_month_lags=6
    )
  elif i in second_date:
    df_forecasts = recursive_forecast_exercise(
      gdp_q_col=final_data[[gdp_name]],
      x_m_col=final_data[[series_names[i]]],
      start_est_q="1978Q1",
      end_est_q="1990Q4",
      names = [gdp_name, series_names[i]],
      eval_start_q="1991Q1",
      eval_end_q="2023Q4",
      horizons=range(1,9),
      gdp_lags=1,
      x_month_lags=6
    )
  elif i in third_date:
    df_forecasts = recursive_forecast_exercise(
      gdp_q_col=final_data[[gdp_name]],
      x_m_col=final_data[[series_names[i]]],
      start_est_q="1967Q1",
      end_est_q="1990Q4",
      names = [gdp_name, series_names[i]],
      eval_start_q="1991Q1",
      eval_end_q="2023Q4",
      horizons=range(1,9),
      gdp_lags=1,
      x_month_lags=6
    )
  elif i in fourth_date:
    df_forecasts = recursive_forecast_exercise(
      gdp_q_col=final_data[[gdp_name]],
      x_m_col=final_data[[series_names[i]]],
      start_est_q="1982Q1",
      end_est_q="1992Q4",
      names = [gdp_name, series_names[i]],
      eval_start_q="1993Q1",
      eval_end_q="2023Q4",
      horizons=range(1,9),
      gdp_lags=1,
      x_month_lags=6
    )

  # 1) Garder uniquement les observations valides pour le SSM
  df_eval = df_forecasts.dropna(subset=["ssm_hat", "gdp_real"]).copy()
  df_eval["sq_err"] = (df_eval["ssm_hat"] - df_eval["gdp_real"]) ** 2

  # 2) RMSE par horizon selon la formule (1/N_h * somme)
  rmse_ssm_by_h = (
      df_eval.groupby("h")
      .agg(
          N_h=("sq_err", "size"),
          RMSE_SSM=("sq_err", lambda s: np.sqrt(s.sum() / len(s)))
      )
      .reset_index()
      .sort_values("h")
  )
  df_forecasts.to_excel(f"forecast_{i}.xlsx")
  rmse_ssm_by_h.to_excel(f"rmse_{i}.xlsx")