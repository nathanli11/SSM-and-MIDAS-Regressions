import numpy as np
import pandas as pd

from ORGA_TEST.src.ssm.forecast import (forecast_y_from_state, kalman_filter_forecast)
from ORGA_TEST.src.ssm.periodic_kf import (periodic_steady_state_kf, run_periodic_kf_filter)
from ORGA_TEST.src.midas.forecast import (regular_midas_forecast, multiplicative_midas_forecast)
from ORGA_TEST.src.dgp.simulate import (simulate_one_factor_dgp, simulate_two_factor_dgp)
from ORGA_TEST.src.evaluation.model_selection import (aic, bic, rmspe, kalman_ic_1f, kalman_ic_2f, RHO_GRID, D_GRID)


def run_panel_simulation_1(
    h: int,
    N: int = 500,
    T: int = 40,
    m: int = 3
):
    """
    Runs one panel of Table 4 (fixed horizon h).
    Returns two DataFrames:
      - KF / MIDAS
      - KF / ADL-MIDAS
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
    """
    Runs one panel of Table 5 (Simulation 3).
    Rows: d
    Columns: rho
    """
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


def monte_carlo_simulation_1(
        N=500, T=40, m=3, rho=0.9, d=0.5, h=1
):
    rmspe_midas = []
    rmspe_adl_midas = []
    rmspe_kf = []

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
            rho1=rho,
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
            # Riccati did not converge; skip this iteration
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

    for i in range(N):
        y, x, _ = simulate_one_factor_dgp(
            T=T, m=m, rho=rho, d=d, seed=i
        )

        # ======================
        # MIDAS
        # ======================
        f_m, a_m = regular_midas_forecast(y, x, h=h, m=m)
        rmspe_midas.append(rmspe(f_m, a_m))

        # ======================
        # ADL-MIDAS
        # ======================
        f_a, a_a = multiplicative_midas_forecast(y, x, h=h, m=m)
        rmspe_adl.append(rmspe(f_a, a_a))

        # ======================
        # Kalman: select #factors by IC
        # ======================
        ll1, k1, p1 = kalman_ic_1f(y, x, m)
        ll2, k2, p2 = kalman_ic_2f(y, x, m)

        if criterion == "AIC":
            ic1 = aic(ll1, k1)
            ic2 = aic(ll2, k2)
        else:
            ic1 = bic(ll1, k1, T)
            ic2 = bic(ll2, k2, T)

        # Le papier produit toujours le modèle 1 facteur
        # on suit cette convention ici
        p_hat = p1 
        kf = periodic_steady_state_kf(p_hat)
        _, states_low = run_periodic_kf_filter(kf, y, x)

        fcast = []
        actual = []
        for t in range(len(y) - h):
            fcast.append(forecast_y_from_state(p_hat, states_low[t], h))
            actual.append(y[t + h])

        rmspe_kf.append(rmspe(np.array(fcast), np.array(actual)))

    return {
        "KF / MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_midas),
        "KF / ADL-MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_adl),
    }


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
    """
    Generates all panels of Table 5 (Simulation 3).
    """

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

def generate_table_6(
    N: int = 500,
    rho: float = 0.9,
    d: float = 0.5
):
    """
    Generates Table 6 for Simulation 3.
    """
    table = {}

    for criterion in ["AIC", "BIC"]:
        for h in [1, 4]:
            print(f"Running Simulation 3 | {criterion} | h={h}")
            table[f"{criterion}, h={h}"] = run_panel_simulation_3(
                h=h,
                criterion=criterion,
                N=N,
                rho=rho,
                d=d
            )

    return pd.DataFrame(table).T



#FROM MC 4
#===========================
# Tables
#===========================


tables_4B_100 = generate_table_4B(N=100)
tables_4B_100["Panel C (h=1) - Regular MIDAS"].to_excel("Table_4_PanelC_MIDAS_100.xlsx")
tables_4B_100["Panel C (h=1) - Multiplicative MIDAS"].to_excel("Table_4_PanelC_ADL_MIDAS_100.xlsx")

tables_4B_100["Panel D (h=4) - Regular MIDAS"].to_excel("Table_4_PanelD_MIDAS_100.xlsx")
tables_4B_100["Panel D (h=4) - Multiplicative MIDAS"].to_excel("Table_4_PanelD_ADL_MIDAS_100.xlsx")

tables_4A = generate_table_4A(N=500)
tables_4A["Panel A (h=1) - Regular MIDAS"].to_excel(
    "Table_4A_PanelA_MIDAS.xlsx"
)
tables_4A["Panel A (h=1) - Multiplicative MIDAS"].to_excel(
    "Table_4A_PanelA_Multiplicative_MIDAS.xlsx"
)

tables_4A["Panel B (h=4) - Regular MIDAS"].to_excel(
    "Table_4A_PanelB_MIDAS.xlsx"
)

tables_4A["Panel B (h=4) - Multiplicative MIDAS"].to_excel(
    "Table_4A_PanelB_Multiplicative_MIDAS.xlsx"
)

tables_4B = generate_table_4B(N=500)
tables_4B["Panel C (h=1) - Regular MIDAS"].to_excel("Table_4_PanelC_MIDAS_500.xlsx")
tables_4B["Panel C (h=1) - Multiplicative MIDAS"].to_excel("Table_4_PanelC_ADL_MIDAS_500.xlsx")
tables_4B["Panel D (h=4) - Regular MIDAS"].to_excel("Table_4_PanelD_MIDAS_500.xlsx")
tables_4B["Panel D (h=4) - Multiplicative MIDAS"].to_excel("Table_4_PanelD_ADL_MIDAS_500.xlsx")


tables_5 = generate_table_5(N=500)

tables_5["Panel A (AIC, h=1) - Regular MIDAS"].to_excel(
    "Table_5_PanelA_MIDAS.xlsx"
)
tables_5["Panel A (AIC, h=1) - Multiplicative MIDAS"].to_excel(
    "Table_5_PanelA_Multiplicative_MIDAS.xlsx"
)

tables_5["Panel B (BIC, h=1) - Regular MIDAS"].to_excel(
    "Table_5_PanelB_MIDAS.xlsx"
)
tables_5["Panel B (BIC, h=1) - Multiplicative MIDAS"].to_excel(
    "Table_5_PanelB_Multiplicative_MIDAS.xlsx"
)

tables_5["Panel C (AIC, h=4) - Regular MIDAS"].to_excel(
    "Table_5_PanelC_MIDAS.xlsx"
)
tables_5["Panel C (AIC, h=4) - Multiplicative MIDAS"].to_excel(
    "Table_5_PanelC_Multiplicative_MIDAS.xlsx"
)

tables_5["Panel D (BIC, h=4) - Regular MIDAS"].to_excel(
    "Table_5_PanelD_MIDAS.xlsx"
)
tables_5["Panel D (BIC, h=4) - Multiplicative MIDAS"].to_excel(
    "Table_5_PanelD_Multiplicative_MIDAS.xlsx"
)






### UNUSED ? ###

# def kalman_forecast_series(y, x, h=1, m=3):
#     p_hat = sim1.fit_kalman_mle(y, x, m=m)
#     kf = sim1.periodic_steady_state_kf(p_hat)
#     _, states_low = sim1.run_periodic_kf_filter(kf, y, x)

#     fcasts = []
#     actuals = []
#     for t in range(len(y) - h):
#         fcasts.append(sim1.forecast_y_from_state(p_hat, states_low[t], h))
#         actuals.append(y[t+h])

#     return np.array(fcasts), np.array(actuals), p_hat

# def kalman_forecast_series(y, x, h=1, m=3):
#     p_hat = fit_kalman_mle(y, x, m=m)
#     kf = periodic_steady_state_kf(p_hat)
#     _, states_low = run_periodic_kf_filter(kf, y, x)

#     fcasts = []
#     actuals = []
#     for t in range(len(y) - h):
#         fcasts.append(forecast_y_from_state(p_hat, states_low[t], h))
#         actuals.append(y[t+h])

#     return np.array(fcasts), np.array(actuals), p_hat



### CHOIX EFFECTUES ###

#       -> regular_midas_forecast à la place de midas_regular_forecast

# def midas_regular_forecast(y, x, h=1, m=3, K=12):
#     T = len(y)
#     X = []
#     Y = []

#     for t in range(K, T-h):
#         row = [1, y[t]]
#         for j in range(K):
#             row.append(x[t*m - j - 1, 0])
#         X.append(row)
#         Y.append(y[t + h])
    
#     X = np.array(X)
#     Y = np.array(Y)

#     beta = lstsq(X, Y, rcond=None)[0]

#     forecasts = X @ beta
#     actuals = Y

#     return forecasts, actuals




# def monte_carlo_simulation_1(
#         N=500, T=40, m=3, rho=0.9, d=0.5, h=1
# ):
#     rmspe_midas = []
#     rmspe_adl_midas = []
#     rmspe_kf = []

#     for i in range(N):
#         y, x, _= simulate_one_factor_dgp(T=T, m=m, rho=rho, d=d, seed=i)
#         # MIDAS Forecast
#         midas_forecast, midas_actual = midas_regular_forecast(y, x, h=h, m=m)
#         rmspe_midas.append(rmspe(midas_forecast, midas_actual))

#         # ADL-MIDAS Forecast
#         adl_forecast, adl_actual = adl_midas_forecast(y, x, h=h, m=m)
#         rmspe_adl_midas.append(rmspe(adl_forecast, adl_actual))

#         kf_forecast, kf_actual = kalman_filter_forecast(y, x, h=h, m=m)
#         rmspe_kf.append(rmspe(kf_forecast, kf_actual))

#     return {
#         "KF / MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_midas),
#         "KF / ADL-MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_adl_midas),
#     }


# def monte_carlo_simulation_2(
#         N=500,
#         T=40,
#         m=3,
#         rho1=0.9,
#         rho2=0.5,
#         d=0.5,
#         h=1
# ):
#     rmspe_midas = []
#     rmspe_adl   = []
#     rmspe_kf    = []

#     for i in range(N):
#         y, x, _, _ = simulate_two_factor_dgp(
#             T=T,
#             m=m,
#             rho1=rho1,
#             rho2=rho2,
#             d=d,
#             seed=i
#         )

#         # MIDAS Forecast
#         midas_forecast, midas_actual = sim1.midas_regular_forecast(y, x, h=h, m=m)
#         rmspe_midas.append(sim1.rmspe(midas_forecast, midas_actual))

#         # ADL-MIDAS Forecast
#         adl_forecast, adl_actual = sim1.adl_midas_forecast(y, x, h=h, m=m)
#         rmspe_adl.append(sim1.rmspe(adl_forecast, adl_actual))

#         # Kalman Filter Forecast (MISSPECIFIED)
#         kf_forecast, kf_actual = sim1.kalman_filter_forecast(y, x, h=h, m=m)
#         rmspe_kf.append(sim1.rmspe(kf_forecast, kf_actual))

#     return {
#         "KF / MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_midas),
#         "KF / ADL-MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_adl),
#     }



# def monte_carlo_simulation_3(
#         N=500,
#         T=40,
#         m=3,
#         rho=0.9,
#         d=0.5,
#         h=1,
#         criterion="AIC" #or BIC
# ):
#     rmspe_selected = []
#     selection_count = {
#         "KF": 0,
#         "MIDAS": 0,
#         "ADL-MIDAS": 0
#     }

#     for i in range(N):
#         y, x, _ = sim1.simulate_one_factor_dgp(
#             T=T,
#             m=m,
#             rho=rho,
#             d=d,
#             seed=i
#         )

#         # MIDAS IC
#         midas_ll, midas_k, Tm = midas_ic(y, x, h=h, m=m)
#         midas_ic_value = aic(midas_ll, midas_k) if criterion == "AIC" else bic(midas_ll, midas_k, Tm)

#         # ADL-MIDAS IC
#         adl_ll, adl_k, Ta = adl_midas_ic(y, x, h=h, m=m)
#         adl_ic_value = aic(adl_ll, adl_k) if criterion == "AIC" else bic(adl_ll, adl_k, Ta)
        
#         # Kalman Filter IC
#         kf_ll, kf_k, p_hat = kalman_ic_1f(y, x, m=m)
#         kf_ic_value = aic(kf_ll, kf_k) if criterion == "AIC" else bic(kf_ll, kf_k, T)

#         # Select best model
#         ic_dict = {
#             "MIDAS": midas_ic_value,
#             "ADL-MIDAS": adl_ic_value,
#             "KF": kf_ic_value
#         }
#         best_model = min(ic_dict, key=ic_dict.get)
#         selection_count[best_model] += 1

#         # Forecast with selected model
#         if best_model == "KF":
#             kf = sim1.periodic_steady_state_kf(p_hat)
#             _, states_low = sim1.run_periodic_kf_filter(kf, y, x)
#             fcast = sim1.forecast_y_from_state(p_hat, states_low[-h-1], h)
#         elif best_model == "MIDAS":
#             f, a = sim1.midas_regular_forecast(y, x, h=h, m=m)
#             fcast = f[-1]
#         else:
#             f, a = sim1.adl_midas_forecast(y, x, h=h, m=m)
#             fcast = f[-1]
        
#         rmspe_selected.append((y[-1] - fcast)**2)
#     return {
#         "RMSPE Selected": np.sqrt(np.mean(rmspe_selected)),
#         "Selection frequencies": {
#             k: v / N for k, v in selection_count.items()
#         }
#     }



# def midas_ic(y, x, h=1, m=3, K=12):
#     fcst, act = sim1.midas_regular_forecast(y, x, h=h, m=m, K=K)
#     residuals = act - fcst
#     loglike = gaussian_loglike(residuals)
#     k = K + 2  # K coefficients + intercept + AR
#     return loglike, k, len(residuals)




#           -> multiplicative_midas_forecast à la place de adl_midas_forecast

# def adl_midas_ic(y, x, h=1, m=3, Ky=4, Kx=4):
#     fcst, act = sim1.adl_midas_forecast(y, x, h=h, m=m, Ky=Ky, Kx=Kx)
#     resid = act - fcst
#     ll = gaussian_loglike(resid)
#     k = 6   # beta_y, beta_x, 4 theta parameters
#     return ll, k, len(resid)



# def adl_midas_forecast(
#     y: np.ndarray,
#     x: np.ndarray,
#     h: int = 1,
#     m: int = 3,
#     Ky: int = 4,
#     Kx: int = 4
# ):
#     """
#     ADL-MIDAS forecast following Eq. (2.25)-(2.26).
#     Returns (forecasts, actuals).
#     """
#     T = len(y)
#     t_start = max(Ky, Kx)

#     def model(theta):
#         beta_y, beta_x = theta[0], theta[1]
#         th_y1, th_y2 = theta[2], theta[3]
#         th_x1, th_x2 = theta[4], theta[5]

#         forecasts = []
#         actuals = []

#         for t in range(t_start, T - h):
#             y_part = midas_y_term(y, t, Ky, th_y1, th_y2)
#             x_part = midas_x_term(x, t, m, Kx, th_x1, th_x2)

#             yhat = beta_y * y_part + beta_x * x_part
#             forecasts.append(yhat)
#             actuals.append(y[t + h])

#         return np.array(forecasts), np.array(actuals)

#     def objective(theta):
#         f, a = model(theta)
#         err = a-f
#         err = np.clip(err, -1e6, 1e6)  # avoid overflow
#         return np.mean(err ** 2)

#     # Initial values (important for convergence)
#     theta0 = np.array([
#         0.5,    # beta_y
#         0.5,    # beta_x
#        -0.1,    # theta_y1
#        -0.01,   # theta_y2
#        -0.1,    # theta_x1
#        -0.01    # theta_x2
#     ])

#     res = minimize(objective, theta0, method="L-BFGS-B")

#     return model(res.x)







# def midas_aggregate_x(
#     x: np.ndarray,
#     t: int,
#     m: int,
#     theta1: float,
#     theta2: float
# ) -> float:
#     """
#     MIDAS aggregation of high-frequency x at low-frequency time t.
#     x: shape (T_high, 1)
#     """
#     w = exp_almon_weights(m - 1, theta1, theta2)
#     return x[t*m-1, 0] #end of period observation


# def run_panel_simulation_2(
#     h: int,
#     N: int = 500,
#     T: int = 40,
#     m: int = 3,
#     rho1: float = 0.9
# ):
#     """
#     Runs one panel of Table 5 for fixed horizon h and rho1.
#     Rows: d
#     Columns: rho2
#     """
#     res_midas = pd.DataFrame(index=D_GRID, columns=RHO2_GRID, dtype=float)
#     res_adl   = pd.DataFrame(index=D_GRID, columns=RHO2_GRID, dtype=float)

#     for d in D_GRID:
#         for rho2 in RHO2_GRID:
#             out = monte_carlo_simulation_2(
#                 N=N,
#                 T=T,
#                 m=m,
#                 rho1=rho1,
#                 rho2=rho2,
#                 d=d,
#                 h=h
#             )

#             res_midas.loc[d, rho2] = out["KF / MIDAS"]
#             res_adl.loc[d, rho2]   = out["KF / ADL-MIDAS"]

#             print(f"h={h} | rho1={rho1} | d={d:>4} | rho2={rho2:>5} | done")

#     return res_midas, res_adl


# def run_panel_simulation_3(
#     h: int,
#     criterion: str,
#     N: int = 500,
#     T: int = 40,
#     m: int = 3,
#     rho: float = 0.9,
#     d: float = 0.5
# ):
#     """
#     Runs one panel of Table 6.
#     Returns a Series with RMSPE and selection frequencies.
#     """
#     out = monte_carlo_simulation_3(
#         N=N,
#         T=T,
#         m=m,
#         rho=rho,
#         d=d,
#         h=h,
#         criterion=criterion
#     )

#     res = {
#         "RMSPE": out["RMSPE"],
#         "KF": out["Selection frequencies"]["KF"],
#         "MIDAS": out["Selection frequencies"]["MIDAS"],
#         "ADL-MIDAS": out["Selection frequencies"]["ADL-MIDAS"],
#     }

#     return pd.Series(res)


# def generate_table_4(N=500):
#     print("=== Panel A: One-Factor DGP, h = 1 ===")
#     A_midas, A_adl = run_panel(h=1, N=N)

#     print("=== Panel B: One-Factor DGP, h = 4 ===")
#     B_midas, B_adl = run_panel(h=4, N=N)

#     return {
#         "Panel A (h=1) - Regular MIDAS": A_midas,
#         "Panel A (h=1) - ADL-MIDAS": A_adl,
#         "Panel B (h=4) - Regular MIDAS": B_midas,
#         "Panel B (h=4) - ADL-MIDAS": B_adl,
#     }

# def generate_table_4B(
#     N: int = 500,
#     rho1: float = 0.9
# ):
#     """
#     Generates all panels of Table 4B (Simulation 2).
#     """
#     print("=== Table 4B, Panel C: h = 1 ===")
#     A_midas, A_adl = run_panel_simulation_2(
#         h=1, N=N, rho1=rho1
#     )

#     print("=== Table 4B, Panel D: h = 4 ===")
#     B_midas, B_adl = run_panel_simulation_2(
#         h=4, N=N, rho1=rho1
#     )

#     return {
#         "Panel C (h=1) - Regular MIDAS": A_midas,
#         "Panel C (h=1) - ADL-MIDAS": A_adl,
#         "Panel D (h=4) - Regular MIDAS": B_midas,
#         "Panel D (h=4) - ADL-MIDAS": B_adl,
#     }


# def midas_aggregate_x(
#     x: np.ndarray,
#     t: int,
#     m: int,
#     theta1: float,
#     theta2: float
# ) -> float:
#     """
#     MIDAS aggregation of high-frequency x at low-frequency time t.
#     x: shape (T_high, 1)
#     """
#     w = exp_almon_weights(m - 1, theta1, theta2)
#     val = 0.0
#     for k in range(m):
#         idx = t*m-1-k
#         if idx<0:
#             continue
#         val += w[k] * x[idx, 0]

#     return val

# def midas_x_term(
#     x: np.ndarray,
#     t: int,
#     m: int,
#     Kx: int,
#     theta_x1: float,
#     theta_x2: float
# ) -> float:
#     w_x = exp_almon_weights(Kx, theta_x1, theta_x2)

#     return sum(
#         w_x[j] * midas_aggregate_x(x, t - j, m, theta_x1, theta_x2)
#         for j in range(Kx + 1)
#     )

# def midas_y_term(
#     y: np.ndarray,
#     t: int,
#     Ky: int,
#     theta_y1: float,
#     theta_y2: float
# ) -> float:
#     w_y = exp_almon_weights(Ky, theta_y1, theta_y2)
#     return sum(w_y[j] * y[t - j] for j in range(Ky + 1))


# FROM MC2

# tables = generate_table_4B() # Reduced N for quicker runs

# tables["Panel C (h=1) - Regular MIDAS"]
# tables["Panel C (h=1) - ADL-MIDAS"]
# tables["Panel D (h=4) - Regular MIDAS"]
# tables["Panel D (h=4) - ADL-MIDAS"]


# with open("table4.tex", "w") as f:
#     for name, df in tables.items():
#         f.write(f"% {name}\n")
#         f.write(df.to_latex(float_format="%.2f"))
#         f.write("\n\n")


# FROM MC3

# ====================================================
# GENERATE TABLE 6
# ====================================================

# DEBUG MODE
# table6 = generate_table_6(
#     N=10,
#     rho=0.9,
#     d=0.5
# )

# print(table6)
# NORMAL MODE
#table6 = generate_table_6(
#    N=500,
#    rho=0.9,
#    d=0.5
#)  