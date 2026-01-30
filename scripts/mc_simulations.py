import numpy as np
import pandas as pd

from src.ssm.forecast import (
    forecast_y_from_state, kalman_filter_forecast, forecast_y_from_state_2f
)
from src.ssm.periodic_kf import (
    periodic_steady_state_kf, run_periodic_kf_filter, periodic_steady_state_kf_2f, run_periodic_kf_filter_2f
)
from src.midas.forecast import (
    regular_midas_forecast, multiplicative_midas_forecast
)
from src.dgp.simulate import (
    simulate_one_factor_dgp, simulate_two_factor_dgp
)
from src.evaluation.model_selection import (
    aic, bic, rmspe, kalman_ic_1f, kalman_ic_2f, RHO_GRID, D_GRID
    )

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


def main():
    repo = "scripts/results/table_4_5/"
    tables_4A = generate_table_4A(N=1)
    tables_4A["Panel A (h=1) - Regular MIDAS"].to_excel(repo + "Table_4A_PanelA_MIDAS.xlsx")
    tables_4A["Panel A (h=1) - Multiplicative MIDAS"].to_excel(repo + "Table_4A_PanelA_Multiplicative_MIDAS.xlsx")
    tables_4A["Panel B (h=4) - Regular MIDAS"].to_excel(repo + "Table_4A_PanelB_MIDAS.xlsx")
    tables_4A["Panel B (h=4) - Multiplicative MIDAS"].to_excel(repo + "Table_4A_PanelB_Multiplicative_MIDAS.xlsx")


    tables_4B = generate_table_4B(N=1)
    tables_4B["Panel C (h=1) - Regular MIDAS"].to_excel(repo + "Table_4B_PanelC_MIDAS.xlsx")
    tables_4B["Panel C (h=1) - Multiplicative MIDAS"].to_excel(repo + "Table_4B_PanelC_Multiplicative_MIDAS.xlsx")
    tables_4B["Panel D (h=4) - Regular MIDAS"].to_excel(repo + "Table_4B_PanelD_MIDAS.xlsx")
    tables_4B["Panel D (h=4) - Multiplicative MIDAS"].to_excel(repo + "Table_4B_PanelD_Multiplicative_MIDAS.xlsx")


    tables_5 = generate_table_5(N=1)
    tables_5["Panel A (AIC, h=1) - Regular MIDAS"].to_excel(repo + "Table_5_PanelA_MIDAS.xlsx")
    tables_5["Panel A (AIC, h=1) - Multiplicative MIDAS"].to_excel(repo + "Table_5_PanelA_Multiplicative_MIDAS.xlsx")
    tables_5["Panel B (BIC, h=1) - Regular MIDAS"].to_excel(repo + "Table_5_PanelB_MIDAS.xlsx")
    tables_5["Panel B (BIC, h=1) - Multiplicative MIDAS"].to_excel(repo + "Table_5_PanelB_Multiplicative_MIDAS.xlsx")
    tables_5["Panel C (AIC, h=4) - Regular MIDAS"].to_excel(repo + "Table_5_PanelC_MIDAS.xlsx")
    tables_5["Panel C (AIC, h=4) - Multiplicative MIDAS"].to_excel(repo + "Table_5_PanelC_Multiplicative_MIDAS.xlsx")
    tables_5["Panel D (BIC, h=4) - Regular MIDAS"].to_excel(repo + "Table_5_PanelD_MIDAS.xlsx")
    tables_5["Panel D (BIC, h=4) - Multiplicative MIDAS"].to_excel(repo + "Table_5_PanelD_Multiplicative_MIDAS.xlsx")

if __name__ == "__main__":
    main()
