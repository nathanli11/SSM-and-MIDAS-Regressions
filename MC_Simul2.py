import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple

import MC_Simul1 as sim1

def simulate_two_factor_dgp(
        T=40,
        m=3,
        rho1=0.9,
        rho2=0.5,
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
        f1[t] = rho1 * f1[t - 1] + eta1[t]
        f2[t] = rho2 * f2[t - 1] + eta2[t]
    
    # Measurment errors
    uy = np.zeros(Th)
    ux = np.zeros(Th)
    for t in range(1, Th):
        uy[t] = d * uy[t - 1] + epsy[t]
        ux[t] = d * ux[t - 1] + epsx[t]
    
    # Observations
    y_star = f1 + f2 + uy
    x = f1 + ux

    # Low frequency
    y = y_star[m-1::m]

    return y, x.reshape(-1,1), f1, f2

def monte_carlo_simulation_2(
        N=500,
        T=40,
        m=3,
        rho1=0.9,
        rho2=0.5,
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
            rho1=rho1,
            rho2=rho2,
            d=d,
            seed=i
        )

        # MIDAS Forecast
        midas_forecast, midas_actual = sim1.midas_regular_forecast(y, x, h=h, m=m)
        rmspe_midas.append(sim1.rmspe(midas_forecast, midas_actual))

        # ADL-MIDAS Forecast
        adl_forecast, adl_actual = sim1.adl_midas_forecast(y, x, h=h, m=m)
        rmspe_adl.append(sim1.rmspe(adl_forecast, adl_actual))

        # Kalman Filter Forecast (MISSPECIFIED)
        kf_forecast, kf_actual = sim1.kalman_filter_forecast(y, x, h=h, m=m)
        rmspe_kf.append(sim1.rmspe(kf_forecast, kf_actual))

        return {
            "KF / MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_midas),
            "KF / ADL-MIDAS": np.mean(rmspe_kf) / np.mean(rmspe_adl),
        }





# ====================================================
# PARAMETER GRIDS (as in the paper)
# ====================================================
RHO1_GRID = [-0.9, -0.5, 0.5, 0.95]   # observed factor (in x)
RHO2_GRID = [-0.9, -0.5, 0.5, 0.95]   # unobserved factor (in y only)
D_GRID    = [-0.9, -0.5, 0.0, 0.5, 0.95]

# ====================================================
# ONE PANEL (fixed h)
# ====================================================
def run_panel_simulation_2(
    h: int,
    N: int = 500,
    T: int = 40,
    m: int = 3,
    rho1: float = 0.9
):
    """
    Runs one panel of Table 5 for fixed horizon h and rho1.
    Rows: d
    Columns: rho2
    """
    res_midas = pd.DataFrame(index=D_GRID, columns=RHO2_GRID, dtype=float)
    res_adl   = pd.DataFrame(index=D_GRID, columns=RHO2_GRID, dtype=float)

    for d in D_GRID:
        for rho2 in RHO2_GRID:
            out = monte_carlo_simulation_2(
                N=N,
                T=T,
                m=m,
                rho1=rho1,
                rho2=rho2,
                d=d,
                h=h
            )

            res_midas.loc[d, rho2] = out["KF / MIDAS"]
            res_adl.loc[d, rho2]   = out["KF / ADL-MIDAS"]

            print(f"h={h} | rho1={rho1} | d={d:>4} | rho2={rho2:>5} | done")

    return res_midas, res_adl


# ====================================================
# GENERATE TABLE 5
# ====================================================
def generate_table_5(
    N: int = 500,
    rho1: float = 0.9
):
    """
    Generates all panels of Table 5 (Simulation 2).
    """
    print("=== Table 5, Panel A: h = 1 ===")
    A_midas, A_adl = run_panel_simulation_2(
        h=1, N=N, rho1=rho1
    )

    print("=== Table 5, Panel B: h = 4 ===")
    B_midas, B_adl = run_panel_simulation_2(
        h=4, N=N, rho1=rho1
    )

    return {
        "Panel A (h=1) - Regular MIDAS": A_midas,
        "Panel A (h=1) - ADL-MIDAS": A_adl,
        "Panel B (h=4) - Regular MIDAS": B_midas,
        "Panel B (h=4) - ADL-MIDAS": B_adl,
    }

tables = generate_table_5() # Reduced N for quicker runs

tables["Panel C (h=1) - Regular MIDAS"]
tables["Panel C (h=1) - ADL-MIDAS"]
tables["Panel D (h=4) - Regular MIDAS"]
tables["Panel D (h=4) - ADL-MIDAS"]


with open("table4.tex", "w") as f:
    for name, df in tables.items():
        f.write(f"% {name}\n")
        f.write(df.to_latex(float_format="%.2f"))
        f.write("\n\n")