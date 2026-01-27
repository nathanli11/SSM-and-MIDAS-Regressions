'''
Le but est de mettre en concurrence les 3 modèles.
On calcule AIC / BIC.
On choisit le modèle avec le critère minimal.
On compare les performances de prévision hors échantillon.
'''

import table1_2_3 as tb
import numpy as np
import pandas as pd
import MC_Simul1 as sim1

def gaussian_loglike(residuals: np.ndarray) -> float:
    T = len(residuals)
    sigma2 = np.var(residuals)
    loglike = -0.5 * T * (np.log(2 * np.pi * sigma2) + 1)
    return loglike

def aic(loglike: float, k: int) -> float:
    return -2 * loglike + 2 * k

def bic(loglike: float, k: int, T: int) -> float:
    return -2 * loglike + k * np.log(T)

def midas_ic(y, x, h=1, m=3, K=12):
    fcst, act = sim1.midas_regular_forecast(y, x, h=h, m=m, K=K)
    residuals = act - fcst
    loglike = gaussian_loglike(residuals)
    k = K + 2  # K coefficients + intercept + AR
    return loglike, k

def adl_midas_ic(y, x, h=1, m=3, Ky=4, Kx=4):
    fcst, act = sim1.adl_midas_forecast(y, x, h=h, m=m, Ky=Ky, Kx=Kx)
    resid = act - fcst
    ll = gaussian_loglike(resid)
    k = 6   # beta_y, beta_x, 4 theta parameters
    return ll, k

def kalman_ic(y, x, m=3):
    p_hat = sim1.fit_kalman_mle(y, x, m=m)
    ll = sim1.kalman_loglike_full(p_hat, y, x)
    k = 2 + 3   # rho, d + 3 variances
    return ll, k, p_hat


'''Simulation de Monte Carlo :
    A chaque répication:
    - on estime les 3 modèles
    - on calcule AIC/BIC
    - on choisit le meilleur
    - on prévoit avec celui-là
'''

def monte_carlo_simulation_3(
        N=500,
        T=40,
        m=3,
        rho=0.9,
        d=0.5,
        h=1,
        criterion="AIC" #or BIC
):
    rmspe_selected = []
    selection_count = {
        "KF": 0,
        "MIDAS": 0,
        "ADL-MIDAS": 0
    }

    for i in range(N):
        y, x, _, _ = sim1.simulate_one_factor_dgp(
            T=T,
            m=m,
            rho=rho,
            d=d,
            seed=i
        )

        # MIDAS IC
        midas_ll, midas_k = midas_ic(y, x, h=h, m=m)
        midas_ic_value = aic(midas_ll, midas_k) if criterion == "AIC" else bic(midas_ll, midas_k, T)

        # ADL-MIDAS IC
        adl_ll, adl_k = adl_midas_ic(y, x, h=h, m=m)
        adl_ic_value = aic(adl_ll, adl_k) if criterion == "AIC" else bic(adl_ll, adl_k, T)
        
        # Kalman Filter IC
        kf_ll, kf_k, p_hat = kalman_ic(y, x, m=m)
        kf_ic_value = aic(kf_ll, kf_k) if criterion == "AIC" else bic(kf_ll, kf_k, T)

        # Select best model
        ic_dict = {
            "MIDAS": midas_ic_value,
            "ADL-MIDAS": adl_ic_value,
            "KF": kf_ic_value
        }
        best_model = min(ic_dict, key=ic_dict.get)
        selection_count[best_model] += 1

        # Forecast with selected model
        if best_model == "KF":
            kf = sim1.periodic_steady_state_kf(p_hat)
            _, states_low = sim1.run_periodic_kf_filter(kf, y, x)
            fcast = sim1.forecast_y_from_state(p_hat, states_low[-h-1], h)
        elif best_model == "MIDAS":
            f, a = sim1.midas_regular_forecast(y, x, h=h, m=m)
            fcast = f[-1]
        else:
            f, a = sim1.adl_midas_forecast(y, x, h=h, m=m)
            fcast = f[-1]
        
        rmspe_selected.append((a[-1] - fcast)**2)
    return {
        "RMSPE Selected": np.sqrt(np.mean(rmspe_selected)),
        "Selection frequencies": {
            k: v / N for k, v in selection_count.items()
        }
    }


# ====================================================
# ONE PANEL (fixed h, fixed criterion)
# ====================================================
def run_panel_simulation_3(
    h: int,
    criterion: str,
    N: int = 500,
    T: int = 40,
    m: int = 3,
    rho: float = 0.9,
    d: float = 0.5
):
    """
    Runs one panel of Table 6.
    Returns a Series with RMSPE and selection frequencies.
    """
    out = monte_carlo_simulation_3(
        N=N,
        T=T,
        m=m,
        rho=rho,
        d=d,
        h=h,
        criterion=criterion
    )

    res = {
        "RMSPE": out["RMSPE"],
        "KF": out["Selection frequencies"]["KF"],
        "MIDAS": out["Selection frequencies"]["MIDAS"],
        "ADL-MIDAS": out["Selection frequencies"]["ADL-MIDAS"],
    }

    return pd.Series(res)


# ====================================================
# GENERATE TABLE 6
# ====================================================
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

# DEBUG MODE
table6 = generate_table_6(
    N=10,
    rho=0.9,
    d=0.5
)

print(table6)
# NORMAL MODE
#table6 = generate_table_6(
#    N=500,
#    rho=0.9,
#    d=0.5
#)  