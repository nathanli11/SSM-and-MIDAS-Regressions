
import pandas as pd
import numpy as np
from typing import Dict
from scipy.optimize import minimize

from src.ssm.params import OneFactorParams
from src.evaluation.model_selection import K_BAR, OPT_TOL, D_GRID, RHO_GRID
from src.ssm.kalman_weights import kalman_weights_by_impulses, kalman_weights_by_impulses_multi_x
from src.midas.midas_fit import fit_regular_midas_to_kf_by_l2, fit_multiplicative_midas_to_kf_by_l2
from src.midas.midas_base import exp_almon_weights
from src.evaluation.table3_population import (
    cov_one_or_two_factor_yx, sigma_matrix_for_upsilon, fit_regular_midas_by_pe_variance, ss1_best_pe_variance_under_true_sigma
)

RICCATI_TOL = 1e-9
KF_WARMUP_PERIODS = 100 

# -----------------------------
# construction des tableaux
# -----------------------------
def table1() -> Dict[str, pd.DataFrame]:

    out: Dict[str, pd.DataFrame] = {}

    for m in [3, 13]:
        for h in [1, 4]:
            reg_vals = np.zeros((len(D_GRID), len(RHO_GRID)))
            mul_vals = np.zeros((len(D_GRID), len(RHO_GRID)))

            for i_d, d in enumerate(D_GRID):
                for i_r, rho in enumerate(RHO_GRID):
                    # modèle à un facteur
                    p = OneFactorParams(
                        m=m, rho=rho, d=d,
                        lam_y=1.0, lam_x=np.array([1.0]),
                        sig2_f=1.0, sig2_uy=1.0, sig2_ux=np.array([1.0])
                    )

                    wy_kf, wx_kf = kalman_weights_by_impulses(p, h=h, K_bar=K_BAR)

                    reg_vals[i_d, i_r] = fit_regular_midas_to_kf_by_l2(wy_kf, wx_kf, K_bar=K_BAR, m=m)
                    mul_vals[i_d, i_r] = fit_multiplicative_midas_to_kf_by_l2(wy_kf, wx_kf, K_bar=K_BAR, m=m)

            df_reg = pd.DataFrame(reg_vals, index=D_GRID, columns=RHO_GRID)
            df_mul = pd.DataFrame(mul_vals, index=D_GRID, columns=RHO_GRID)
            out[f"Table1_m={m}_h={h}_regular"] = df_reg
            out[f"Table1_m={m}_h={h}_multiplicative"] = df_mul

    return out


def table2() -> Dict[str, pd.DataFrame]:

    out: Dict[str, pd.DataFrame] = {}
    h = 1

    for m in [3, 13]:
        for unequal in [False, True]:
            reg_vals = np.zeros((len(D_GRID), len(RHO_GRID)))
            mul_vals = np.zeros((len(D_GRID), len(RHO_GRID)))

            for i_d, d in enumerate(D_GRID):
                for i_r, rho in enumerate(RHO_GRID):
                    sig2_ux = np.array([1.0, 1.0]) if not unequal else np.array([1.0, 10.0])  # variance égale ou inégale

                    p = OneFactorParams(
                        m=m, rho=rho, d=d,
                        lam_y=1.0, lam_x=np.array([1.0, 1.0]),
                        sig2_f=1.0, sig2_uy=1.0, sig2_ux=sig2_ux
                    )

                    wy_kf, wx_kf_all = kalman_weights_by_impulses_multi_x(p, h=h, K_bar=K_BAR)
                    # Pour MIDAS à deux x, nous devons ajuster les deux vecteurs de poids x séparément.
                    # On minimise la somme des distances L2 pour les deux x.

                    def fit_regular_two_x() -> float:
                        def obj(u: np.ndarray) -> float:
                            th_y = (u[0], u[1])
                            th_x1 = (u[2], u[3])
                            th_x2 = (u[4], u[5])

                            sy = exp_almon_weights(K_BAR, th_y[0], th_y[1])
                            sx1 = exp_almon_weights(m * K_BAR, th_x1[0], th_x1[1])
                            sx2 = exp_almon_weights(m * K_BAR, th_x2[0], th_x2[1])

                            by = float(np.dot(wy_kf, sy) / max(1e-15, np.dot(sy, sy)))
                            b1 = float(np.dot(wx_kf_all[0], sx1) / max(1e-15, np.dot(sx1, sx1)))
                            b2 = float(np.dot(wx_kf_all[1], sx2) / max(1e-15, np.dot(sx2, sx2)))

                            wy_m = by * sy
                            wx1_m = b1 * sx1
                            wx2_m = b2 * sx2

                            return float(np.sum((wy_kf - wy_m) ** 2) +
                                         np.sum((wx_kf_all[0] - wx1_m) ** 2) +
                                         np.sum((wx_kf_all[1] - wx2_m) ** 2))

                        x0 = np.array([-0.2, 0.0, -0.2, 0.0, -0.2, 0.0])
                        res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": OPT_TOL, "fatol": OPT_TOL, "maxiter": 80_000})
                        return float(res.fun)

                    def fit_multiplicative_two_x() -> float:
                        def obj(u: np.ndarray) -> float:
                            th_y = (u[0], u[1])

                            th_o1 = (u[2], u[3])
                            th_i1 = (u[4], u[5])

                            th_o2 = (u[6], u[7])
                            th_i2 = (u[8], u[9])

                            sy = exp_almon_weights(K_BAR, th_y[0], th_y[1])

                            # construction des formes sx pour les deux x
                            def sx_shape(th_o, th_i):
                                w_outer = exp_almon_weights(K_BAR, th_o[0], th_o[1])
                                w_inner = exp_almon_weights(m - 1, th_i[0], th_i[1])
                                sx = np.zeros(m * K_BAR + 1)
                                for jj in range(K_BAR + 1):
                                    for r in range(m):
                                        k = jj * m + r
                                        if k <= m * K_BAR:
                                            sx[k] += w_outer[jj] * w_inner[r]
                                return sx

                            sx1 = sx_shape(th_o1, th_i1)
                            sx2 = sx_shape(th_o2, th_i2)

                            by = float(np.dot(wy_kf, sy) / max(1e-15, np.dot(sy, sy)))
                            b1 = float(np.dot(wx_kf_all[0], sx1) / max(1e-15, np.dot(sx1, sx1)))
                            b2 = float(np.dot(wx_kf_all[1], sx2) / max(1e-15, np.dot(sx2, sx2)))

                            wy_m = by * sy
                            wx1_m = b1 * sx1
                            wx2_m = b2 * sx2

                            return float(np.sum((wy_kf - wy_m) ** 2) +
                                         np.sum((wx_kf_all[0] - wx1_m) ** 2) +
                                         np.sum((wx_kf_all[1] - wx2_m) ** 2))

                        x0 = np.array([-0.2, 0.0,  -0.2, 0.0, -0.2, 0.0,  -0.2, 0.0, -0.2, 0.0])
                        res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": OPT_TOL, "fatol": OPT_TOL, "maxiter": 120_000})
                        return float(res.fun)

                    reg_vals[i_d, i_r] = fit_regular_two_x()
                    mul_vals[i_d, i_r] = fit_multiplicative_two_x()

            tag = "unequal" if unequal else "equal"
            out[f"Table2_m={m}_{tag}_regular"] = pd.DataFrame(reg_vals, index=D_GRID, columns=RHO_GRID)
            out[f"Table2_m={m}_{tag}_multiplicative"] = pd.DataFrame(mul_vals, index=D_GRID, columns=RHO_GRID)

    return out


def table3() -> Dict[str, pd.DataFrame]:

    out: Dict[str, pd.DataFrame] = {}

    # 2 facteurs latents
    a = np.array([0.9, 0.1])
    b = np.array([[0.1, 0.9]])  # un seul x

    for m in [3, 13]:
        print(f"m={m}")
        for h in [1, 4]:
            print(f"  h={h}")
            ratios = np.zeros((len(D_GRID), len(RHO_GRID)))
            for i_d, d in enumerate(D_GRID):
                for i_r, rho in enumerate(RHO_GRID):
                    # Construction de la matrice de covariance vraie Sigma sous le DGP à deux facteurs
                    # chaque facteur est AR(1) avec paramètre rho
                    var_f = 1.0 / (1.0 - rho * rho) if abs(rho) < 1 else 1e12

                    Vyy = var_f * float(a[0] ** 2 + a[1] ** 2)
                    Vxx = var_f * float(b[0, 0] ** 2 + b[0, 1] ** 2)
                    Vxy = var_f * float(a[0] * b[0, 0] + a[1] * b[0, 1])
                    V_factor = np.array([[Vyy, Vxy], [Vxy, Vxx]], dtype=float)

                    cov_yy, cov_xx, cov_xy = cov_one_or_two_factor_yx(
                        m=m, rho=rho, d=d,
                        V_factor=V_factor,
                        sig2_uy=1.0,
                        sig2_ux=1.0,
                    )
                    Sigma_true = sigma_matrix_for_upsilon(m=m, h=h, K_bar=K_BAR, cov_yy=cov_yy, cov_xx=cov_xx, cov_xy=cov_xy)

                    # SS1 meilleur PE variance sous vraie Sigma
                    pe_ss1 = ss1_best_pe_variance_under_true_sigma(
                        Sigma_true=Sigma_true, m=m, h=h, K_bar=K_BAR, d_fixed=d, rho_init=rho
                    )

                    # MIDAS régulier meilleur PE variance sous vraie Sigma
                    pe_midas, _, _ = fit_regular_midas_by_pe_variance(
                        Sigma=Sigma_true, m=m, h=h, K_bar=K_BAR
                    )

                    ratios[i_d, i_r] = pe_midas / pe_ss1 if pe_ss1 > 0 else np.nan

            out[f"Table3_m={m}_h={h}"] = pd.DataFrame(ratios, index=D_GRID, columns=RHO_GRID)

    return out



# -----------------------------
# Appel principal
# -----------------------------
def main():
    pd.set_option("display.float_format", lambda x: f"{x:0.3f}")

    print(f"Using K_BAR={K_BAR}\n")

    print("Computing Table 1...")
    t1 = table1()

    out_txt = "scripts/results/table_1_2_3/table1.txt"
    with open(out_txt, "w", encoding="utf-8") as f:

        # -------- TABLE 1 ----------
        for k, df in t1.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TABLE 1 - {k}\n")
            f.write("=" * 80 + "\n")
            f.write(df.to_string())
            f.write("\n")
    print("\nComputing Table 2...")
    t2 = table2()

    out_txt = "scripts/results/table_1_2_3/table2.txt"

    with open(out_txt, "w", encoding="utf-8") as f:

        # -------- TABLE 2 ----------
        for k, df in t2.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TABLE 2 - {k}\n")
            f.write("=" * 80 + "\n")
            f.write(df.to_string())
            f.write("\n")

    print("\nComputing Table 3...")
    t3 = table3()

    out_txt = "scripts/results/table_1_2_3/table3.txt"

    with open(out_txt, "w", encoding="utf-8") as f:

        # -------- TABLE 3 ----------
        for k, df in t3.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TABLE 3 - {k}\n")
            f.write("=" * 80 + "\n")
            f.write(df.to_string())
            f.write("\n")

    print(f"TXT écrit : {out_txt}")


if __name__ == "__main__":
    main()

