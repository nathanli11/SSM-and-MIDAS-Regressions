# SSM AND MIDAS REGRESSIONS

## INSTALLATIONS
- Dans un terminal, à la racine du projet, lancer la commande: "pip install -e ."
- Les scripts pouvant être exécutés directement se trouvent dans le dossier scripts/

## ORGANISATION DU CODE:

SSM-AND-MIDAS-REGRESSIONS/
├─ data/
│  ├─ data.xlsx                        # données brutes (source)
│  ├─ stationary_data.xlsx             # données transformées (stationnaires + alignées)
│  └─ y_sparse.xlsx                    # version low-frequency / cible avec trous, etc.
│
├─ scripts/                            # “entry points” : scripts exécutables (pipeline, tables)
│  └─ results/
│     ├─ table_1_2_3/
│     │  ├─ table1.txt                 # outputs (papier) / logs / tables finales
│     │  ├─ table2.txt
│     │  └─ table3.txt
│     ├─ table_4_5/
│     │  ├─ table_4_5.py               # script de print des tables 4-5 en LaTex
│     │  └─ Table_4A_*.xlsx ...        # outputs détaillés par panel
│     ├─ table_7_8/
|     │   ├─ results_rmse_kalman_2024/
|     │   |   └─ rmse_kalman_*.xlsx    # outputs rmse kalman par regressor
|     │   ├─ results_rmse_midas_2024/
|     │   |   └─ rmse_midas_*.csv      # outputs rmse midas (regular et multiplicative) par regressor
|     │   ├─ forecast _*.xlsx          # outputs forecasts kalman par regressor
│     │   ├─ table_7_8.py              # script principal table 7-8 (RMSE, forecasting)
│     │   ├─ table_*_final_2024.csv    # output csv des données pour les tableaux 7 et 8 (RMSE, forecasting, Kalman + MIDAS)
│     │   └─ table_*_final_2024.tex    # output LaTex
│     ├─ data_prep.py                  # préparation des datasets pour table 7-8
│     ├─ mc_simulations.py             # simulations Monte-Carlo (sections 3/4)
│     ├─ ssm_midas_rmse.py             # exécutions forecasting & RMSE (SSM + MIDAS)
│     └─ ssm_midas_oos.py              # out-of-sample / recursive evaluation
│
└─ src/                          # code “librairie” réutilisable (importable + testable)
   ├─ dgp/
   │  └─ simulate.py             # DGP: générateurs de données simulées
   ├─ evaluation/
   │  ├─ data_management.py      # lecture/alignement/normalisation des données
   │  ├─ model_selection.py      # AIC/BIC, choix lags, choix facteurs
   │  ├─ recursive_table7.py     # boucle recursive RMSE (table 7)
   │  └─ table3_population.py    # calculs “en population” (section 3)
   ├─ midas/
   │  ├─ midas_base.py           # poids Almon, design matrices, indexer mixed-freq
   │  ├─ midas_fit.py            # estimation (NLS/OLS), objets de résultats
   │  ├─ adl_midas.py            # modèles ADL-MIDAS (regular + multiplicative)
   │  └─ forecast.py             # prévisions MIDAS (horizons, nowcast)
   └─ ssm/
      ├─ params.py               # dataclasses paramètres SSM
      ├─ measurement.py          # construction Z_j / H_j / mapping obs->state
      ├─ periodic_kf.py          # Riccati périodique, steady-state, filter
      ├─ likelihood.py           # loglike Kalman + MLE
      ├─ kalman_weights.py       # extraction des poids (Kalman gain / weights)
      ├─ kalman_two_series.py    # cas 2 séries (ou variantes) si nécessaire
      └─ forecast.py             # prévisions SSM à partir des états filtrés

