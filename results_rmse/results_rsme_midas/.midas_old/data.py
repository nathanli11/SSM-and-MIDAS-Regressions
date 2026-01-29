import pandas as pd
import numpy as np
import json
from typing import Iterable, Optional, Tuple, Dict, Any, List, Callable, Union
from midas_old.dl_midas import DLMidas
from midas_old.adl_midas import ADLMidas

MIDASModel = Union[DLMidas, ADLMidas]
ModelFactory = Callable[[], MIDASModel]

# ---------- Stationnarisation et normalisation ----------
def _safe_log(s: pd.Series) -> np.ndarray:
    """ Passe une série en log """
    s = s.astype(float)
    s = s.where(s > 0, np.nan)
    return np.log(s)

def stationarize(df: pd.DataFrame, rules: Dict[str, str]) -> pd.DataFrame:
    """ Applique une méthode de stationaisation (basée sur le nom de la série et un set de règles) à chaque colonne d'un DataFrame """
    if df.shape[1] < 1:
        raise ValueError("DataFrame must have at least one column")
    
    out = {}
    for col in df.columns:
        # Stationarization method
        if col in rules['level']:
            out[col] = df[col].copy()
        elif col in rules['log']:
            out[col] = _safe_log(df[col])
        else:
            out[col] = _safe_log(df[col]).diff()
    
    return pd.DataFrame(out).sort_index()

def normalize_full_sample(df: pd.DataFrame) -> pd.DataFrame:
    """ Z-score full sample, colonne par colonne, après transformation """
    df = df.copy()

    means = df.mean(skipna=True)
    stds = df.std(skipna=True, ddof=0)
    stds = stds.replace(0.0, np.nan)

    return (df - means) / stds

# ---------- Data management ----------
def _read_sheet_as_series(xls_path: str, sheet_name: str) -> pd.Series:
    """ Lit une feuille Excel donnée et retourne une Series avec un index DateTime """
    df = pd.read_excel(xls_path, sheet_name=sheet_name)

    if df.shape[1] < 2:
        raise ValueError(f"La feuille '{sheet_name}' doit contenir au moins 2 colonnes (date, valeur).")

    date_col = df.columns[0]
    val_col = df.columns[1]

    df = df[[date_col, val_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    s = df.set_index(date_col)[val_col].sort_index()
    s.name = sheet_name
    return s

def _is_daily(s: pd.Series) -> bool:
    """ Détermine si une série est en fréquence daily ou non """
    if len(s.index) < 2:
        return False
    idx = pd.DatetimeIndex(s.index).sort_values()
    delta_days = (idx[1] - idx[0]).days
    return delta_days <= 15

def _to_monthly_first_obs(s: pd.Series) -> pd.Series:
    """ Mensualise une série journalière en prenant la première observation de chaque mois """
    s = s.sort_index()
    m = s.groupby(pd.Grouper(freq="MS")).first()
    return m

def keep_solid_series(df: pd.DataFrame) -> pd.DataFrame:
    """ Pour chaque colonne d'un DataFrame, retourne la série ininterrompue la plus longue de chaque colonne (complétées par des NaN)
    en démarrant à la start date entrée """

    df = df.copy().sort_index()

    for col in df.columns:
        s = df[col]
        valid = s.notna()

        if not valid.any():
            # colonne full NA
            continue

        # Identifie les "runs" (blocs) de True/False
        run_id = (valid != valid.shift(fill_value=False)).cumsum()

        # Tailles des blocs True uniquement
        true_run_sizes = valid.groupby(run_id).sum()  # somme des True = longueur du bloc

        # On garde seulement les runs où valid==True (le groupe commence par True)
        # (groupe True si la première valeur du groupe est True)
        is_true_run = valid.groupby(run_id).first()
        true_run_sizes = true_run_sizes[is_true_run]

        if true_run_sizes.empty:
            continue

        # Run le plus long (en nombre de lignes)
        best_run = true_run_sizes.idxmax()

        # On met tout le reste à NA
        keep_mask = (run_id == best_run) & valid
        df.loc[~keep_mask, col] = pd.NA

    return df

def load_excel_timeseries(xls_path: str, gdp_sheet: str = "GDP",
                          exclude_sheets: Optional[Iterable[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Applique les transformations et renvoie un tuple de DataFrames: le GDP et les regresseurs """
    exclude = set(exclude_sheets or [])

    xls = pd.ExcelFile(xls_path)
    sheets = xls.sheet_names

    if gdp_sheet not in sheets:
        raise ValueError(f"Feuille GDP introuvable. Feuilles dispo: {sheets}")

    # --- GDP ---
    series_gdp = _read_sheet_as_series(xls_path, gdp_sheet)
    df_gdp = series_gdp.to_frame()
    df_gdp = keep_solid_series(df_gdp)

    # --- Regressors ---
    regressors = []
    for sheet in sheets:
        if sheet == gdp_sheet:
            continue
        if sheet in exclude:
            continue

        s = _read_sheet_as_series(xls_path, sheet)

        if _is_daily(s):
            s = _to_monthly_first_obs(s)

        regressors.append(s)

    df_reg = pd.concat(regressors, axis=1).sort_index() if regressors else pd.DataFrame()
    print('debug\n', df_reg)
    df_reg = keep_solid_series(df_reg)

    # --- Combine Manu / Manu1 ---
    if "Manu" in df_reg.columns or "Manu1" in df_reg.columns:
        manu = df_reg["Manu"] if "Manu" in df_reg.columns else pd.Series(dtype="float64", name="Manu")
        manu1 = df_reg["Manu1"] if "Manu1" in df_reg.columns else pd.Series(dtype="float64", name="Manu1")

        # combine_first : prend Manu, et complète les NaN avec Manu1
        manu_combined = manu.combine_first(manu1)
        manu_combined.name = "Manu"

        # Drop anciennes colonnes et remettre Manu
        cols_to_drop = [c for c in ["Manu", "Manu1"] if c in df_reg.columns]
        df_reg = df_reg.drop(columns=cols_to_drop, errors="ignore")
        df_reg["Manu"] = manu_combined

    # --- TERM = T10 - T1 ---
    if "T10" in df_reg.columns and "T1" in df_reg.columns:
        df_reg["TERM"] = df_reg["T10"] - df_reg["T1"]
        df_reg = df_reg.drop(columns=["T10", "T1"], errors="ignore")
    else:
        # Si l'un manque, on ne crée pas TERM (ou tu peux lever une erreur si tu préfères)
        pass

    # Tri final
    df_reg = df_reg.sort_index()

    # Stationnarisation
    df_gdp_stat = stationarize(df_gdp, stat_rules)              # DataFrame trimestriel (Δln GDP)
    df_reg_stat = stationarize(df_reg, stat_rules)              # DataFrame mensuel (lv/ln/Δln selon rules)

    # Normalisation full-sample
    df_gdp_norm = normalize_full_sample(df_gdp_stat)
    df_reg_norm = normalize_full_sample(df_reg_stat)
    
    # -- Export --
    df_gdp_norm.to_csv("gdp_stat_norm.csv")
    df_reg_norm.to_csv("regressors_stat_norm.csv")

    return df_gdp_norm['GDP'], df_reg_norm

def to_numpy(x, as_2d: bool = False) -> np.ndarray:
    """ Convertit une Series ou un DataFrame (1 colonne) en np.ndarray """
    if isinstance(x, pd.Series):
        arr = x.to_numpy()
        return arr.reshape(-1, 1) if as_2d else arr

    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Le DataFrame doit avoir exactement une colonne.")
        arr = x.to_numpy()
        return arr if as_2d else arr.ravel()

    raise TypeError("x doit être une pd.Series ou un DataFrame à une colonne.")

# ------------------------------- Recursive estimation -------------------------------

# def _find_effective_dates(y: pd.Series, x: pd.Series, Kx_HF: int, dates_info: List[str], h: int) -> pd.Period:
#     """ Calibre les dates selon les données disponibles, les retards, et l'horizon de forecast 
#         dates_info = [start_estimation_month, end_estimation_quarter, end_eval_month]
#     """
#     temp_start_estimation: pd.Period = pd.Period(dates_info[0], freq='M')
#     temp_end_estimation: pd.Period = pd.Period(dates_info[1], freq='Q')
#     end_eval: pd.Period = pd.Period(dates_info[2], freq='M').asfreq("Q", how="end")

#     # Find estimation period
#     start_estimation: pd.Period = max(temp_start_estimation.asfreq("Q", how="end"), (x.first_valid_index() + Kx_HF + 1).asfreq("Q", how="end") + 1)
#     end_estimation: pd.Period = max(start_estimation + 40, temp_end_estimation)

#     # Find evaluation period
#     start_eval: pd.Period = end_estimation + h

#     # Vérification
#     if start_estimation > end_estimation or end_estimation > start_eval or start_eval > end_eval:
#         raise ValueError(
#             f"{x.name}: Données insuffisantes pour estimer un MIDAS avec k={Kx_HF} et h={h}"
#         )
#     else:
#         return start_estimation, end_estimation, start_eval, end_eval

def first_usable_origin_quarter(
    y: pd.Series,
    x: pd.Series,
    m: int,
    Kx_LF: int,
    j_obs: int,
) -> pd.Period:
    """
    Premier trimestre t tel que:
    - on trouve une observation HF dans t au rang j_cut=min(j_obs, n_t)
    - et on a assez d'historique HF pour Kx_HF lags (cut_pos >= Kx_HF-1)
    """
    Kx_HF = m * Kx_LF

    # y en PeriodIndex trimestriel
    if not isinstance(y.index, pd.PeriodIndex):
        y = y.copy()
        y.index = pd.PeriodIndex(y.index, freq="Q")
    y = y.sort_index()

    # x en DatetimeIndex
    if isinstance(x.index, pd.PeriodIndex):
        x = x.copy()
        x.index = x.index.to_timestamp(how="start")
    elif not isinstance(x.index, pd.DatetimeIndex):
        x = x.copy()
        x.index = pd.to_datetime(x.index)

    df = pd.DataFrame({"x": x})
    idx = pd.DatetimeIndex(df.index)
    df["q"] = idx.to_period("Q")
    df["j"] = df.groupby("q").cumcount() + 1

    for t in y.index:
        g = df[df["q"] == t]
        if g.empty:
            continue
        n_t = int(g["j"].max())
        j_cut = min(int(j_obs), n_t)

        mask_cut = (df["q"] == t) & (df["j"] == j_cut)
        pos = np.flatnonzero(mask_cut.values)
        if pos.size == 0:
            continue
        cut_pos = int(pos[0])

        if cut_pos >= (Kx_HF - 1):
            return t

    raise ValueError("Aucun trimestre d'origine faisable (HF insuffisant / trimestres vides).")

def _find_effective_dates(
    y: pd.Series,
    x: pd.Series,
    model: MIDASModel,
    dates_info: List[str],
    h: int,
) -> Tuple[pd.Period, pd.Period, pd.Period, pd.Period]:

    # paramètres MIDAS
    m = model.m
    Kx_LF = model.Kx_LF
    j_obs = model.j_obs if getattr(model, "j_obs", None) is not None else (m - 1)

    # bornes venant du JSON
    # start est en "M" dans ton fichier; on le convertit vers le quarter de fin de mois
    temp_start_est = pd.Period(dates_info[0], freq="M").asfreq("Q", how="end")
    temp_end_est   = pd.Period(dates_info[1], freq="Q")
    end_eval       = pd.Period(dates_info[2], freq="M").asfreq("Q", how="end")

    # 1) faisabilité technique (dépend de x + m + Kx_LF + j_obs)
    first_feasible = first_usable_origin_quarter(y, x, m=m, Kx_LF=Kx_LF, j_obs=j_obs)

    # 2) start estimation = max(borne JSON, borne technique)
    start_estimation = max(temp_start_est, first_feasible)

    # 3) end estimation: au moins 40 trimestres, mais >= temp_end_est
    end_estimation = max(start_estimation + 40, temp_end_est)

    # 4) fenêtre d’évaluation
    start_eval = end_estimation + h

    if start_estimation > end_estimation or end_estimation > start_eval or start_eval > end_eval:
        Kx_HF = m * Kx_LF
        raise ValueError(
            f"{getattr(x, 'name', 'x')}: Données insuffisantes pour DL-MIDAS "
            f"(m={m}, Kx_LF={Kx_LF}, Kx_HF={Kx_HF}, j_obs={j_obs}, h={h})."
        )

    return start_estimation, end_estimation, start_eval, end_eval


def recursive_rmse_midas(y: pd.Series, x: pd.Series, model_factory: ModelFactory, h_list: Iterable[int], reg_info: Dict[str, Any]):

    # Conversion de l'index du GDP en pd.Period
    if not isinstance(y.index, pd.PeriodIndex):
        y = y.copy()
        print(y)
        y.index = pd.PeriodIndex(y.index, freq="Q")
    
    rmses = {}
    for h in h_list:
        # Définition des périodes selon h et les données dispo de la série
        tmp_model = model_factory()

        start_estimation, end_estimation, start_eval, end_eval = _find_effective_dates(
            y=y,
            x=x,
            model=tmp_model,
            dates_info=reg_info[x.name],
            h=h)
        print(f'Horizon: {h}, {start_estimation} : {end_estimation}, {start_eval} : {end_eval}')

        errors = []
        # t = trimestre "à prévoir" en nowcast (h=1) ou à horizon h
        # Ici on prend comme origine d’estimation: y dispo jusqu’à (t-1)
        for t in pd.period_range(start_eval, end_eval, freq="Q"):
            est_end = t - 1  # y observé jusqu’à t-1
            if est_end < end_estimation:
                continue
            if (t + (h-1)) not in y.index:
                continue

            model = model_factory()

            y_est = y.loc[start_estimation:est_end]
            # Preparation de Y et X
            # Preparation de Y et X
            if hasattr(model, "p"):  # ADL-MIDAS
                Y, Xlags, Ylags, _ = model.build_adl_midas_xy(
                    y=y_est,
                    x=x,
                    h=h,
                    j_obs=getattr(model, "j_obs", None)
                )
                model.fit(Y, Xlags, Ylags)
            else:  # DL-MIDAS
                Y, Xlags, _ = model.build_midas_xy_generic(
                    y=y_est,
                    x=x,
                    h=h,
                    j_obs=getattr(model, "j_obs", None)
                )
                model.fit(Y, Xlags)


            # Prévision de y_{t+(h-1)} en utilisant info mensuelle jusqu’au 2e mois de t
            # (pour h=1 => y_t)
            target = t + (h-1)
            y_true = float(y.loc[target])

            # construit une seule ligne X au trimestre t (origine = t)
            y_for_pred = y.loc[:est_end]  # y dispo
            yhat = model.predict(y_for_pred, x, h=h, t=t)

            errors.append((yhat - y_true) ** 2)
        rmses[h] = float(np.sqrt(np.mean(errors)))

    return rmses

# ------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # -------------- Inputs --------------
    data_path = "data.xlsx"
    gdp_path = "gdp_stat_norm.csv"
    reg_path = "regressors_stat_norm.csv"

    stat_rules = {
        'level': ['GDP', 'TERM', 'LEI'],
        'log': ['Exptn']
    }
    h_list = range(1, 9)
    with open("regressors_info.json", "r", encoding="utf-8") as f:
        reg_info = json.load(f)
    #model_factory = lambda: DLMidas(Kx_LF=5, m=3, include_intercept=False)
    model_factory = lambda: ADLMidas(Kx_LF=5, m=3, p=1, include_intercept=True, j_obs=None)

    reload_data = False
    # -------------- Data --------------
    if reload_data:
        load_excel_timeseries(
            data_path,
            gdp_sheet="GDP",
            exclude_sheets=["Exptn1"]
        )

    # -------------- Model --------------

    # Import clean data from csv files
    gdp = pd.read_csv(gdp_path)
    gdp["Date"] = pd.to_datetime(gdp["Date"]).dt.to_period("Q")
    gdp = gdp.set_index("Date")
    y = gdp["GDP"]
    
    reg = pd.read_csv(reg_path)
    #reg["Date"] = pd.to_datetime(reg["Date"]).dt.to_period("M").dt.to_timestamp(how="start")
    reg["Date"] = pd.to_datetime(reg["Date"])
    reg = reg.set_index("Date")

    res = {}
    for asset in reg.columns:
        print(f'Asset: {asset}')
        # Select asset
        x = reg[asset]
        # Run model
        res_rmse = recursive_rmse_midas(y, x, model_factory, h_list, reg_info)
        # Append res
        res[asset] = res_rmse
    print(res)
    
    with open('resultats_rmse.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)
