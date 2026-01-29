import numpy as np
import pandas as pd

# -----------------------------
# Réglages "papier" (Table 6 + Section 4)
# -----------------------------
PAPER_Q_START = "1959Q1"
PAPER_Q_END   = "2026Q1"   # GDP jusqu'à 2009Q1
PAPER_M_END   = "2026-01-31"  # indicateurs mensuels jusqu'à May 2009

print("RUNNING prepare_data_table6.py from:", __file__)


def _to_datetime(df: pd.DataFrame, date_col="Date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col)

def _series_from_sheet(xlsx_path: str, sheet: str, value_col: str) -> pd.Series:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df = _to_datetime(df, "Date")
    s = pd.Series(df[value_col].astype(float).values, index=df["Date"])
    s.name = sheet
    return s

def dlog(s: pd.Series, scale: float = 100.0) -> pd.Series:
    """Δ ln(s) (Table 6)"""
    s = s.astype(float)
    s = s.where(s > 0, np.nan)
    return scale * np.log(s).diff()

def ln(s: pd.Series) -> pd.Series:
    """ln(s) (Table 6)"""
    s = s.astype(float)
    s = s.where(s > 0, np.nan)
    return np.log(s)

def zscore_on_window(s: pd.Series, start=None, end=None) -> pd.Series:
    """
    Normalisation "papier": moyenne/variance sur une fenêtre fixe (full sample de l'étude).
    """
    ss = s.copy()
    if start is not None:
        ss = ss[ss.index >= pd.Timestamp(start)]
    if end is not None:
        ss = ss[ss.index <= pd.Timestamp(end)]
    mu = ss.mean()
    sd = ss.std(ddof=1)
    return (s - mu) / sd

def prepare_table6_from_xlsx(
    xlsx_path: str,
    scale_dln: float = 100.0,
    normalize_like_paper: bool = True,
    out_gdp_csv: str = "gdp_table6.csv",
    out_reg_csv: str = "regressors_table6.csv",
):
    # -----------------------------
    # 1) GDP (trimestriel) : growth rate
    # -----------------------------
    gdp_raw = _series_from_sheet(xlsx_path, "GDP", "A191RL1Q225SBEA")
    # GDP growth rate = Δ ln(GDP) (souvent *100)
    gdp_gr = dlog(gdp_raw, scale=scale_dln)
    gdp_q = gdp_gr.copy()
    gdp_q.index = gdp_q.index.to_period("Q")
    gdp_q = gdp_q.sort_index()
    gdp_q.name = "GDP_GROWTH"

    # Restrict paper evaluation sample (pour reproduire Table 7)
    gdp_q_paper = gdp_q.loc[pd.Period(PAPER_Q_START):pd.Period(PAPER_Q_END)]

    if normalize_like_paper:
        # Normalisation sur la fenêtre "papier" (full sample de l'étude)
        # Ici: mean/var sur 1959Q1..2009Q1
        mask = (gdp_q.index >= pd.Period(PAPER_Q_START)) & (gdp_q.index <= pd.Period(PAPER_Q_END))
        mu = gdp_q.loc[mask].mean()
        sd = gdp_q.loc[mask].std(ddof=1)
        gdp_q_z = (gdp_q - mu) / sd
        gdp_q_z = gdp_q_z.loc[pd.Period(PAPER_Q_START):pd.Period(PAPER_Q_END)]
    else:
        gdp_q_z = gdp_q_paper.copy()

    gdp_out = pd.DataFrame({
        "Date": gdp_q_z.index.to_timestamp(how="start"),
        "GDP_GROWTH": gdp_q_z.values
    })
    gdp_out.to_csv(out_gdp_csv, index=False)

    # -----------------------------
    # 2) Mensuels : Table 6 transformations
    # -----------------------------
    T10 = _series_from_sheet(xlsx_path, "T10", "DGS10")
    T1  = _series_from_sheet(xlsx_path, "T1",  "DGS1")
    TERM = (T10 - T1)
    TERM.name = "TERM"  # lv

    SP500 = _series_from_sheet(xlsx_path, "SP", "SPX Index")
    SP500_t = dlog(SP500, scale=scale_dln)  # Δln

    IP = _series_from_sheet(xlsx_path, "IP", "INDPRO")
    IP_t = dlog(IP, scale=scale_dln)        # Δln

    Emply = _series_from_sheet(xlsx_path, "Emply", "PAYEMS")
    Emply_t = dlog(Emply, scale=scale_dln)  # Δln

    Exptn = _series_from_sheet(xlsx_path, "Exptn", "UMCSENT")
    Exptn_t = ln(Exptn)                     # ln

    PI = _series_from_sheet(xlsx_path, "PI", "W875RX1")
    PI_t = dlog(PI, scale=scale_dln)        # Δln

    LEI = _series_from_sheet(xlsx_path, "LEI", "USALORSGPNOSTSAM")
    LEI_t = LEI.astype(float)               # lv (déjà % change / mois)

    Manu = _series_from_sheet(xlsx_path, "Manu", "INVHMRMT")
    Manu_t = dlog(Manu, scale=scale_dln)    # Δln

    Oil = _series_from_sheet(xlsx_path, "Oil", "WTISPLC")
    Oil_t = dlog(Oil, scale=scale_dln)      # Δln

    # assembler en DataFrame mensuel
    reg = pd.concat([
        TERM,
        SP500_t.rename("SP500"),
        IP_t.rename("IP"),
        Emply_t.rename("Emply"),
        Exptn_t.rename("Exptn"),
        PI_t.rename("PI"),
        LEI_t.rename("LEI"),
        Manu_t.rename("Manu"),
        Oil_t.rename("Oil"),
    ], axis=1).sort_index()

    # limiter à l’échantillon papier (mensuel jusqu’à 2009:05)
    reg = reg.loc[(reg.index >= pd.Timestamp("1959-01-01")) & (reg.index <= pd.Timestamp(PAPER_M_END))]

    if normalize_like_paper:
        # Normalisation "papier": mean/var sur full sample de l'étude (1959-01..2009-05)
        reg_z = reg.apply(lambda s: zscore_on_window(s, start="1959-01-01", end=PAPER_M_END))
    else:
        reg_z = reg.copy()

    reg_out = reg_z.copy()
    reg_out.insert(0, "Date", reg_out.index)
    reg_out.to_csv(out_reg_csv, index=False)

    print(f"Saved: {out_gdp_csv}")
    print(f"Saved: {out_reg_csv}")
    return gdp_out, reg_out


if __name__ == "__main__":
    prepare_table6_from_xlsx(
        xlsx_path="data.xlsx",
        scale_dln=100.0,                 # mettre 1.0 si tu veux Δln “sans %”
        normalize_like_paper=True,       # comme le papier (full sample 1959..2009)
        out_gdp_csv="gdp_table6_stat_norm.csv",
        out_reg_csv="regressors_table6_stat_norm.csv",
    )
