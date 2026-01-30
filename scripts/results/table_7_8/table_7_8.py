import glob
import os
from typing import Dict, List

import pandas as pd

# output dir
MIDAS_DIR = "scripts/results/table_7_8/results_rmse_midas"
KALMAN_DIR = "scripts/results/table_7_8/results_rmse_kalman"

# output table 7
OUTPUT_PATH_7 = "scripts/results/table_7_8/table_7_final_2024"
# output  table 8
OUTPUT_PATH_8 = "scripts/results/table_7_8/table_8_final_2024"

HORIZONS = list(range(1, 9))

DISPLAY_NAMES: Dict[str, str] = {
    "TERM": "Term Spread",
    "SP": "SP 500",
    "IP": "Industrial Production",
    "Emply": "Employment",
    "Exptn": "Expectations",
    "PI": "Personal Income",
    "Oil": "Crude Oil Price",
    "LEI": "Leading Index",
    "Manu": "Manufacturing",
    "Exptn": "Expectations",
}

SHORT_NAMES: Dict[str, str] = {
    "TERM": "Term",
    "SP": "SP",
    "IP": "IP",
    "Emply": "Emply",
    "Exptn": "Exptn",
    "PI": "PI",
    "Oil": "Oil",
    "LEI": "LEI",
    "Manu": "Manu",
}

ROW_LABELS = [
    "State Space (m0)",
    "Regular Midas (m1)",
    "Multiple Midas (m2)",
    "Ratio (m0/m1)",
    "Ratio (m0/m2)",
]

TABLE8_MODELS = [
    "State Space (m0)",
    "Regular Midas (m1)",
    "Multiple Midas (m2)",
]

TABLE8_MODEL_LABELS = {
    "State Space (m0)": "State Space",
    "Regular Midas (m1)": "Regular MIDAS",
    "Multiple Midas (m2)": "Multiplicative MIDAS",
}


def _load_midas(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["h"].isin(HORIZONS)].copy()
    df = df.set_index("h")[["rmse_regular", "rmse_multiplicative"]]
    return df


def _load_kalman(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df = df[df["h"].isin(HORIZONS)].copy()
    if "RMSE_SSM" in df.columns:
        df = df.rename(columns={"RMSE_SSM": "rmse_ssm"})
    df = df.set_index("h")[["rmse_ssm"]]
    return df


def _regressor_code(path: str) -> str:
    base = os.path.basename(path)
    name = base.rsplit(".", 1)[0]
    if name.startswith("rmse_kalman_"):
        remainder = name[len("rmse_kalman_") :]
        return remainder.rsplit("_", 1)[0]
    if name.startswith("rmse_midas_"):
        remainder = name[len("rmse_midas_") :]
        return remainder.split("_", 1)[0]
    return base.split("_")[-1].split(".")[0]


def _build_block(code: str, midas: pd.DataFrame, kalman: pd.DataFrame) -> pd.DataFrame:
    merged = kalman.join(midas, how="inner")
    merged = merged.reindex(HORIZONS)
    merged["ratio_m0_m1"] = merged["rmse_ssm"] / merged["rmse_regular"]
    merged["ratio_m0_m2"] = merged["rmse_ssm"] / merged["rmse_multiplicative"]

    rows = [
        merged["rmse_ssm"],
        merged["rmse_regular"],
        merged["rmse_multiplicative"],
        merged["ratio_m0_m1"],
        merged["ratio_m0_m2"],
    ]
    block = pd.DataFrame(rows, index=ROW_LABELS, columns=HORIZONS)
    block.index.name = "model"
    return block


def load_blocks():
    midas_files = { _regressor_code(p): p for p in glob.glob(os.path.join(MIDAS_DIR, "rmse_midas_*.csv")) }
    kalman_files = { _regressor_code(p): p for p in glob.glob(os.path.join(KALMAN_DIR, "rmse_kalman_*.xlsx")) }

    all_codes = set(midas_files) & set(kalman_files)
    preferred_order = [k for k in DISPLAY_NAMES if k in all_codes]
    remaining = sorted(all_codes - set(preferred_order))
    codes: List[str] = preferred_order + remaining
    
    blocks = []
    for code in codes:
        midas = _load_midas(midas_files[code])
        kalman = _load_kalman(kalman_files[code])
        block = _build_block(code, midas, kalman)
        reg_name = DISPLAY_NAMES.get(code, code)
        short_name = SHORT_NAMES.get(code, code)
        blocks.append({"code": code, "name": reg_name, "short": short_name, "block": block})
    if not blocks:
        raise ValueError("No matching regressor files found in results_rmse.")

    return blocks


def build_final_table(blocks) -> pd.DataFrame:
    return _flatten_blocks(blocks)


def _flatten_blocks(blocks) -> pd.DataFrame:
    rows = []
    for item in blocks:
        reg_name = item["name"]
        block = item["block"]
        header = {"label": reg_name}
        for h in HORIZONS:
            header[h] = ""
        rows.append(header)
        for model in ROW_LABELS:
            row = {"label": model}
            for h in HORIZONS:
                row[h] = block.loc[model, h]
            rows.append(row)
    return pd.DataFrame(rows, columns=["label"] + HORIZONS)


def build_table8(blocks) -> pd.DataFrame:
    best_reg = {model: {} for model in TABLE8_MODELS}
    best_rmse = {model: {} for model in TABLE8_MODELS}

    for h in HORIZONS:
        for model in TABLE8_MODELS:
            best_item = min(
                blocks,
                key=lambda item: item["block"].loc[model, h],
            )
            best_reg[model][h] = best_item["short"]
            best_rmse[model][h] = best_item["block"].loc[model, h]

    rows = []
    for idx, model in enumerate(TABLE8_MODELS):
        group_label = "Best Predictor" if idx == 0 else ""
        row = {"group": group_label, "model": TABLE8_MODEL_LABELS[model]}
        for h in HORIZONS:
            row[h] = best_reg[model][h]
        rows.append(row)

    for idx, model in enumerate(TABLE8_MODELS):
        group_label = "RMSE" if idx == 0 else ""
        row = {"group": group_label, "model": TABLE8_MODEL_LABELS[model]}
        for h in HORIZONS:
            row[h] = best_rmse[model][h]
        rows.append(row)

    return pd.DataFrame(rows, columns=["group", "model"] + HORIZONS)


def _add_hlines(latex: str) -> str:
    lines = latex.splitlines()
    out = []
    for line in lines:
        out.append(line)
        stripped = line.strip()
        if stripped.startswith("h (Quarters) &"):
            out.append("\\hline")
            continue
        if stripped.endswith("\\\\") and " &  & " in stripped and not stripped.startswith("h (Quarters) &"):
            out.insert(-1, "\\hline")
            out.append("\\hline")
    return "\n".join(out)


def _use_hlines(latex: str) -> str:
    return (
        latex.replace("\\toprule", "\\hline")
        .replace("\\midrule", "\\hline")
        .replace("\\bottomrule", "\\hline")
    )


def _add_hlines_table8(latex: str) -> str:
    lines = latex.splitlines()
    out = []
    for line in lines:
        out.append(line)
        stripped = line.strip()
        if stripped.startswith("h (Quarter) &"):
            out.append("\\hline")
            continue
        if stripped.startswith("RMSE &"):
            out.insert(-1, "\\hline")
    return "\n".join(out)


def main() -> None:
    blocks = load_blocks()

    table7 = build_final_table(blocks)
    table7 = table7.round(2)
    table7.to_csv(OUTPUT_PATH_7 + ".csv")

    latex = table7.to_latex(
        index=False,
        float_format="%.2f",
        column_format="lcccccccc"
    )
    latex = latex.replace("label", "h (Quarters)", 1)
    latex = _add_hlines(latex)
    latex = _use_hlines(latex)
    latex = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\small\n"
        + latex
        + "\n\\end{table}\n"
    )
    with open(OUTPUT_PATH_7 + ".tex", "w", encoding="utf-8") as f:
        f.write(latex)

    table8 = build_table8(blocks)
    table8.to_csv(OUTPUT_PATH_8 + ".csv", index=False)
    latex8 = table8.to_latex(
        index=False,
        float_format="%.2f",
        column_format="clcccccccc",
    )
    latex8 = latex8.replace("group", "h (Quarter)", 1)
    latex8 = _add_hlines_table8(latex8)
    latex8 = _use_hlines(latex8)
    latex8 = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\small\n"
        + latex8
        + "\n\\end{table}\n"
    )
    with open(OUTPUT_PATH_8 + ".tex", "w", encoding="utf-8") as f:
        f.write(latex8)


if __name__ == "__main__":
    main()
