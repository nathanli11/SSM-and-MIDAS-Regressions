import pandas as pd

'''
    Code à lancer uniquement pour récupérer le code LaTex associé aux résultats
'''

repo = "scripts/results/table_4_5/"

panel_A_h1_midas = pd.read_excel(repo + "Table_4A_PanelA_MIDAS.xlsx", index_col=0)
panel_A_h1_adl = pd.read_excel(repo + "Table_4A_PanelA_Multiplicative_MIDAS.xlsx", index_col=0)

panel_B_h4_midas = pd.read_excel(repo + "Table_4A_PanelB_MIDAS.xlsx", index_col=0)
panel_B_h4_adl = pd.read_excel(repo + "Table_4A_PanelB_Multiplicative_MIDAS.xlsx", index_col=0)
panel_C_h1_midas = pd.read_excel(repo + "Table_4B_PanelC_MIDAS.xlsx", index_col=0)
panel_C_h1_adl = pd.read_excel(repo + "Table_4B_PanelC_Multiplicative_MIDAS.xlsx", index_col=0)

panel_D_h4_midas = pd.read_excel(repo + "Table_4B_PanelD_MIDAS.xlsx", index_col=0)
panel_D_h4_adl = pd.read_excel(repo + "Table_4B_PanelD_Multiplicative_MIDAS.xlsx", index_col=0)

panel_A = pd.concat(
    [
        panel_A_h1_midas,   
        panel_A_h1_adl,
        panel_B_h4_midas,
        panel_B_h4_adl
    ],
    axis=1,
    keys=[
        "Regular MIDAS",
        "Multiplicative MIDAS",
        "Regular MIDAS",
        "Multiplicative MIDAS"
    ]
)
panel_A = panel_A.round(3)

latex_panel_A = panel_A.to_latex(
    float_format="%.2f",
    multicolumn=True,
    multirow=True,
    escape=False
)
print(latex_panel_A)

panel_B = pd.concat(
    [
        panel_C_h1_midas,  
        panel_C_h1_adl,
        panel_D_h4_midas,
        panel_D_h4_adl
    ],
    axis=1,
    keys=[
        "Regular MIDAS",
        "Multiplicative MIDAS",
        "Regular MIDAS",
        "Multiplicative MIDAS"
    ]
)
panel_B = panel_B.round(3)

latex_panel_B = panel_B.to_latex(
    float_format="%.2f",
    multicolumn=True,
    multirow=True,
    escape=False
)
print(latex_panel_B)


panel_A_midas = pd.read_excel(repo + "Table_5_PanelA_MIDAS.xlsx", index_col=0)
panel_A_adl = pd.read_excel(repo + "Table_5_PanelA_Multiplicative_MIDAS.xlsx", index_col=0)

panel_B_midas = pd.read_excel(repo + "Table_5_PanelB_MIDAS.xlsx", index_col=0)
panel_B_adl = pd.read_excel(repo + "Table_5_PanelB_Multiplicative_MIDAS.xlsx", index_col=0)
panel_C_midas = pd.read_excel(repo + "Table_5_PanelC_MIDAS.xlsx", index_col=0)
panel_C_adl = pd.read_excel(repo + "Table_5_PanelC_Multiplicative_MIDAS.xlsx", index_col=0)

panel_D_midas = pd.read_excel(repo + "Table_5_PanelD_MIDAS.xlsx", index_col=0)
panel_D_adl = pd.read_excel(repo + "Table_5_PanelD_Multiplicative_MIDAS.xlsx", index_col=0)
panel_C = pd.concat(
    [
        panel_A_midas,  
        panel_A_adl,
        panel_B_midas,
        panel_B_adl
    ],
    axis=1,
    keys=[
        "Regular MIDAS",
        "Multiplicative MIDAS",
        "Regular MIDAS",
        "Multiplicative MIDAS"
    ]
)
panel_C = panel_C.round(3)

latex_panel_c = panel_C.to_latex(
    float_format="%.2f",
    multicolumn=True,
    multirow=True,
    escape=False
)
print(latex_panel_c)

panel_D = pd.concat(
    [
        panel_C_midas,  
        panel_C_adl,
        panel_D_midas,
        panel_D_adl
    ],
    axis=1,
    keys=[
        "Regular MIDAS",
        "Multiplicative MIDAS",
        "Regular MIDAS",
        "Multiplicative MIDAS"
    ]
)
panel_D = panel_D.round(3)

latex_panel_D = panel_D.to_latex(
    float_format="%.2f",
    multicolumn=True,
    multirow=True,
    escape=False
)
print(latex_panel_D)

