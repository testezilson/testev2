import pandas as pd

df = pd.read_csv("database_transformed.csv")

ligas_remover = [
    "PRMP",
    "PCS",
    "HC",
    "LAS",
    "VCS",
    "LJL",
    "EBL",
    "LPLOL",
    "CT",
    "AL",
    "HW",
    "HLL",
    "Asia Master",
    "RL",
    "LRN",
    "NEXO",
    "LFL2",
    "ROL",
    "LTA",
]

df = df[~df["league"].isin(ligas_remover)]

Q1_gamelength = df["gamelength"].quantile(0.25)
Q3_gamelength = df["gamelength"].quantile(0.75)
IQR_gamelength = Q3_gamelength - Q1_gamelength
df = df[
    (df["gamelength"] >= Q1_gamelength - 1.5 * IQR_gamelength)
    & (df["gamelength"] <= Q3_gamelength + 1.5 * IQR_gamelength)
]

Q1_kills = df["total_kills"].quantile(0.25)
Q3_kills = df["total_kills"].quantile(0.75)
IQR_kills = Q3_kills - Q1_kills
df = df[
    (df["total_kills"] >= Q1_kills - 1.5 * IQR_kills)
    & (df["total_kills"] <= Q3_kills + 1.5 * IQR_kills)
]

colunas = [
    "league",
    "t1",
    "t2",
    "top_t1",
    "jung_t1",
    "mid_t1",
    "adc_t1",
    "sup_t1",
    "top_t2",
    "jung_t2",
    "mid_t2",
    "adc_t2",
    "sup_t2",
    "total_kills",
]

df = df[colunas]

df.to_csv("database_clean_to_ml.csv", index=False)
