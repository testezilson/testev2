import pandas as pd

df = pd.read_csv("lol_bets_with_draft_filtered.csv")

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
    "bet_value",
]

df = df[colunas]

df.to_csv("real_data_clean_ml.csv", index=False)
