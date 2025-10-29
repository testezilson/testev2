import pandas as pd

# Caminho do dataset original
CAMINHO_DATASET = "lol_under_over_model/original_dataset.csv"

# Lê o dataset
print("📂 Carregando dataset...")
df = pd.read_csv(CAMINHO_DATASET)

print(f"✅ Dataset carregado ({len(df)} jogos totais)\n")

# Liga e campeão desejados
league = input("Digite a liga (ex: LPL, LCK, CBLOL): ").strip().upper()
champion = input("Digite o nome do campeão (ex: Mordekaiser): ").strip()

# Filtra pela liga
df_league = df[df["league"].str.upper() == league]

# Colunas de campeões (time 1 e time 2)
champ_cols = [
    "top_t1", "jung_t1", "mid_t1", "adc_t1", "sup_t1",
    "top_t2", "jung_t2", "mid_t2", "adc_t2", "sup_t2"
]

# Conta aparições
count = (df_league[champ_cols] == champion).sum().sum()

print(f"🔎 {champion} apareceu {count} vezes na liga {league}.")

# Exibe percentual
total_games = len(df_league)
print(f"📊 Representa {count / total_games * 100:.2f}% dos {total_games} jogos da {league}.")
