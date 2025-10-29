import pickle
import pandas as pd

# Caminho do arquivo de impactos
CAMINHO_IMPACTS = "lol_under_over_model/champion_impacts_v2.pkl"

# Carrega o dicionário de impactos
with open(CAMINHO_IMPACTS, "rb") as f:
    champion_impacts = pickle.load(f)

# Mostra as ligas disponíveis
ligas = list(champion_impacts.keys())
print("=== LIGAS DISPONÍVEIS ===")
for liga in ligas:
    print("-", liga)

# Escolhe a liga
league = input("\nDigite o nome da liga para visualizar (ex: LCK, LPL, LEC, CBLOL): ").strip().upper()

if league not in champion_impacts:
    print(f"❌ Liga '{league}' não encontrada.")
else:
    data = champion_impacts[league]
    df = pd.DataFrame(list(data.items()), columns=["Campeão", "Impacto"])
    df = df.sort_values(by="Impacto", ascending=False).reset_index(drop=True)

    print(f"\n=== IMPACTOS DOS CAMPEÕES - {league} ===")
    print(df.head(20).to_string(index=False))

    # Estatísticas gerais
    print("\n--- Estatísticas ---")
    print(f"Campeões únicos: {len(df)}")
    print(f"Média de impacto: {df['Impacto'].mean():+.2f}")
    print(f"Maior impacto: {df['Impacto'].max():+.2f}")
    print(f"Menor impacto: {df['Impacto'].min():+.2f}")

    # 🔍 Pesquisa de campeão específico
    while True:
        search = input("\nDigite o nome de um campeão para buscar (ou 'sair' para encerrar): ").strip()
        if search.lower() == "sair":
            break
        if search in df["Campeão"].values:
            valor = df.loc[df["Campeão"] == search, "Impacto"].values[0]
            print(f"✅ {search} na {league}: {valor:+.2f}")
        else:
            print(f"⚠️ Campeão '{search}' não encontrado na {league}.")

    # Exportar se quiser
    salvar = input("\nDeseja exportar para Excel? (s/n): ").strip().lower()
    if salvar == "s":
        caminho_saida = f"impactos_{league}.xlsx"
        df.to_excel(caminho_saida, index=False)
        print(f"✅ Exportado para {caminho_saida}")
