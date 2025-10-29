# %% Pipeline completo para predição ao vivo - AJUSTADO
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

ROLES = ["top", "jung", "mid", "adc", "sup"]

# ============================================================
# 1. CARREGAR E PROCESSAR DATABASE
# ============================================================

print("=" * 70)
print("CARREGANDO DATABASE E CALCULANDO IMPACTOS")
print("=" * 70)

# Carregar dataset
matches_df = pd.read_csv("database/database_transformed.csv")
print(f"Dataset original: {matches_df.shape[0]} jogos")

# Remover outliers de gamelength
Q1 = matches_df["gamelength"].quantile(0.25)
Q3 = matches_df["gamelength"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_mask = (matches_df["gamelength"] < lower_bound) | (
    matches_df["gamelength"] > upper_bound
)
matches_df = matches_df[~outliers_mask].copy()
print(f"Dataset após remoção de outliers: {matches_df.shape[0]} jogos")

# Remover colunas não necessárias
matches_df = matches_df.drop(
    ["gamelength", "year", "date", "patch", "game"], axis=1, errors="ignore"
)

# ============================================================
# 2. CALCULAR IMPACTOS DOS CAMPEÕES (MÉTODO SIMPLE)
# ============================================================


def calculate_champion_impacts_simple(df, min_games=15):
    overall_mean = df["total_kills"].mean()
    champion_columns = [f"{role}_{team}" for role in ROLES for team in ["t1", "t2"]]
    champion_impacts = {}
    champion_counts = {}

    for col in champion_columns:
        for champion in df[col].dropna().unique():
            if champion not in champion_counts:
                champion_counts[champion] = 0
                champion_impacts[champion] = []

            champion_games = df[df[col] == champion]["total_kills"]
            champion_impacts[champion].extend(champion_games.tolist())
            champion_counts[champion] += len(champion_games)

    final_impacts = {}
    for champion in champion_impacts:
        if champion_counts[champion] >= min_games:
            mean_kills = np.mean(champion_impacts[champion])
            final_impacts[champion] = mean_kills - overall_mean
        else:
            final_impacts[champion] = 0.0

    return final_impacts, overall_mean, champion_counts


impacts_simple, overall_mean, counts_simple = calculate_champion_impacts_simple(
    matches_df, min_games=15
)

impactos_com_efeito = {k: v for k, v in impacts_simple.items() if v != 0.0}
print(f"\nCampeões com impacto calculado: {len(impactos_com_efeito)}")
print(f"Média geral de kills: {overall_mean:.2f}")

# ============================================================
# 3. CALCULAR MÉDIAS DOS TIMES
# ============================================================

all_teams = pd.concat([matches_df["t1"], matches_df["t2"]]).unique()
team_means = {}

for team in all_teams:
    team_games = matches_df[(matches_df["t1"] == team) | (matches_df["t2"] == team)]
    team_means[team] = team_games["total_kills"].mean()

print(f"Times cadastrados: {len(team_means)}")

# Calcular médias por liga também (fallback)
league_means = matches_df.groupby("league")["total_kills"].mean().to_dict()

# ============================================================
# 4. SALVAR DICIONÁRIOS
# ============================================================

data_to_save = {
    "champion_impacts": impacts_simple,
    "team_means": team_means,
    "league_means": league_means,
    "overall_mean": overall_mean,
    "champion_counts": counts_simple,
}

joblib.dump(data_to_save, "champion_team_data.pkl")
print("\n✅ Dados salvos em: champion_team_data.pkl")

# ============================================================
# 5. CARREGAR MODELO E INFO
# ============================================================

MODEL_PATH = "betting_model_pipeline_under.pkl"
INFO_PATH = "model_info_under.pkl"

model = joblib.load(MODEL_PATH)
info = joblib.load(INFO_PATH)

FEATURES = list(info["feature_columns"])  # 12 features
THRESH = 0.55
ODDS = float(info.get("odds", 1.83))

print(f"\n{'=' * 70}")
print(f"MODELO CARREGADO")
print(f"{'=' * 70}")
print(f"Modelo: {info.get('model_name', 'Model')}")
print(f"Threshold: {THRESH}")
print(f"Odds ref: {ODDS}")
print(f"Features esperadas ({len(FEATURES)}): {FEATURES}")

# ============================================================
# 6. PIPELINE DE PREDIÇÃO AO VIVO
# ============================================================


class LivePredictionPipeline:
    def __init__(self, model, features, champion_data):
        self.model = model
        self.features = features
        self.champion_impacts = champion_data["champion_impacts"]
        self.team_means = champion_data["team_means"]
        self.league_means = champion_data["league_means"]
        self.overall_mean = champion_data["overall_mean"]

        print(f"\n✅ Pipeline inicializado")
        print(f"- {len(self.champion_impacts)} campeões")
        print(f"- {len(self.team_means)} times")
        print(f"- {len(self.league_means)} ligas")

    def get_team_avg_kills(self, team_name, league):
        """Busca média de kills do time com fallbacks"""
        if team_name in self.team_means:
            return self.team_means[team_name]
        if league in self.league_means:
            return self.league_means[league]
        return self.overall_mean

    def get_champion_impact(self, champion_name):
        """Busca impacto do campeão"""
        return self.champion_impacts.get(champion_name, 0.0)

    def processar_jogo(self, jogo_data):
        """
        Transforma dados do draft em features para o modelo
        """
        liga = jogo_data["liga"]
        team1 = jogo_data["team1"]
        team2 = jogo_data["team2"]
        draft_blue = jogo_data["draft_blue"]
        draft_red = jogo_data["draft_red"]

        # Buscar médias dos times
        mean_t1 = self.get_team_avg_kills(team1, liga)
        mean_t2 = self.get_team_avg_kills(team2, liga)

        # Criar features
        features = {
            "mean_total_kills_t1": mean_t1,
            "mean_total_kills_t2": mean_t2,
            "impact_top_t1": self.get_champion_impact(draft_blue["top"]),
            "impact_jung_t1": self.get_champion_impact(draft_blue["jung"]),
            "impact_mid_t1": self.get_champion_impact(draft_blue["mid"]),
            "impact_adc_t1": self.get_champion_impact(draft_blue["adc"]),
            "impact_sup_t1": self.get_champion_impact(draft_blue["sup"]),
            "impact_top_t2": self.get_champion_impact(draft_red["top"]),
            "impact_jung_t2": self.get_champion_impact(draft_red["jung"]),
            "impact_mid_t2": self.get_champion_impact(draft_red["mid"]),
            "impact_adc_t2": self.get_champion_impact(draft_red["adc"]),
            "impact_sup_t2": self.get_champion_impact(draft_red["sup"]),
        }

        return pd.DataFrame([features])

    def prever(self, jogo_data, threshold=0.55, odds=1.83):
        """Faz predição completa para um jogo"""
        # Processar features
        X = self.processar_jogo(jogo_data)

        # Garantir que X tem todas as features na ordem correta
        X = X[self.features]

        # Predição
        prob_under = self.model.predict_proba(X)[0][1]
        prob_over = self.model.predict_proba(X)[0][0]

        # Decisão
        apostar_under = prob_under >= threshold

        # EV
        ev_under = prob_under * (odds - 1) - (1 - prob_under)

        # Calcular estimativa
        soma_impactos = X.iloc[0, 2:].sum()  # Todos os impactos
        estimativa_kills = (X.iloc[0, 0] + X.iloc[0, 1]) / 2 + soma_impactos

        return {
            "prob_under": prob_under,
            "prob_over": prob_over,
            "apostar_under": apostar_under,
            "ev_under": ev_under,
            "estimativa_kills": estimativa_kills,
            "features": X.iloc[0].to_dict(),
        }


# ============================================================
# 7. INICIALIZAR PIPELINE
# ============================================================

pipeline = LivePredictionPipeline(
    model=model, features=FEATURES, champion_data=data_to_save
)

# ============================================================
# 8. EXEMPLO DE USO COM DADOS DO APP
# ============================================================

print(f"\n{'=' * 70}")
print("TESTE COM DADOS DO APP")
print(f"{'=' * 70}")

# Dados do jogo (formato do seu app)
jogo = {
    "liga": "LEC",
    "team1": "Movistar KOI",
    "team2": "G2 Esports",
    "draft_blue": {
        "top": "Sion",
        "jung": "Wukong",
        "mid": "Azir",
        "adc": "Sivir",
        "sup": "Braum",
    },
    "draft_red": {
        "top": "Kled",
        "jung": "Xin Zhao",
        "mid": "Taliyah",
        "adc": "Zeri",
        "sup": "Rell",
    },
}

print(f"\nJogo: {jogo['team1']} vs {jogo['team2']} ({jogo['liga']})")
print(f"\nDRAFT BLUE ({jogo['team1']}):")
for role, champ in jogo["draft_blue"].items():
    impact = pipeline.get_champion_impact(champ)
    print(f"  {role.upper():5} - {champ:12} | Impacto: {impact:+6.3f}")

print(f"\nDRAFT RED ({jogo['team2']}):")
for role, champ in jogo["draft_red"].items():
    impact = pipeline.get_champion_impact(champ)
    print(f"  {role.upper():5} - {champ:12} | Impacto: {impact:+6.3f}")

# Fazer predição
resultado = pipeline.prever(jogo, threshold=THRESH, odds=ODDS)

print(f"\n{'=' * 70}")
print("FEATURES QUE ENTRAM NO MODELO")
print(f"{'=' * 70}")
for key, value in resultado["features"].items():
    print(f"{key:25} = {value:8.4f}")

print(f"\n{'=' * 70}")
print("RESULTADO DA PREDIÇÃO")
print(f"{'=' * 70}")
print(f"Probabilidade UNDER:     {resultado['prob_under']:.1%}")
print(f"Probabilidade OVER:      {resultado['prob_over']:.1%}")
print(f"Valor Esperado (EV):     {resultado['ev_under']:+.4f}")
print(f"Estimativa total_kills:  {resultado['estimativa_kills']:.1f}")
print(
    f"\nRecomendação: {'✅ APOSTAR UNDER' if resultado['apostar_under'] else '❌ NÃO APOSTAR'}"
)

# ============================================================
# 9. FUNÇÃO PARA INTEGRAÇÃO COM APP
# ============================================================


def predict_from_app_data(app_data, pipeline, threshold=0.55, odds=1.83):
    """
    Recebe dados do app e retorna predição formatada
    """
    try:
        resultado = pipeline.prever(app_data, threshold=threshold, odds=odds)

        return {
            "sucesso": True,
            "prob_under": round(resultado["prob_under"], 3),
            "prob_over": round(resultado["prob_over"], 3),
            "ev": round(resultado["ev_under"], 4),
            "apostar": resultado["apostar_under"],
            "estimativa": round(resultado["estimativa_kills"], 1),
            "confianca": "ALTA"
            if resultado["prob_under"] >= 0.70
            else "MEDIA"
            if resultado["prob_under"] >= 0.55
            else "BAIXA",
            "features": resultado["features"],
        }
    except Exception as e:
        return {"sucesso": False, "erro": str(e)}


resultado_app = predict_from_app_data(jogo, pipeline, threshold=THRESH, odds=ODDS)

print(f"\n{'=' * 70}")
print("FORMATO PARA APP")
print(f"{'=' * 70}")
for key, value in resultado_app.items():
    if key != "features":
        print(f"{key}: {value}")
