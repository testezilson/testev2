import pickle
import json
import numpy as np
import pandas as pd

# ==========================
# 1. Carrega modelos e componentes
# ==========================
def load_components():
    with open('lol_under_over_model/trained_models_v2.pkl', 'rb') as f:
        models = pickle.load(f)
    with open('lol_under_over_model/scaler_v2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('lol_under_over_model/champion_impacts_v2.pkl', 'rb') as f:
        champion_impacts = pickle.load(f)
    with open('lol_under_over_model/league_stats_v2.pkl', 'rb') as f:
        league_stats = pickle.load(f)
    with open('lol_under_over_model/feature_columns_v2.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return models, scaler, champion_impacts, league_stats, feature_cols


# ==========================
# 2. Predição principal
# ==========================
def predict_game(game_data, models, scaler, champion_impacts, league_stats, feature_cols, threshold=0.65):
    from lol_under_over_model.load_and_predict_v2 import predict_game as predict_core
    return predict_core(game_data, models, scaler, champion_impacts, league_stats, feature_cols, threshold)


# ==========================
# 3. Conversor JSON seguro
# ==========================
def default_serializer(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray, list)):
        return obj.tolist()
    raise TypeError(f"Tipo não serializável: {type(obj)}")


# ==========================
# 4. Função auxiliar – impacto de campeões
# ==========================
def get_champion_impact(champion_impacts, league, champion_name):
    """Retorna o impacto do campeão na liga e o motivo (caso 0.00)."""
    impacts_liga = champion_impacts.get(league, {})
    if champion_name in impacts_liga:
        impact_value = impacts_liga[champion_name]
        if impact_value == 0:
            reason = "🟡 impacto neutro (diferença ≈ 0)"
        else:
            reason = None
    else:
        if len(impacts_liga) == 0:
            reason = f"🔴 liga {league} sem dados"
        else:
            reason = "⚠️ sem dados suficientes ou nome incorreto"
        impact_value = 0
    return impact_value, reason


# ==========================
# 5. Execução interativa (internacional)
# ==========================
if __name__ == "__main__":
    print("🚀 Carregando modelo...")
    models, scaler, champion_impacts, league_stats, feature_cols = load_components()
    print("✅ Modelo carregado com sucesso!\n")

    while True:
        print("=== Novo jogo (Internacional) ===")
        league_t1 = input("Liga do Time 1 (ex: LCK, LPL, LEC): ").strip().upper()
        league_t2 = input("Liga do Time 2 (ex: LCK, LPL, LEC): ").strip().upper()
        top_t1 = input("Top Time 1: ")
        jung_t1 = input("Jungler Time 1: ")
        mid_t1 = input("Mid Time 1: ")
        adc_t1 = input("ADC Time 1: ")
        sup_t1 = input("Support Time 1: ")
        top_t2 = input("Top Time 2: ")
        jung_t2 = input("Jungler Time 2: ")
        mid_t2 = input("Mid Time 2: ")
        adc_t2 = input("ADC Time 2: ")
        sup_t2 = input("Support Time 2: ")
        threshold = float(input("Threshold (ex: 0.55, 0.65, 0.75): "))

        league_base = league_t1  # ✅ compatível com o modelo

        def fmt(val): return f"{val:+.2f}"

        print("\n--- IMPACTO DOS CAMPEÕES ---")
        imp_t1, imp_t2 = [], []
        missing_count = 0  # contagem de campeões sem dados

        roles = ["Top", "Jungler", "Mid", "ADC", "Support"]
        champs_t1 = [top_t1, jung_t1, mid_t1, adc_t1, sup_t1]
        champs_t2 = [top_t2, jung_t2, mid_t2, adc_t2, sup_t2]

        for role, champ in zip(roles, champs_t1):
            impact, reason = get_champion_impact(champion_impacts, league_t1, champ)
            imp_t1.append(impact)
            info = f"{role.upper()} Time 1: {champ} ({fmt(impact)})"
            if reason:
                info += f" ← {reason}"
                if "⚠️" in reason or "🔴" in reason:
                    missing_count += 1
            print(info)

        print()

        for role, champ in zip(roles, champs_t2):
            impact, reason = get_champion_impact(champion_impacts, league_t2, champ)
            imp_t2.append(impact)
            info = f"{role.upper()} Time 2: {champ} ({fmt(impact)})"
            if reason:
                info += f" ← {reason}"
                if "⚠️" in reason or "🔴" in reason:
                    missing_count += 1
            print(info)

        if missing_count > 0:
            print(f"\n⚠️ Aviso: {missing_count} campeões sem dados suficientes ou com nome incorreto.\n")

        # Impacto total
        total_t1 = np.sum(imp_t1)
        total_t2 = np.sum(imp_t2)

        # Pega as médias de cada liga
        liga_stats_val_t1 = league_stats.get(league_t1, 28.0)
        liga_stats_val_t2 = league_stats.get(league_t2, 28.0)

        if isinstance(liga_stats_val_t1, dict):
            base_t1 = liga_stats_val_t1.get("mean_kills", 28.0)
        else:
            base_t1 = float(liga_stats_val_t1)

        if isinstance(liga_stats_val_t2, dict):
            base_t2 = liga_stats_val_t2.get("mean_kills", 28.0)
        else:
            base_t2 = float(liga_stats_val_t2)

        # Média combinada e kills estimadas
        base = (base_t1 + base_t2) / 2
        estimated_kills = base + (total_t1 + total_t2) / 2

        print(f"\n⚖️ Impacto total: Time 1 = {total_t1:+.2f} | Time 2 = {total_t2:+.2f}")
        print(f"🎯 Kills estimados: {estimated_kills:.2f}")

        # Predição detalhada
        predictions = predict_game(
            {
                'league': league_base,
                'top_t1': top_t1,
                'jung_t1': jung_t1,
                'mid_t1': mid_t1,
                'adc_t1': adc_t1,
                'sup_t1': sup_t1,
                'top_t2': top_t2,
                'jung_t2': jung_t2,
                'mid_t2': mid_t2,
                'adc_t2': adc_t2,
                'sup_t2': sup_t2
            },
            models,
            scaler,
            champion_impacts,
            league_stats,
            feature_cols,
            threshold=threshold
        )

        print("\n--- RESULTADOS COMPLETOS ---")
        for line, pred in predictions.items():
            prob = pred["probability_under"] * 100
            conf = pred["confidence"]
            bet = "UNDER" if pred["bet_under"] else "OVER"
            print(f"Linha {line:>5}: {bet:<5} | Prob(UNDER): {prob:6.1f}% | Confiança: {conf}")

        print("\n---------------------------\n")

        again = input("Analisar outro jogo? (s/n): ").lower()
        if again != 's':
            break

    print("✅ Encerrado.")
