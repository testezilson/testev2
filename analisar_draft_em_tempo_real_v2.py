import pickle
import numpy as np
from lol_under_over_model.load_and_predict_v2 import predict_game as predict_core

# ==========================
# 1) Carrega componentes
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
# 2) Impacto dos campeões
# ==========================
def get_champion_impact(champion_impacts, league, champion_name):
    impacts_liga = champion_impacts.get(league, {})
    if champion_name in impacts_liga:
        impact_value = impacts_liga[champion_name]
        reason = None if impact_value != 0 else "🟡 impacto neutro (≈ 0)"
    else:
        impact_value = 0
        reason = "⚠️ sem dados suficientes ou nome incorreto"
    return impact_value, reason


# ==========================
# 3) Funções auxiliares
# ==========================
def fmt(v):
    return f"{v:+.2f}"

def sum_side_impact(champs, champion_impacts, league):
    return sum(get_champion_impact(champion_impacts, league, c)[0] for c in champs)

def base_kills_for_league(league_stats, league):
    v = league_stats.get(league, 28.0)
    return v.get("mean_kills", 28.0) if isinstance(v, dict) else float(v)

def show_partial(league, champion_impacts, champs_blue, champs_red, last_side):
    total_blue = sum_side_impact(champs_blue, champion_impacts, league)
    total_red = sum_side_impact(champs_red, champion_impacts, league)
    if last_side == "Blue":
        print(f"➡️ Impacto {last_side} Side: {fmt(total_blue)} | Diferença (Blue-Red): {fmt(total_blue - total_red)}\n")
    else:
        print(f"➡️ Impacto {last_side} Side: {fmt(total_red)} | Diferença (Blue-Red): {fmt(total_blue - total_red)}\n")


# ==========================
# 4) Execução principal
# ==========================
if __name__ == "__main__":
    print("🚀 Carregando modelo...")
    models, scaler, champion_impacts, league_stats, feature_cols = load_components()
    print("✅ Modelo carregado com sucesso!\n")

    print("=== Novo Draft (Tempo Real) ===")
    league = input("Liga (ex: LCK, LPL, LEC, CBLOL): ").strip().upper()
    threshold = float(input("Threshold (ex: 0.55, 0.65, 0.75): "))

    champs_blue, champs_red = [], []

    print("\nEscolha os campeões conforme o draft oficial (sem roles):\n")

    # Ordem oficial
    champs_blue.append(input("1️⃣ Blue Pick 1: ").strip())
    show_partial(league, champion_impacts, champs_blue, champs_red, "Blue")

    champs_red.append(input("2️⃣ Red Pick 2: ").strip())
    show_partial(league, champion_impacts, champs_blue, champs_red, "Red")

    champs_red.append(input("3️⃣ Red Pick 3: ").strip())
    show_partial(league, champion_impacts, champs_blue, champs_red, "Red")

    champs_blue.append(input("4️⃣ Blue Pick 4: ").strip())
    show_partial(league, champion_impacts, champs_blue, champs_red, "Blue")

    champs_blue.append(input("5️⃣ Blue Pick 5: ").strip())
    show_partial(league, champion_impacts, champs_blue, champs_red, "Blue")

    champs_red.append(input("6️⃣ Red Pick 6: ").strip())
    show_partial(league, champion_impacts, champs_blue, champs_red, "Red")

    champs_red.append(input("7️⃣ Red Pick 7: ").strip())
    show_partial(league, champion_impacts, champs_blue, champs_red, "Red")

    champs_blue.append(input("8️⃣ Blue Pick 8: ").strip())
    show_partial(league, champion_impacts, champs_blue, champs_red, "Blue")

    champs_blue.append(input("9️⃣ Blue Pick 9: ").strip())
    show_partial(league, champion_impacts, champs_blue, champs_red, "Blue")

    champs_red.append(input("🔟 Red Pick 10: ").strip())
    show_partial(league, champion_impacts, champs_blue, champs_red, "Red")

    # ==========================
    # Draft completo
    # ==========================
    print("\n✅ Draft completo!")

    print("\n--- IMPACTO DOS CAMPEÕES ---")
    for i, champ in enumerate(champs_blue, 1):
        imp, reason = get_champion_impact(champion_impacts, league, champ)
        msg = f"Blue Side Pick {i}: {champ} ({fmt(imp)})"
        if reason:
            msg += f" ← {reason}"
        print(msg)

    print()
    for i, champ in enumerate(champs_red, 1):
        imp, reason = get_champion_impact(champion_impacts, league, champ)
        msg = f"Red Side Pick {i}: {champ} ({fmt(imp)})"
        if reason:
            msg += f" ← {reason}"
        print(msg)

    total_blue = sum_side_impact(champs_blue, champion_impacts, league)
    total_red = sum_side_impact(champs_red, champion_impacts, league)
    base = base_kills_for_league(league_stats, league)
    estimated_kills = base + (total_blue + total_red) / 2

    print(f"\n⚖️ Impacto total: Blue Side = {fmt(total_blue)} | Red Side = {fmt(total_red)}")
    print(f"🎯 Kills estimados: {estimated_kills:.2f}")

    # Predição final (mantém a estrutura padrão)
    if len(champs_blue) == 5 and len(champs_red) == 5:
        game_data = {
            'league': league,
            'top_t1': champs_blue[0],
            'jung_t1': champs_blue[1],
            'mid_t1': champs_blue[2],
            'adc_t1': champs_blue[3],
            'sup_t1': champs_blue[4],
            'top_t2': champs_red[0],
            'jung_t2': champs_red[1],
            'mid_t2': champs_red[2],
            'adc_t2': champs_red[3],
            'sup_t2': champs_red[4]
        }

        predictions = predict_core(game_data, models, scaler, champion_impacts, league_stats, feature_cols, threshold)

        print("\n--- RESULTADOS COMPLETOS ---")
        for line, pred in predictions.items():
            prob = pred["probability_under"] * 100
            conf = pred["confidence"]
            bet = "UNDER" if pred["bet_under"] else "OVER"
            print(f"Linha {line:>5}: {bet:<5} | Prob(UNDER): {prob:6.1f}% | Confiança: {conf}")
    else:
        print("\n⚠️ Draft incompleto (faltando picks para um dos lados).")

    print("\n🏁 Fim da análise do draft (tempo real).")
