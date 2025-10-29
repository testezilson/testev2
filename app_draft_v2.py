import streamlit as st
import pickle
import numpy as np
from lol_under_over_model.load_and_predict_v2 import predict_game

# -----------------------------
# CONFIGURAÇÕES DA PÁGINA
# -----------------------------
st.set_page_config(page_title="LoL Draft Analyzer v2", page_icon="🎮", layout="wide")

# -----------------------------
# FUNÇÕES AUXILIARES
# -----------------------------
@st.cache_resource
def load_components():
    with open("lol_under_over_model/trained_models_v2.pkl", "rb") as f:
        models = pickle.load(f)
    with open("lol_under_over_model/scaler_v2.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("lol_under_over_model/champion_impacts_v2.pkl", "rb") as f:
        champion_impacts = pickle.load(f)
    with open("lol_under_over_model/league_stats_v2.pkl", "rb") as f:
        league_stats = pickle.load(f)
    with open("lol_under_over_model/feature_columns_v2.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return models, scaler, champion_impacts, league_stats, feature_cols

models, scaler, champion_impacts, league_stats, feature_cols = load_components()

def get_champ_impact(champion, league):
    try:
        impact = champion_impacts.get(league, {}).get(champion, 0.0)
        if impact == 0.0:
            return f"{champion} (+0.00) ← ⚠️ sem dados suficientes ou nome incorreto", 0.0
        sign = "+" if impact >= 0 else "-"
        return f"{champion} ({sign}{abs(impact):.2f})", impact
    except Exception:
        return f"{champion} (+0.00) ← ⚠️ erro ao calcular", 0.0

# -----------------------------
# INTERFACE STREAMLIT
# -----------------------------
st.title("🎮 LoL Draft Analyzer v2")
st.caption("Analise o impacto dos campeões e preveja resultados UNDER/OVER de partidas de League of Legends.")

selected_league = st.selectbox("Selecione a liga", ["LCK", "LPL", "LEC", "LCS", "CBLOL", "PCS", "VCS", "LLA", "LJL", "TCL"])

col1, col2 = st.columns(2)
blue_picks, red_picks = [], []

with col1:
    st.subheader("Blue Side Picks")
    for i in range(5):
        champ = st.text_input(f"Blue Pick {i+1}", key=f"blue{i}")
        if champ:
            blue_picks.append(champ)
            texto, impacto = get_champ_impact(champ, selected_league)
            st.write(f"🧩 {texto}")
            soma_blue = sum(champion_impacts.get(selected_league, {}).get(c, 0.0) for c in blue_picks)
            st.write(f"⚖️ Impacto parcial Blue: {soma_blue:+.2f}")

with col2:
    st.subheader("Red Side Picks")
    for i in range(5):
        champ = st.text_input(f"Red Pick {i+1}", key=f"red{i}")
        if champ:
            red_picks.append(champ)
            texto, impacto = get_champ_impact(champ, selected_league)
            st.write(f"🧩 {texto}")
            soma_red = sum(champion_impacts.get(selected_league, {}).get(c, 0.0) for c in red_picks)
            st.write(f"⚖️ Impacto parcial Red: {soma_red:+.2f}")

# -----------------------------
# RESULTADO FINAL
# -----------------------------
if len(blue_picks) == 5 and len(red_picks) == 5:
    st.success("✅ Draft completo!")

    game_data = {
        "league": selected_league,
        "top_t1": blue_picks[0],
        "jung_t1": blue_picks[1],
        "mid_t1": blue_picks[2],
        "adc_t1": blue_picks[3],
        "sup_t1": blue_picks[4],
        "top_t2": red_picks[0],
        "jung_t2": red_picks[1],
        "mid_t2": red_picks[2],
        "adc_t2": red_picks[3],
        "sup_t2": red_picks[4],
    }

    predictions = predict_game(
        game_data, models, scaler, champion_impacts, league_stats, feature_cols, threshold=0.55
    )

    base = 28.0
    if isinstance(league_stats, dict):
        base = league_stats.get(selected_league, {}).get("mean_kills", 28.0)

    blue_impact = sum(champion_impacts.get(selected_league, {}).get(c, 0.0) for c in blue_picks)
    red_impact = sum(champion_impacts.get(selected_league, {}).get(c, 0.0) for c in red_picks)
    kills_estimados = base + (blue_impact + red_impact) / 2

    st.markdown("---")
    st.markdown(f"⚖️ **Impacto total:** Blue = {blue_impact:+.2f} | Red = {red_impact:+.2f}")
    st.markdown(f"🎯 **Kills estimados:** {kills_estimados:.2f}")

    st.markdown("### --- RESULTADOS COMPLETOS ---")
    for linha, pred in predictions.items():
        prob = pred["probability_under"] * 100
        conf = pred["confidence"]
        result = "UNDER" if pred["bet_under"] else "OVER"
        st.write(f"Linha {linha:5}: {result:5} | Prob(UNDER): {prob:5.1f}% | Confiança: {conf}")
