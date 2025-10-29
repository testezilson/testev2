import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# --- Import seguro para diferentes versões do scikit-learn ---
try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:
    class InconsistentVersionWarning(Warning):
        pass

# --- Configuração da página ---
st.set_page_config(
    page_title="LoL Draft Analyzer v2",
    page_icon="🎮",
    layout="wide"
)

# --- Função para carregar os componentes do modelo ---
@st.cache_resource
def load_components():
    base_path = "lol_under_over_model"
    with open(os.path.join(base_path, "trained_models_v2.pkl"), "rb") as f:
        models = pickle.load(f)
    with open(os.path.join(base_path, "scaler_v2.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base_path, "champion_impacts_v2.pkl"), "rb") as f:
        champion_impacts = pickle.load(f)
    with open(os.path.join(base_path, "league_stats_v2.pkl"), "rb") as f:
        league_stats = pickle.load(f)
    with open(os.path.join(base_path, "feature_columns_v2.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    return models, scaler, champion_impacts, league_stats, feature_cols


# --- Interface principal ---
st.title("🎮 LoL Draft Analyzer v2")
st.markdown("Analise o impacto dos campeões e preveja resultados UNDER/OVER de partidas de League of Legends.")

models, scaler, champion_impacts, league_stats, feature_cols = load_components()

# --- Inputs de seleção ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Blue Side Picks")
    blue_picks = []
    for i in range(5):
        champ = st.text_input(f"Blue Pick {i+1}", key=f"blue_{i}")
        if champ:
            blue_picks.append(champ.strip())

with col2:
    st.subheader("Red Side Picks")
    red_picks = []
    for i in range(5):
        champ = st.text_input(f"Red Pick {i+1}", key=f"red_{i}")
        if champ:
            red_picks.append(champ.strip())

# --- Função auxiliar ---
def get_champ_impact(league, champion):
    value = champion_impacts.get(league, {}).get(champion)
    if value is None:
        return 0.0, f"• {champion} (+0.00) ← ⚠️ sem dados suficientes ou nome incorreto"
    else:
        sign = "+" if value >= 0 else ""
        return value, f"• {champion} ({sign}{round(value, 2)})"

# --- Processar Draft ---
if len(blue_picks) == 5 and len(red_picks) == 5:
    st.success("✅ Draft completo!")

    st.subheader("--- IMPACTO DOS CAMPEÕES ---")

    st.write("**Blue Side:**")
    blue_total = 0
    for champ in blue_picks:
        imp, msg = get_champ_impact("LCK", champ)
        st.write(msg)
        blue_total += imp

    st.write("\n**Red Side:**")
    red_total = 0
    for champ in red_picks:
        imp, msg = get_champ_impact("LCK", champ)
        st.write(msg)
        red_total += imp

    base = league_stats.get("LCK", {}).get("mean_kills", 28.0)
    kills_est = round(base + (blue_total + red_total) / 2, 2)

    st.markdown(f"⚖️ **Impacto total:** Blue = {blue_total:+.2f} | Red = {red_total:+.2f}")
    st.markdown(f"📊 **Média base da liga LCK:** {base}")
    st.markdown(f"🎯 **Kills estimados:** {kills_est}")

    # --- Resultados completos (mock) ---
    st.subheader("--- RESULTADOS COMPLETOS ---")
    lines = [25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5]
    for line in lines:
        prob_under = np.clip(np.random.normal(0.5, 0.15), 0, 1)
        confidence = "High" if prob_under > 0.7 or prob_under < 0.3 else "Medium"
        result = "UNDER" if prob_under > 0.5 else "OVER"
        st.write(f"Linha {line:>5}: {result:5} | Prob(UNDER): {prob_under*100:5.1f}% | Confiança: {confidence}")

else:
    st.info("🧩 Insira todos os 10 campeões (5 de cada lado) para ver os resultados.")
