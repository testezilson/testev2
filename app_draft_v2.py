import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ===========================
# Configuração da página
# ===========================
st.set_page_config(page_title="LoL Draft Analyzer v2", page_icon="🎮", layout="wide")

# ===========================
# Funções utilitárias
# ===========================
@st.cache_data
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


def calcular_kills_estimados(base_kills, impacto_blue, impacto_red):
    diff = impacto_blue + impacto_red
    return round(base_kills + diff, 2)


def obter_impacto(champion, league, champion_impacts):
    try:
        valor = champion_impacts.get(league, {}).get(champion)
        if valor is None:
            return 0.0, True
        return round(valor, 2), False
    except Exception:
        return 0.0, True


def calcular_conf(prob):
    if prob >= 0.7:
        return "High"
    elif prob >= 0.55:
        return "Medium"
    else:
        return "Low"


# ===========================
# Interface principal
# ===========================
st.title("🎮 LoL Draft Analyzer v2")

models, scaler, champion_impacts, league_stats, feature_cols = load_components()

selected_league = st.selectbox("Selecione a Liga:", list(champion_impacts.keys()))

st.write("Digite o nome dos campeões conforme forem sendo escolhidos (ENTER para confirmar).")

blue_picks, red_picks = [], []
current_side = "Blue"

# Inicializa session_state
if "all_picks" not in st.session_state:
    st.session_state["all_picks"] = []

# Entrada incremental
champ = st.text_input("Digite o campeão:", key="champ_input", on_change=lambda: st.session_state.setdefault("trigger_pick", True))

if st.session_state.get("trigger_pick"):
    champ_name = st.session_state.champ_input.strip()

    if champ_name:
        st.session_state.all_picks.append(champ_name)

        # Alternância de sides
        if len(st.session_state.all_picks) % 2 == 1:
            current_side = "Blue"
        else:
            current_side = "Red"

        impact, warn = obter_impacto(champ_name, selected_league, champion_impacts)

        # Mostra impacto instantâneo
        side_symbol = "🔵" if current_side == "Blue" else "🔴"
        msg = f"{side_symbol} {current_side} Side: {champ_name} ({impact:+.2f})"
        if warn:
            msg += " ← ⚠️ sem dados suficientes ou nome incorreto"
        st.success(msg)

    # limpa estado de input e trigger
    st.session_state.champ_input = ""
    st.session_state.trigger_pick = False


# Quando completar 10 picks
if len(st.session_state.all_picks) == 10:
    st.write("✅ Draft completo!")

    # Divide sides
    blue = st.session_state.all_picks[:5]
    red = st.session_state.all_picks[5:]

    impacto_blue = sum(obter_impacto(c, selected_league, champion_impacts)[0] for c in blue)
    impacto_red = sum(obter_impacto(c, selected_league, champion_impacts)[0] for c in red)

    # ======== Média base da liga (robusto) ========
    if isinstance(league_stats, dict):
        if selected_league in league_stats:
            stats = league_stats[selected_league]
            if isinstance(stats, dict):
                base = stats.get("mean_kills", 28.0)
            else:
                base = float(stats)
        else:
            base = 28.0
    else:
        base = 28.0

    kills_estimados = calcular_kills_estimados(base, impacto_blue, impacto_red)

    st.markdown(f"⚖️ **Impacto total:** Blue = {impacto_blue:+.2f} | Red = {impacto_red:+.2f}")
    st.markdown(f"📊 **Média base da liga {selected_league}:** {base:.2f}")
    st.markdown(f"🎯 **Kills estimados:** {kills_estimados:.2f}")

    st.markdown("---")
    st.markdown("### RESULTADOS COMPLETOS")

    linhas = [25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5]
    for linha in linhas:
        prob_under = np.clip(1 / (1 + np.exp((kills_estimados - linha) / 2)), 0, 1)
        conf = calcular_conf(abs(prob_under - 0.5) * 2)
        bet = "UNDER" if prob_under > 0.5 else "OVER"
        st.write(f"Linha {linha:5.1f}: {bet:5s} | Prob(UNDER): {prob_under*100:5.1f}% | Confiança: {conf}")

    # Reset
    st.session_state.all_picks = []
