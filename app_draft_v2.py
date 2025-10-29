import streamlit as st
import pickle
import numpy as np
import os

# =============================
# CONFIGURAÇÃO INICIAL
# =============================
st.set_page_config(page_title="LoL Draft Analyzer v2", page_icon="🎮", layout="wide")

st.title("🎮 LoL Draft Analyzer v2")
st.markdown("Analise o impacto dos campeões e preveja resultados UNDER/OVER de partidas de League of Legends.")

# =============================
# FUNÇÕES DE SUPORTE
# =============================
@st.cache_resource
def load_components():
    base_path = "lol_under_over_model"

    def load_pkl(name):
        with open(os.path.join(base_path, f"{name}_v2.pkl"), "rb") as f:
            return pickle.load(f)

    models = load_pkl("trained_models")
    scaler = load_pkl("scaler")
    champion_impacts = load_pkl("champion_impacts")
    league_stats = load_pkl("league_stats")
    feature_cols = load_pkl("feature_columns")
    return models, scaler, champion_impacts, league_stats, feature_cols


def obter_impacto(champion_name, league, champion_impacts):
    impact = champion_impacts.get(league, {}).get(champion_name)
    if impact is None:
        return 0.0, True
    return float(impact), False


def calcular_kills_estimados(base, impacto_blue, impacto_red):
    diff = impacto_blue - impacto_red
    return max(5, round(base + diff, 2))


def classificar_confiança(prob):
    if prob >= 0.75:
        return "High"
    elif prob >= 0.6:
        return "Medium"
    else:
        return "Low"


# =============================
# CARREGAR MODELOS
# =============================
st.write("🚀 Carregando modelo...")
models, scaler, champion_impacts, league_stats, feature_cols = load_components()
st.success("✅ Modelo carregado com sucesso!")

# =============================
# ESTADO INICIAL
# =============================
if "blue_picks" not in st.session_state:
    st.session_state.blue_picks = []
if "red_picks" not in st.session_state:
    st.session_state.red_picks = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# =============================
# SELEÇÃO DE LIGA
# =============================
selected_league = st.selectbox("Selecione a liga:", list(league_stats.keys()), index=0)

# =============================
# INPUT DE CAMPEÕES
# =============================
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Blue Side Picks")
    champ = st.text_input("Digite o campeão (Blue):", key=f"champ_input_{st.session_state.input_key}")

    if champ:
        champ_name = champ.strip()
        impact, warn = obter_impacto(champ_name, selected_league, champion_impacts)
        st.session_state.blue_picks.append((champ_name, impact, warn))

        msg = f"🔵 Blue Side: {champ_name} ({impact:+.2f})"
        if warn:
            msg += " ← ⚠️ sem dados suficientes ou nome incorreto"
        st.success(msg)

        st.session_state.input_key += 1
        st.rerun()

    for i, (champ, impact, warn) in enumerate(st.session_state.blue_picks, start=1):
        st.write(f"**Blue Pick {i}:** {champ} ({impact:+.2f})" + (" ⚠️" if warn else ""))

with col2:
    st.subheader("Red Side Picks")
    champ_red = st.text_input("Digite o campeão (Red):", key=f"champ_input_red_{st.session_state.input_key}")

    if champ_red:
        champ_name = champ_red.strip()
        impact, warn = obter_impacto(champ_name, selected_league, champion_impacts)
        st.session_state.red_picks.append((champ_name, impact, warn))

        msg = f"🔴 Red Side: {champ_name} ({impact:+.2f})"
        if warn:
            msg += " ← ⚠️ sem dados suficientes ou nome incorreto"
        st.success(msg)

        st.session_state.input_key += 1
        st.rerun()

    for i, (champ, impact, warn) in enumerate(st.session_state.red_picks, start=1):
        st.write(f"**Red Pick {i}:** {champ} ({impact:+.2f})" + (" ⚠️" if warn else ""))

# =============================
# RESULTADO FINAL (quando 10 picks)
# =============================
if len(st.session_state.blue_picks) == 5 and len(st.session_state.red_picks) == 5:
    st.divider()
    st.success("✅ Draft completo!")

    total_blue = sum(x[1] for x in st.session_state.blue_picks)
    total_red = sum(x[1] for x in st.session_state.red_picks)

    # 🔧 Compatibilidade com diferentes formatos do arquivo league_stats
    league_info = league_stats.get(selected_league, {})
    if isinstance(league_info, dict):
        base = league_info.get("mean_kills", 28.0)
    else:
        base = float(league_info) if isinstance(league_info, (int, float)) else 28.0

    kills_estimados = calcular_kills_estimados(base, total_blue, total_red)

    st.markdown(
        f"""
        ⚖️ **Impacto total:** Blue = {total_blue:+.2f} | Red = {total_red:+.2f}  
        📊 **Média base da liga {selected_league}:** {base:.2f}  
        🎯 **Kills estimados:** {kills_estimados:.2f}
        """
    )

    # Simula probabilidades baseadas em kills estimados
    linhas = np.arange(25.5, 33.0, 1.0)
    st.markdown("---\n**--- RESULTADOS COMPLETOS ---**")
    for linha in linhas:
        prob_under = max(0.05, min(0.95, 1 - abs(kills_estimados - linha) / 10))
        conf = classificar_confiança(prob_under)
        escolha = "UNDER" if prob_under > 0.5 else "OVER"
        st.write(f"Linha {linha:5.1f}: {escolha:5} | Prob(UNDER): {prob_under*100:5.1f}% | Confiança: {conf}")
