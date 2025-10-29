#!/usr/bin/env python3
"""
LoL Esports Betting App - Versão Refatorada
Integração completa com modelo UNDER/OVER e sistema de debug robusto
"""

import streamlit as st
import requests
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configuração da página
st.set_page_config(
    page_title="LoL Esports Betting Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Configuração de logging para debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSS simplificado
st.markdown(
    """
<style>
    /* Tema escuro */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Botões */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Títulos */
    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 24px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Containers */
    .stContainer {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class DebugLogger:
    """Sistema de debug para inconsistências de nomes"""

    def __init__(self):
        self.csv_filename = "debug_inconsistencies.csv"
        self.inconsistencies = {"leagues": set(), "teams": set(), "champions": set()}
        self.debug_file = "debug_inconsistencies.json"
        self._initialize_csv()
        self.load_existing_debug()

    def _initialize_csv(self):
        """Inicializa arquivo CSV com cabeçalhos"""
        try:
            import csv

            if not os.path.exists(self.csv_filename):
                with open(self.csv_filename, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "category", "name", "context"])
        except Exception as e:
            logger.error(f"Erro ao inicializar CSV: {e}")

    def load_existing_debug(self):
        """Carrega inconsistências já registradas"""
        try:
            if os.path.exists(self.debug_file):
                with open(self.debug_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key in self.inconsistencies:
                        self.inconsistencies[key] = set(data.get(key, []))
        except Exception as e:
            logger.error(f"Erro ao carregar debug: {e}")

    def log_inconsistency(self, category: str, value: str, context: str = ""):
        """Registra uma inconsistência"""
        if category in self.inconsistencies:
            full_entry = f"{value} ({context})" if context else value

            # Evitar duplicatas
            if full_entry not in self.inconsistencies[category]:
                self.inconsistencies[category].add(full_entry)
                self.save_debug()
                self._save_to_csv(category, value, context)
                logger.warning(f"Inconsistência {category}: {value} - {context}")

    def _save_to_csv(self, category: str, name: str, context: str):
        """Salva uma linha no CSV"""
        try:
            import csv

            with open(self.csv_filename, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().isoformat(), category, name, context])
        except Exception as e:
            logger.error(f"Erro ao salvar no CSV: {e}")

    def save_debug(self):
        """Salva inconsistências no arquivo JSON (backup)"""
        try:
            data = {key: list(values) for key, values in self.inconsistencies.items()}
            with open(self.debug_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao salvar debug JSON: {e}")

    def get_summary(self) -> Dict[str, int]:
        """Retorna resumo das inconsistências"""
        return {key: len(values) for key, values in self.inconsistencies.items()}

    def get_csv_stats(self) -> Dict[str, int]:
        """Retorna estatísticas do arquivo CSV"""
        try:
            import csv

            if not os.path.exists(self.csv_filename):
                return {"total": 0, "leagues": 0, "teams": 0, "champions": 0}

            stats = {"total": 0, "leagues": 0, "teams": 0, "champions": 0}

            with open(self.csv_filename, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats["total"] += 1
                    category = row.get("category", "")
                    if category in stats:
                        stats[category] += 1

            return stats
        except Exception as e:
            logger.error(f"Erro ao ler estatísticas do CSV: {e}")
            return {"total": 0, "leagues": 0, "teams": 0, "champions": 0}


class LoLPredictor:
    """Classe integrada do modelo de predição"""

    def __init__(self):
        self.models = None
        self.scaler = None
        self.champion_impacts = None
        self.league_stats = None
        self.feature_cols = None
        self.winrates = {
            "BOA": {"UNDER": 58.6, "OVER": 61.0},
            "MUITO_BOA": {"UNDER": 67.4, "OVER": 65.7},
            "EXCELENTE": {"UNDER": 73.0, "OVER": 65.4},
        }
        self.debug_logger = DebugLogger()
        self._load_model()

    def _load_model(self):
        """Carrega todos os componentes do modelo"""
        try:
            # Detectar caminho dos arquivos
            if os.path.exists("lol_under_over_model/trained_models.pkl"):
                base_path = "lol_under_over_model/"
            elif os.path.exists("trained_models.pkl"):
                base_path = ""
            else:
                raise FileNotFoundError("Arquivos do modelo não encontrados")

            # Carregar componentes
            with open(f"{base_path}trained_models.pkl", "rb") as f:
                self.models = pickle.load(f)
            with open(f"{base_path}scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(f"{base_path}champion_impacts.pkl", "rb") as f:
                self.champion_impacts = pickle.load(f)
            with open(f"{base_path}league_stats.pkl", "rb") as f:
                self.league_stats = pickle.load(f)
            with open(f"{base_path}feature_columns.pkl", "rb") as f:
                self.feature_cols = pickle.load(f)

            logger.info("Modelo carregado com sucesso!")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            st.error(f"Erro ao carregar modelo: {e}")

    def normalize_league_name(self, league_name: str) -> str:
        """Normaliza nomes de ligas"""
        league_mapping = {
            # Mapeamentos conhecidos
            "LPL": "LPL",
            "LCK": "LCK",
            "LEC": "LEC",
            "LCS": "LCS",
            "CBLOL": "CBLOL",
            "PCS": "PCS",
            "VCS": "VCS",
            "LJL": "LJL",
            "LLA": "LLA",
            "TCL": "TCL",
            "LCO": "LCO",
            "EMEA Masters": "EMEA",
            "European Masters": "EMEA",
            "EM": "EMEA",
            "MSI": "MSI",
            "Worlds": "Worlds",
            "World Championship": "Worlds",
        }

        # Busca direta
        if league_name in league_mapping:
            return league_mapping[league_name]

        # Busca por substring
        for key, value in league_mapping.items():
            if key.lower() in league_name.lower():
                return value

        # Log inconsistência
        self.debug_logger.log_inconsistency("leagues", league_name, "Nome não mapeado")
        return league_name

    def normalize_team_name(self, team_name: str, league: str) -> str:
        """Normaliza nomes de times"""
        team_mappings = {
            "LPL": {
                "WBG": "Weibo Gaming",
                "JDG": "JD Gaming",
                "OMG": "Oh My God",
                "LNG": "LNG Esports",
                "TES": "Top Esports",
                "BLG": "Bilibili Gaming",
                "FPX": "FunPlus Phoenix",
                "IG": "Invictus Gaming",
                "RNG": "Royal Never Give Up",
                "EDG": "Edward Gaming",
                "AL": "Anyone's Legend",
                "UP": "Ultra Prime",
                "WE": "Team WE",
                "LGD": "LGD Gaming",
                "RA": "Rare Atom",
                "TT": "ThunderTalk Gaming",
                "NIP": "Ninjas in Pyjamas",
            },
            "LCK": {
                "T1": "T1",
                "GEN": "Gen.G",
                "GG": "Gen.G",
                "DRX": "DRX",
                "KT": "KT Rolster",
                "LSB": "Liiv SANDBOX",
                "DK": "DWG KIA",
                "DWG": "DWG KIA",
                "HLE": "Hanwha Life Esports",
                "NS": "Nongshim RedForce",
                "NSRF": "Nongshim RedForce",
                "FOX": "FOX",
                "BRO": "BRION",
                "KDF": "Kwangdong Freecs",
            },
            "LEC": {
                "G2": "G2 Esports",
                "FNC": "Fnatic",
                "MAD": "MAD Lions",
                "VIT": "Team Vitality",
                "KC": "Karmine Corp",
                "BDS": "Team BDS",
                "SK": "SK Gaming",
                "GX": "GIANTX",
                "TH": "Team Heretics",
                "FLY": "FlyQuest RED",
            },
            "LCS": {
                "TL": "Team Liquid",
                "C9": "Cloud9",
                "100": "100 Thieves",
                "TSM": "TSM",
                "FLY": "FlyQuest",
                "DIG": "Dignitas",
                "IMT": "Immortals",
                "EG": "Evil Geniuses",
                "GG": "Golden Guardians",
                "CLG": "Counter Logic Gaming",
            },
        }

        # Busca por mapeamento
        if league in team_mappings:
            league_teams = team_mappings[league]
            if team_name in league_teams:
                return league_teams[team_name]

            # Busca parcial
            for code, full_name in league_teams.items():
                if (
                    code.lower() in team_name.lower()
                    or team_name.lower() in full_name.lower()
                ):
                    return full_name

        # Log inconsistência
        self.debug_logger.log_inconsistency(
            "teams", f"{team_name} ({league})", "Time não mapeado"
        )
        return team_name

    def normalize_champion_name(self, champion_name: str) -> str:
        """Normaliza nomes de campeões"""
        champion_mapping = {
            # Mapeamentos conhecidos da API para o modelo
            "JarvanIV": "Jarvan IV",
            "XinZhao": "Xin Zhao",
            "LeeSin": "Lee Sin",
            "TwistedFate": "Twisted Fate",
            "MissFortune": "Miss Fortune",
            "DrMundo": "Dr. Mundo",
            "TahmKench": "Tahm Kench",
            "AurelionSol": "Aurelion Sol",
            "MonkeyKing": "Wukong",
            "KSante": "K'Sante",
            "RenataGlasc": "Renata Glasc",
            "NunuWillump": "Nunu & Willump",
            "BelVeth": "Bel'Veth",
            "RekSai": "Rek'Sai",
            "KhaZix": "Kha'Zix",
            "VelKoz": "Vel'Koz",
            "ChoGath": "Cho'Gath",
            "KaiSa": "Kai'Sa",
            "KogMaw": "Kog'Maw",
            "MasterYi": "Master Yi",
        }

        # Primeiro, tentar mapeamento direto
        if champion_name in champion_mapping:
            normalized = champion_mapping[champion_name]
        else:
            normalized = champion_name

        # Verificar se existe no modelo (buscar em todas as ligas)
        for league_impacts in self.champion_impacts.values():
            if normalized in league_impacts:
                return normalized

        # Se não encontrou, tentar busca case-insensitive
        for league_impacts in self.champion_impacts.values():
            for model_champ in league_impacts.keys():
                if model_champ.lower() == normalized.lower():
                    return model_champ

        # Se ainda não encontrou, tentar busca por substring
        for league_impacts in self.champion_impacts.values():
            for model_champ in league_impacts.keys():
                if (
                    normalized.lower() in model_champ.lower()
                    or model_champ.lower() in normalized.lower()
                ):
                    return model_champ

        # Log inconsistência apenas se realmente não encontrar
        self.debug_logger.log_inconsistency(
            "champions",
            champion_name,
            f"Campeão não encontrado no modelo (tentou: {normalized})",
        )

        return normalized

    def get_team_average(self, team_name: str, league: str) -> float:
        """Busca média do time"""
        normalized_league = self.normalize_league_name(league)
        normalized_team = self.normalize_team_name(team_name, normalized_league)

        if normalized_league in self.league_stats:
            return self.league_stats.get(normalized_league, 28.5)

        return 28.5  # Valor padrão

    def get_champion_impact(self, champion_name: str, league: str = None) -> float:
        """Busca impacto do campeão na liga específica"""
        normalized = self.normalize_champion_name(champion_name)

        # Se uma liga foi especificada, buscar nela primeiro
        if league:
            normalized_league = self.normalize_league_name(league)
            if normalized_league in self.champion_impacts:
                league_impacts = self.champion_impacts[normalized_league]
                if normalized in league_impacts:
                    return league_impacts[normalized]

        # Se não encontrou na liga específica, buscar em todas as ligas
        for league_name, league_impacts in self.champion_impacts.items():
            if normalized in league_impacts:
                return league_impacts[normalized]

        # Se não encontrou em nenhuma liga, retornar 0.0
        return 0.0

    def create_features(
        self,
        blue_team: List[Dict],
        red_team: List[Dict],
        team1_name: str,
        team2_name: str,
        league: str,
    ) -> Dict[str, float]:
        """Cria features para o modelo"""
        normalized_league = self.normalize_league_name(league)
        league_avg = self.league_stats.get(normalized_league, 28.5)

        positions = ["top", "jung", "mid", "adc", "sup"]

        # Impactos dos times
        team1_impacts = []
        team2_impacts = []

        for i, player in enumerate(blue_team[:5]):
            champion = player.get("championId", "")
            impact = self.get_champion_impact(champion, league)
            team1_impacts.append(impact)

        for i, player in enumerate(red_team[:5]):
            champion = player.get("championId", "")
            impact = self.get_champion_impact(champion, league)
            team2_impacts.append(impact)

        # Criar features
        features = {
            "league_encoded": hash(normalized_league) % 100,
            "mean_league_kills": league_avg,
            "std_league_kills": 7.99,
            "mean_impact_team1": np.mean(team1_impacts),
            "mean_impact_team2": np.mean(team2_impacts),
            "total_impact": np.mean(team1_impacts) + np.mean(team2_impacts),
            "impact_diff": np.mean(team1_impacts) - np.mean(team2_impacts),
        }

        # Adicionar impactos individuais
        for i, impact in enumerate(team1_impacts):
            features[f"impact_t1_pos{i + 1}"] = impact

        for i, impact in enumerate(team2_impacts):
            features[f"impact_t2_pos{i + 1}"] = impact

        return features

    def predict(
        self,
        blue_team: List[Dict],
        red_team: List[Dict],
        team1_name: str,
        team2_name: str,
        league: str,
        bet_line: float,
    ) -> Dict[str, Any]:
        """Faz predição completa"""
        if not self.models:
            return {"error": "Modelo não carregado"}

        # Criar features
        features = self.create_features(
            blue_team, red_team, team1_name, team2_name, league
        )

        # Processar features
        features_df = pd.DataFrame([features])[self.feature_cols]
        features_scaled = self.scaler.transform(features_df)

        # Encontrar modelo para a linha
        if bet_line not in self.models:
            bet_line = min(self.models.keys(), key=lambda x: abs(x - bet_line))

        # Fazer predição
        prob_under = self.models[bet_line].predict_proba(features_scaled)[0, 1]
        prob_over = 1 - prob_under

        # Classificar apostas
        def classify_probability(prob):
            if prob >= 0.75:
                return "EXCELENTE"
            elif prob >= 0.65:
                return "MUITO_BOA"
            elif prob >= 0.55:
                return "BOA"
            return None

        # Analisar recomendações
        recommendation_under = None
        recommendation_over = None

        if prob_under > 0.55:
            categoria = classify_probability(prob_under)
            if categoria:
                winrate = self.winrates[categoria]["UNDER"]
                roi = (winrate / 100 * 0.83) - ((1 - winrate / 100) * 1.00)
                recommendation_under = {
                    "tipo": "UNDER",
                    "categoria": categoria,
                    "probabilidade": prob_under,
                    "winrate_esperado": winrate,
                    "roi_esperado": roi * 100,
                }

        if prob_over > 0.55:
            categoria = classify_probability(prob_over)
            if categoria:
                winrate = self.winrates[categoria]["OVER"]
                roi = (winrate / 100 * 0.83) - ((1 - winrate / 100) * 1.00)
                recommendation_over = {
                    "tipo": "OVER",
                    "categoria": categoria,
                    "probabilidade": prob_over,
                    "winrate_esperado": winrate,
                    "roi_esperado": roi * 100,
                }

        # Escolher melhor recomendação
        best_recommendation = None
        if recommendation_under and recommendation_over:
            if (
                recommendation_under["roi_esperado"]
                >= recommendation_over["roi_esperado"]
            ):
                best_recommendation = recommendation_under
            else:
                best_recommendation = recommendation_over
        elif recommendation_under:
            best_recommendation = recommendation_under
        elif recommendation_over:
            best_recommendation = recommendation_over

        return {
            "probabilities": {
                "under": round(prob_under, 3),
                "over": round(prob_over, 3),
            },
            "recommendation": best_recommendation,
            "all_options": {"under": recommendation_under, "over": recommendation_over},
            "features": features,
            "debug": self.debug_logger.get_summary(),
        }


# Cache do predictor
@st.cache_resource
def get_predictor():
    """Carrega o predictor com cache"""
    return LoLPredictor()


@st.cache_data(ttl=60)
def get_schedule():
    """Busca agenda de jogos da API"""
    url = "https://esports-api.lolesports.com/persisted/gw/getSchedule?hl=en-US"
    headers = {
        "x-api-key": "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z",
        "User-Agent": "Mozilla/5.0",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("data", {}).get("schedule", {}).get("events", [])

        # Ordenar por prioridade (ao vivo > agendado > finalizado)
        def get_priority(event):
            state = event.get("state", "")
            if state == "inProgress":
                return 0
            elif state == "unstarted":
                return 1
            else:
                return 2

        return sorted(events, key=get_priority)

    except Exception as e:
        logger.error(f"Erro ao buscar agenda: {e}")
        st.error(f"Erro ao buscar dados da API: {e}")
        return []


def get_draft_data(
    match_id: str, game_number: int
) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
    """Busca dados do draft"""
    try:
        if match_id == "N/A" or not str(match_id).isdigit():
            return None, None

        game_id = int(match_id) + game_number
        url = f"https://feed.lolesports.com/livestats/v1/window/{game_id}"

        response = requests.get(url, timeout=5)
        data = response.json()

        game_metadata = data.get("gameMetadata", {})
        blue_team = game_metadata.get("blueTeamMetadata", {}).get(
            "participantMetadata", []
        )
        red_team = game_metadata.get("redTeamMetadata", {}).get(
            "participantMetadata", []
        )

        return blue_team, red_team

    except Exception as e:
        logger.error(f"Erro ao buscar draft: {e}")
        return None, None


def format_datetime(start_time: str) -> str:
    """Formata data/hora para exibição"""
    if not start_time:
        return ""

    try:
        brazil_tz = pytz.timezone("America/Sao_Paulo")
        utc_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        brazil_time = utc_time.astimezone(brazil_tz)

        today = datetime.now(brazil_tz).date()
        game_date = brazil_time.date()

        if game_date == today:
            day_str = "Hoje"
        elif game_date == today + timedelta(days=1):
            day_str = "Amanhã"
        else:
            day_str = brazil_time.strftime("%d/%m")

        time_str = brazil_time.strftime("%H:%M")
        return f"{day_str} • {time_str}"

    except Exception:
        return "Horário indefinido"


def render_game_card(game: Dict[str, Any], index: int):
    """Renderiza card de jogo usando apenas componentes nativos do Streamlit"""
    state = game.get("state", "N/A")
    league = game.get("league", {}).get("name", "N/A")
    match = game.get("match", {})
    match_id = match.get("id", "N/A")
    teams = match.get("teams", [])

    # Nomes dos times
    if len(teams) >= 2:
        team1 = teams[0].get("code", teams[0].get("name", "TBD"))
        team2 = teams[1].get("code", teams[1].get("name", "TBD"))
    else:
        team1, team2 = "TBD", "TBD"

    datetime_str = format_datetime(game.get("startTime"))
    best_of = match.get("strategy", {}).get("count", 1)

    # Status
    if state == "inProgress":
        status = "🔴 AO VIVO"
    elif state == "unstarted":
        status = "📅 AGENDADO"
    else:
        status = "✅ FINALIZADO"

    # Resultado se finalizado
    result_text = ""
    if state == "completed" and len(teams) >= 2:
        team1_wins = teams[0].get("result", {}).get("gameWins", 0)
        team2_wins = teams[1].get("result", {}).get("gameWins", 0)
        result_text = f" ({team1_wins}-{team2_wins})"

    # Container usando apenas Streamlit nativo
    with st.container(border=True):
        # Header com status e BO
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{status}**")
        with col2:
            st.markdown(f"*BO{best_of}*")

        # Times (título principal)
        st.subheader(f"{team1} vs {team2}{result_text}")

        # Liga
        st.markdown(f"**{league}**")

        # Data/hora
        if datetime_str:
            st.caption(f"🕐 {datetime_str}")
        else:
            st.caption("🕐 Horário não definido")

        # Espaçamento
        st.write("")

        # Botão de análise
        if st.button(
            "📊 Analisar Jogo",
            key=f"analyze_{index}",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.current_page = "analysis"
            st.session_state.selected_game = {
                "match_id": match_id,
                "team1": team1,
                "team2": team2,
                "league": league,
                "best_of": best_of,
                "state": state,
                "datetime": datetime_str,
            }
            st.rerun()


def render_home_page():
    """Renderiza página inicial"""
    st.markdown(
        '<h1 class="section-title">🎮 LoL Esports Betting Analysis</h1>',
        unsafe_allow_html=True,
    )

    # Buscar jogos
    games = get_schedule()

    if not games:
        st.warning("Nenhum jogo encontrado ou erro na API")
        return

    # Separar jogos por status
    live_games = [g for g in games if g.get("state") == "inProgress"]
    scheduled_games = [g for g in games if g.get("state") == "unstarted"]
    completed_games = [g for g in games if g.get("state") == "completed"]

    # Jogos ao vivo
    if live_games:
        st.markdown("## 🔴 Jogos Ao Vivo")
        cols = st.columns(min(len(live_games), 2))
        for i, game in enumerate(live_games[:4]):  # Máximo 4 jogos
            with cols[i % 2]:
                render_game_card(game, f"live_{i}")
        st.markdown("---")

    # Próximos jogos
    if scheduled_games:
        st.markdown("## 📅 Próximos Jogos")
        cols = st.columns(2)
        for i, game in enumerate(scheduled_games[:6]):  # Máximo 6 jogos
            with cols[i % 2]:
                render_game_card(game, f"scheduled_{i}")
        st.markdown("---")

    # Jogos finalizados (em expander)
    if completed_games:
        with st.expander("✅ Jogos Finalizados", expanded=False):
            cols = st.columns(2)
            for i, game in enumerate(completed_games[:10]):  # Máximo 10 jogos
                with cols[i % 2]:
                    render_game_card(game, f"completed_{i}")

    st.caption("🕐 Horários exibidos no fuso horário de Brasília (GMT-3)")


def render_analysis_page():
    """Renderiza página de análise"""
    game_data = st.session_state.get("selected_game", {})

    if not game_data:
        st.error("Nenhum jogo selecionado")
        return

    # Header com botão voltar
    col1, col2 = st.columns([1, 6])

    with col1:
        if st.button("← Voltar", type="secondary"):
            st.session_state.current_page = "home"
            st.rerun()

    with col2:
        st.markdown(
            f'<h1 class="section-title">🎮 {game_data["team1"]} vs {game_data["team2"]}</h1>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"{game_data['league']} • {game_data.get('datetime', 'Horário não definido')}"
        )

    # Seleção de mapa
    st.markdown("### Selecione o mapa:")
    map_cols = st.columns(min(game_data["best_of"], 5))
    current_map = st.session_state.get("selected_map", 1)

    for i in range(game_data["best_of"]):
        with map_cols[i]:
            if st.button(
                f"Mapa {i + 1}",
                key=f"map_{i}",
                type="primary" if current_map == i + 1 else "secondary",
            ):
                st.session_state.selected_map = i + 1
                st.rerun()

    st.markdown("---")

    # Buscar draft
    blue_team, red_team = get_draft_data(game_data["match_id"], current_map)

    if blue_team and red_team:
        render_betting_analysis(blue_team, red_team, game_data)
        render_draft_display(blue_team, red_team)
    else:
        st.info(f"📊 Draft do Mapa {current_map} não disponível no momento.")
        st.markdown(
            """
        <div class="custom-alert alert-warning">
            <strong>ℹ️ Informação:</strong> O draft pode não estar disponível para jogos que ainda não começaram 
            ou que já finalizaram há muito tempo.
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_betting_analysis(
    blue_team: List[Dict], red_team: List[Dict], game_data: Dict[str, Any]
):
    """Renderiza análise de apostas"""
    predictor = get_predictor()

    st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="section-title">🤖 Análise de Apostas</h2>', unsafe_allow_html=True
    )

    # Input da linha de aposta
    col1, col2 = st.columns([3, 1])

    with col1:
        bet_line = st.number_input(
            "Linha de Kills da Casa de Apostas:",
            min_value=15.0,
            max_value=50.0,
            value=30.5,
            step=0.5,
            help="Digite a linha oferecida pela casa de apostas",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔮 Analisar", type="primary", use_container_width=True):
            # Fazer predição
            with st.spinner("Analisando draft..."):
                prediction = predictor.predict(
                    blue_team,
                    red_team,
                    game_data["team1"],
                    game_data["team2"],
                    game_data["league"],
                    bet_line,
                )

            st.session_state.prediction_result = prediction
            st.session_state.prediction_line = bet_line

    # Mostrar resultado da predição
    if "prediction_result" in st.session_state:
        pred = st.session_state.prediction_result
        pred_line = st.session_state.prediction_line

        if "error" in pred:
            st.error(f"Erro na predição: {pred['error']}")
            return

        st.markdown("---")
        st.markdown("### 📊 Resultado da Análise")

        # Métricas principais usando componentes nativos
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Probabilidade UNDER",
                value=f"{pred['probabilities']['under']:.1%}",
                delta=None,
            )

        with col2:
            st.metric(
                label="Probabilidade OVER",
                value=f"{pred['probabilities']['over']:.1%}",
                delta=None,
            )

        with col3:
            if pred["recommendation"]:
                rec = pred["recommendation"]
                categoria_display = {
                    "BOA": "🟢 BOA",
                    "MUITO_BOA": "🔵 MUITO BOA",
                    "EXCELENTE": "🟡 EXCELENTE",
                }

                st.metric(
                    label="Recomendação",
                    value=f"{rec['tipo']} {pred_line}",
                    delta=categoria_display.get(rec["categoria"], rec["categoria"]),
                )
            else:
                st.metric(
                    label="Recomendação", value="Não apostar", delta="Baixa confiança"
                )

        # Detalhes da recomendação usando componentes nativos
        if pred["recommendation"]:
            rec = pred["recommendation"]

            st.markdown("### 💡 Detalhes da Recomendação")

            categoria_info = {
                "BOA": {"emoji": "🟢", "desc": "Oportunidade válida com ROI moderado"},
                "MUITO_BOA": {
                    "emoji": "🔵",
                    "desc": "Excelente equilíbrio risco/retorno",
                },
                "EXCELENTE": {
                    "emoji": "🟡",
                    "desc": "Máxima qualidade com ROI premium",
                },
            }

            info = categoria_info.get(
                rec["categoria"], {"emoji": "⚪", "desc": "Categoria desconhecida"}
            )

            # Container com borda
            with st.container(border=True):
                st.subheader(f"{info['emoji']} {rec['categoria']}")
                st.write(info["desc"])

                st.write("")  # Espaçamento

                # Métricas em grid
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="📊 Probabilidade", value=f"{rec['probabilidade']:.1%}"
                    )
                    st.metric(
                        label="💰 ROI Esperado", value=f"{rec['roi_esperado']:+.1f}%"
                    )

                with col2:
                    st.metric(
                        label="🎯 Winrate Esperado",
                        value=f"{rec['winrate_esperado']:.1f}%",
                    )

                    # Confiança
                    confianca = (
                        "ALTA 🔥"
                        if rec["probabilidade"] >= 0.75
                        else "MÉDIA ⚡"
                        if rec["probabilidade"] >= 0.65
                        else "BAIXA 💡"
                    )
                    st.metric(label="🎖️ Confiança", value=confianca)

        # Debug info (se houver inconsistências)
        debug_info = pred.get("debug", {})
        if any(debug_info.values()):
            with st.expander("🔧 Informações de Debug", expanded=False):
                st.markdown(
                    f"""
                <div class="debug-panel">
                    <strong>Inconsistências detectadas:</strong><br>
                    • Ligas: {debug_info.get("leagues", 0)}<br>
                    • Times: {debug_info.get("teams", 0)}<br>
                    • Campeões: {debug_info.get("champions", 0)}<br>
                    <br>
                    <em>Verifique o arquivo debug_inconsistencies.json para detalhes.</em>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)


def render_draft_display(blue_team: List[Dict], red_team: List[Dict]):
    """Renderiza display do draft"""
    predictor = get_predictor()

    # Pegar liga do jogo selecionado
    game_data = st.session_state.get("selected_game", {})
    league = game_data.get("league", "")

    st.markdown("### 📋 Draft dos Times")

    col_blue, col_red = st.columns(2)
    roles = ["TOP", "JUNG", "MID", "ADC", "SUP"]

    with col_blue:
        st.markdown("#### 🔵 BLUE SIDE")
        for i, player in enumerate(blue_team[:5]):
            role = roles[i] if i < len(roles) else "ROLE"
            champion = player.get("championId", "Não selecionado")

            # Buscar impacto na liga específica
            impact = predictor.get_champion_impact(champion, league)

            # Cor do impacto
            if impact > 0:
                impact_color = "#2ed573"
                impact_text = f"+{impact:.2f}"
            elif impact < 0:
                impact_color = "#ff4757"
                impact_text = f"{impact:.2f}"
            else:
                impact_color = "#747d8c"
                impact_text = "0.00"

            st.markdown(
                f"""
            <div class="champion-card blue-champion">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; font-size: 12px; color: #3498db; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">
                            {role}
                        </div>
                        <div style="font-size: 15px; font-weight: 600; color: white;">
                            {champion}
                        </div>
                    </div>
                    <div style="font-weight: 700; font-size: 14px; color: {impact_color};">
                        {impact_text}
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col_red:
        st.markdown("#### 🔴 RED SIDE")
        for i, player in enumerate(red_team[:5]):
            role = roles[i] if i < len(roles) else "ROLE"
            champion = player.get("championId", "Não selecionado")

            # Buscar impacto na liga específica
            impact = predictor.get_champion_impact(champion, league)

            # Cor do impacto
            if impact > 0:
                impact_color = "#2ed573"
                impact_text = f"+{impact:.2f}"
            elif impact < 0:
                impact_color = "#ff4757"
                impact_text = f"{impact:.2f}"
            else:
                impact_color = "#747d8c"
                impact_text = "0.00"

            st.markdown(
                f"""
            <div class="champion-card red-champion">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; font-size: 12px; color: #e74c3c; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">
                            {role}
                        </div>
                        <div style="font-size: 15px; font-weight: 600; color: white;">
                            {champion}
                        </div>
                    </div>
                    <div style="font-weight: 700; font-size: 14px; color: {impact_color};">
                        {impact_text}
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_classification_page():
    """Renderiza página do sistema de classificação"""
    st.markdown("# 🎯 Sistema de Classificação")
    st.markdown(
        "Entenda como funciona nosso sistema de classificação de apostas baseado em probabilidades e performance histórica."
    )

    # Sistema de classificação
    st.markdown("## 📊 Tabela de Classificação")

    # Criar DataFrame para a tabela
    import pandas as pd

    classification_data = {
        "Categoria": ["🟢 BOA", "🔵 MUITO BOA", "🟡 EXCELENTE"],
        "Probabilidade": ["0.55-0.64", "0.65-0.74", "≥0.75"],
        "Winrate UNDER": ["58.6%", "67.4%", "73.0%"],
        "Winrate OVER": ["61.0%", "65.7%", "65.4%"],
        "ROI UNDER": ["+7.3%", "+23.4%", "+33.5%"],
        "ROI OVER": ["+11.7%", "+20.3%", "+19.7%"],
    }

    df_classification = pd.DataFrame(classification_data)

    # Exibir tabela
    st.dataframe(df_classification, use_container_width=True, hide_index=True)

    # Explicações
    st.markdown("## 📖 Como Interpretar")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 🟢 BOA")
            st.markdown("**Estratégia de Volume**")
            st.markdown("- Probabilidade moderada (55-64%)")
            st.markdown("- ROI estável (~7-12%)")
            st.markdown("- Ideal para: Operação em escala")
            st.markdown("- Risco: Baixo a moderado")

    with col2:
        with st.container(border=True):
            st.markdown("### 🔵 MUITO BOA")
            st.markdown("**Estratégia Equilibrada**")
            st.markdown("- Probabilidade alta (65-74%)")
            st.markdown("- ROI excelente (~20-23%)")
            st.markdown("- Ideal para: Crescimento sustentável")
            st.markdown("- Risco: Moderado")

    with col3:
        with st.container(border=True):
            st.markdown("### 🟡 EXCELENTE")
            st.markdown("**Estratégia Premium**")
            st.markdown("- Probabilidade muito alta (≥75%)")
            st.markdown("- ROI máximo (~20-34%)")
            st.markdown("- Ideal para: Capital limitado")
            st.markdown("- Risco: Baixo")

    # Informações adicionais
    st.markdown("## ℹ️ Informações Importantes")

    with st.expander("🎯 Break-even e Margem de Segurança"):
        st.markdown("""
        **Break-even:** 54.6% de winrate (devido às odds 1.83)
        
        **Margem de Segurança:**
        - BOA: +4.0% a +6.4% acima do break-even
        - MUITO BOA: +10.8% a +12.8% acima do break-even  
        - EXCELENTE: +18.4% acima do break-even
        
        Todas as categorias operam com margem de segurança robusta.
        """)

    with st.expander("📊 Metodologia"):
        st.markdown("""
        **Base de Dados:** 389 jogos reais de apostas
        
        **Validação:** Testado com dados reais de casas de apostas
        
        **Odds:** 1.83 (Vitória: +0.83, Derrota: -1.00)
        
        **Modelo:** Regressão Logística com features de impacto de campeões por liga
        
        **Performance:** ROC-AUC ~0.746 (excelente para apostas esportivas)
        """)


def render_champion_impacts_page():
    """Renderiza página dos impactos dos campeões"""
    st.markdown("# ⚔️ Impactos dos Campeões por Liga")
    st.markdown(
        "Explore como cada campeão afeta o número de kills em diferentes ligas."
    )

    predictor = get_predictor()

    if predictor.champion_impacts:
        # Seletor de liga
        available_leagues = list(predictor.champion_impacts.keys())
        selected_league = st.selectbox(
            "Selecione uma Liga:",
            available_leagues,
            index=0 if available_leagues else None,
        )

        if selected_league:
            league_impacts = predictor.champion_impacts[selected_league]

            # Estatísticas da liga
            st.markdown(f"## 📊 Liga: {selected_league}")

            col1, col2, col3, col4 = st.columns(4)

            impacts_values = [v for v in league_impacts.values() if v != 0]

            with col1:
                st.metric("Total de Campeões", len(league_impacts))

            with col2:
                st.metric("Campeões Ativos", len(impacts_values))

            with col3:
                if impacts_values:
                    st.metric("Impacto Máximo", f"+{max(impacts_values):.2f}")
                else:
                    st.metric("Impacto Máximo", "N/A")

            with col4:
                if impacts_values:
                    st.metric("Impacto Mínimo", f"{min(impacts_values):.2f}")
                else:
                    st.metric("Impacto Mínimo", "N/A")

            # Filtros
            st.markdown("### 🔍 Filtros")

            col1, col2 = st.columns(2)

            with col1:
                impact_filter = st.selectbox(
                    "Filtrar por Impacto:",
                    ["Todos", "Positivo (+)", "Negativo (-)", "Neutro (0)"],
                )

            with col2:
                sort_by = st.selectbox(
                    "Ordenar por:",
                    [
                        "Impacto (Maior → Menor)",
                        "Impacto (Menor → Maior)",
                        "Nome (A → Z)",
                    ],
                )

            # Filtrar dados
            filtered_impacts = {}

            for champ, impact in league_impacts.items():
                if impact_filter == "Todos":
                    filtered_impacts[champ] = impact
                elif impact_filter == "Positivo (+)" and impact > 0:
                    filtered_impacts[champ] = impact
                elif impact_filter == "Negativo (-)" and impact < 0:
                    filtered_impacts[champ] = impact
                elif impact_filter == "Neutro (0)" and impact == 0:
                    filtered_impacts[champ] = impact

            # Ordenar dados
            if sort_by == "Impacto (Maior → Menor)":
                sorted_impacts = sorted(
                    filtered_impacts.items(), key=lambda x: x[1], reverse=True
                )
            elif sort_by == "Impacto (Menor → Maior)":
                sorted_impacts = sorted(filtered_impacts.items(), key=lambda x: x[1])
            else:  # Nome A → Z
                sorted_impacts = sorted(filtered_impacts.items(), key=lambda x: x[0])

            # Exibir tabela
            st.markdown(f"### 📋 Campeões ({len(sorted_impacts)} encontrados)")

            if sorted_impacts:
                # Criar DataFrame
                import pandas as pd

                df_data = []
                for champ, impact in sorted_impacts:
                    # Determinar categoria do impacto
                    if impact > 2:
                        categoria = "🔥 Alto Positivo"
                    elif impact > 0:
                        categoria = "✅ Positivo"
                    elif impact == 0:
                        categoria = "⚪ Neutro"
                    elif impact > -2:
                        categoria = "❌ Negativo"
                    else:
                        categoria = "🧊 Alto Negativo"

                    df_data.append(
                        {
                            "Campeão": champ,
                            "Impacto": f"{impact:+.2f}" if impact != 0 else "0.00",
                            "Categoria": categoria,
                        }
                    )

                df_impacts = pd.DataFrame(df_data)

                # Exibir com paginação
                st.dataframe(df_impacts, use_container_width=True, hide_index=True)

                # Estatísticas dos filtros
                if len(sorted_impacts) != len(league_impacts):
                    st.info(
                        f"Mostrando {len(sorted_impacts)} de {len(league_impacts)} campeões"
                    )

            else:
                st.warning("Nenhum campeão encontrado com os filtros selecionados.")

    else:
        st.error(
            "Dados de impacto dos campeões não disponíveis. Verifique se o modelo foi carregado corretamente."
        )


def main():
    """Função principal"""
    # Inicializar estado da sessão
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    if "selected_map" not in st.session_state:
        st.session_state.selected_map = 1

    # Sidebar para navegação
    with st.sidebar:
        st.markdown("## 🧭 Navegação")

        if st.button("🏠 Jogos", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()

        if st.button("🎯 Sistema de Classificação", use_container_width=True):
            st.session_state.current_page = "classification"
            st.rerun()

        if st.button("⚔️ Impactos dos Campeões", use_container_width=True):
            st.session_state.current_page = "champion_impacts"
            st.rerun()

    # Roteamento de páginas
    if st.session_state.current_page == "home":
        render_home_page()
    elif st.session_state.current_page == "analysis":
        render_analysis_page()
    elif st.session_state.current_page == "classification":
        render_classification_page()
    elif st.session_state.current_page == "champion_impacts":
        render_champion_impacts_page()
    else:
        render_home_page()


if __name__ == "__main__":
    main()
