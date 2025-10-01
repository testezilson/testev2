import streamlit as st
import requests
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="LoL Esports Schedule", page_icon="🎮", layout="wide")

# CSS customizado
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: rgba(70, 70, 90, 0.8);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 14px;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: rgba(90, 90, 110, 0.9);
        border-color: rgba(255, 255, 255, 0.2);
    }
    .game-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        background: rgba(30, 30, 40, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s;
        min-height: 160px;
        cursor: pointer;
    }
    .game-card:hover {
        background: rgba(35, 35, 45, 0.8);
        border-color: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }
    .live-card {
        border-left: 4px solid #ff4b4b;
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.1), rgba(30, 30, 40, 0.6));
    }
    .scheduled-card {
        border-left: 4px solid #4CAF50;
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1), rgba(30, 30, 40, 0.6));
    }
    .completed-card {
        border-left: 4px solid #666;
        background: rgba(30, 30, 40, 0.4);
    }
    .analysis-container {
        background: rgba(20, 20, 30, 0.9);
        border-radius: 12px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }
    .prediction-section {
        background: rgba(25, 25, 35, 0.9);
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .historical-section {
        background: rgba(30, 25, 35, 0.9);
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .team-info {
        background: rgba(40, 40, 50, 0.8);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #3498db;
        font-size: 16px;
        font-weight: 600;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #fff;
    }
    .champion-entry {
        background: rgba(40, 40, 50, 0.6);
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        border-left: 3px solid;
        color: rgba(255, 255, 255, 0.9);
    }
    .blue-champion {
        border-left-color: #3498db;
        background: linear-gradient(to right, rgba(52, 152, 219, 0.1), rgba(40, 40, 50, 0.6));
    }
    .red-champion {
        border-left-color: #e74c3c;
        background: linear-gradient(to right, rgba(231, 76, 60, 0.1), rgba(40, 40, 50, 0.6));
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Carregamento do modelo e dados
@st.cache_resource
def load_betting_model():
    try:
        model = joblib.load("betting_model_pipeline.pkl")
        model_info = joblib.load("model_info.pkl")
        return model, model_info
    except FileNotFoundError:
        st.error("Modelo de apostas não encontrado!")
        return None, None


@st.cache_data
def load_champion_impacts():
    try:
        with open("champion_impacts.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Dicionário de impacts não encontrado")
        return {}


@st.cache_data
def load_team_averages():
    try:
        with open("team_averages.json", "r") as f:
            data = json.load(f)
            print(f"DEBUG: team_averages.json carregado com sucesso")
            print(f"DEBUG: Ligas encontradas no JSON: {list(data.keys())}")
            return data
    except FileNotFoundError:
        st.warning("Dicionário de médias dos times não encontrado")
        print("DEBUG: Arquivo team_averages.json não encontrado")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"Erro ao decodificar team_averages.json: {e}")
        print(f"DEBUG: Erro JSON: {e}")
        return {}


def get_team_average(team_code, league, team_averages):
    """
    Busca a média do time considerando códigos abreviados da API
    """
    # Debug: imprimir o que estamos procurando
    print(f"DEBUG: Buscando {team_code} na liga {league}")

    # Primeiro, tentar busca direta pelo código
    if league in team_averages:
        league_teams = team_averages[league]
        print(f"DEBUG: Liga {league} encontrada com {len(league_teams)} times")

        # Busca exata pelo código
        if team_code in league_teams:
            result = league_teams[team_code]
            print(f"DEBUG: Encontrado {team_code} diretamente = {result}")
            return result

        # Mapeamento de códigos conhecidos para nomes completos
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
            "EMEA": {
                # Pool 1
                "KCB": "Karmine Corp Blue",
                "KC": "Karmine Corp Blue",
                "LR": "Los Ratones",
                "RAT": "Los Ratones",
                "GXP": "GIANTX PRIDE",
                "GIANTXPRIDE": "GIANTX PRIDE",
                "BIG": "BIG",
                "GAL": "Galions",
                "GLN": "Galions",
                "VER": "Verdant",
                "VDT": "Verdant",
                "LHP": "Los Heretics",
                "HER": "Los Heretics",
                "TOG": "TeamOrangeGaming",
                "ORG": "TeamOrangeGaming",
                # Pool 2
                "GM": "Gentle Mates",
                "GMA": "Gentle Mates",
                "BAR": "Barça eSports",
                "BARC": "Barça eSports",
                "RC": "ROSSMANN Centaurs",
                "CENT": "ROSSMANN Centaurs",
                "MISA": "Misa Esports",
                "MIS": "Misa Esports",
                "GKY": "Geekay Esports",
                "GEEK": "Geekay Esports",
                "BW": "Bushido Wildcats",
                "BUSH": "Bushido Wildcats",
                "GNG": "GnG Amazigh",
                "AMZ": "GnG Amazigh",
                "ULF": "ULF Esports",
                # Pool 3
                "ZENA": "Zena Esports",
                "ZEN": "Zena Esports",
                "TP": "Team Phantasma",
                "PHAN": "Team Phantasma",
                "ESUBA": "eSuba",
                "ESUB": "eSuba",
                "ZT": "Zero Tenacity",
                "ZERT": "Zero Tenacity",
                "CG": "Colossal Gaming",
                "COL": "Colossal Gaming",
                "GSM": "Gamespace MCE",
                "GSP": "Gamespace MCE",
                "ENT": "Entropiq",
                "ENQ": "Entropiq",
                "SMF": "StormMedia FMS",
                "STORM": "StormMedia FMS",
                # Pool 4
                "SEN": "Senshi eSports",
                "SENSHI": "Senshi eSports",
                "PART": "Partizan Sangal",
                "PZS": "Partizan Sangal",
                "OS": "Otter Side",
                "OTTER": "Otter Side",
                "VVV": "Veni Vidi Vici"
            },
        }

        # Tentar mapeamento por código
        if league in team_mappings and team_code in team_mappings[league]:
            full_name = team_mappings[league][team_code]
            print(f"DEBUG: Mapeando {team_code} -> {full_name}")
            if full_name in league_teams:
                result = league_teams[full_name]
                print(f"DEBUG: Encontrado {full_name} = {result}")
                return result
            else:
                print(f"DEBUG: {full_name} não encontrado na liga")

        # Mostrar times disponíveis para debug
        print(f"DEBUG: Times disponíveis na {league}: {list(league_teams.keys())}")

        # Busca parcial por substring (fallback)
        for team_name in league_teams:
            if team_code.lower() in team_name.lower() or team_name.lower().startswith(
                team_code.lower()[:3]
            ):
                result = league_teams[team_name]
                print(
                    f"DEBUG: Encontrado por substring {team_code} -> {team_name} = {result}"
                )
                return result
    else:
        print(f"DEBUG: Liga {league} não encontrada no JSON")
        if team_averages:
            print(f"DEBUG: Ligas disponíveis: {list(team_averages.keys())}")

    # Valor padrão se não encontrar
    print(f"DEBUG: Usando valor padrão 30.0 para {team_code}")
    return 30.0


@st.cache_data(ttl=60)
def get_schedule():
    url = "https://esports-api.lolesports.com/persisted/gw/getSchedule?hl=en-US"
    headers = {
        "x-api-key": "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z",
        "User-Agent": "Mozilla/5.0",
    }

    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("data", {}).get("schedule", {}).get("events", [])

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
        st.error(f"Erro ao buscar dados: {e}")
        return []


def to_brazil_time(utc_time_str):
    if not utc_time_str:
        return None
    brazil_tz = pytz.timezone("America/Sao_Paulo")
    utc_time = datetime.fromisoformat(utc_time_str.replace("Z", "+00:00"))
    brazil_time = utc_time.astimezone(brazil_tz)
    return brazil_time


def normalize_champion_name(champion_name):
    """
    Normaliza nomes de campeões para fazer match entre API e JSON
    """
    name_mapping = {
        # Casos especiais comuns
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

    return name_mapping.get(champion_name, champion_name)


def format_team_names(teams):
    if len(teams) == 2:
        team1 = teams[0].get("code", teams[0].get("name", "TBD"))
        team2 = teams[1].get("code", teams[1].get("name", "TBD"))
        return team1, team2
    return "TBD", "TBD"


def format_datetime(start_time):
    if start_time:
        brazil_time = to_brazil_time(start_time)
        if brazil_time:
            today = datetime.now(pytz.timezone("America/Sao_Paulo")).date()
            game_date = brazil_time.date()

            if game_date == today:
                day_str = "Hoje"
            elif game_date == today + timedelta(days=1):
                day_str = "Amanhã"
            else:
                day_str = brazil_time.strftime("%d/%m")

            time_str = brazil_time.strftime("%H:%M")
            return f"{day_str} • {time_str}"
    return ""


def get_draft(match_id, game_number):
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
    except:
        return None, None


def draft_to_features(
    blue_team,
    red_team,
    champion_impacts,
    team_averages,
    bet_line,
    team1_name,
    team2_name,
    league,
):
    # Usar nova função para buscar médias
    team1_avg = get_team_average(team1_name, league, team_averages)
    team2_avg = get_team_average(team2_name, league, team_averages)

    features = {
        "league_mean_kills": bet_line,
        "mean_total_kills_t1": team1_avg,
        "mean_total_kills_t2": team2_avg,
        "impact_top_t1": 0.0,
        "impact_top_t2": 0.0,
        "impact_jung_t1": 0.0,
        "impact_jung_t2": 0.0,
        "impact_mid_t1": 0.0,
        "impact_mid_t2": 0.0,
        "impact_adc_t1": 0.0,
        "impact_adc_t2": 0.0,
        "impact_sup_t1": 0.0,
        "impact_sup_t2": 0.0,
    }

    roles = ["top", "jung", "mid", "adc", "sup"]

    for i, player in enumerate(blue_team[:5]):
        if i < len(roles):
            champion = player.get("championId", "")
            normalized_champion = normalize_champion_name(champion)
            role = roles[i]
            # Buscar pelo nome normalizado
            impact = champion_impacts.get(normalized_champion, 0.0)
            features[f"impact_{role}_t1"] = impact

    for i, player in enumerate(red_team[:5]):
        if i < len(roles):
            champion = player.get("championId", "")
            normalized_champion = normalize_champion_name(champion)
            role = roles[i]
            # Buscar pelo nome normalizado
            impact = champion_impacts.get(normalized_champion, 0.0)
            features[f"impact_{role}_t2"] = impact

    return features


def predict_betting_line(model, model_info, features):
    feature_columns = model_info["feature_columns"]
    X = pd.DataFrame([features])[feature_columns]

    probability = model.predict_proba(X)[0, 1]
    threshold = model_info["chosen_threshold"]

    if probability >= threshold:
        recommendation = "OVER"
        confidence = "ALTA" if probability >= 0.75 else "MODERADA"
    elif probability <= (1 - threshold):
        recommendation = "UNDER"
        confidence = "ALTA" if probability <= 0.25 else "MODERADA"
    else:
        recommendation = "NEUTRO"
        confidence = "BAIXA"

    return {
        "probability": probability,
        "recommendation": recommendation,
        "confidence": confidence,
        "threshold": threshold,
    }


def render_game_cards(games, prefix):
    if not games:
        return

    for i in range(0, len(games), 2):  # 2 cards por linha
        cols = st.columns(2)

        for j in range(2):
            if i + j < len(games):
                game = games[i + j]
                game_index = f"{prefix}_{i + j}"

                with cols[j]:
                    render_single_game_card(game, game_index)


def render_single_game_card(game, game_index):
    state = game.get("state", "N/A")
    league = game.get("league", {}).get("name", "N/A")
    match = game.get("match", {})
    match_id = match.get("id", "N/A")
    teams = match.get("teams", [])
    team1, team2 = format_team_names(teams)
    datetime_str = format_datetime(game.get("startTime"))
    best_of = match.get("strategy", {}).get("count", 1)

    if state == "inProgress":
        status = "🔴 AO VIVO"
        card_class = "game-card live-card"
        status_color = "#ff4b4b"
    elif state == "unstarted":
        status = "📅 AGENDADO"
        card_class = "game-card scheduled-card"
        status_color = "#4CAF50"
    else:
        status = "✅ FINALIZADO"
        card_class = "game-card completed-card"
        status_color = "#888"

    result_text = ""
    if state == "completed" and len(teams) == 2:
        team1_wins = teams[0].get("result", {}).get("gameWins", 0)
        team2_wins = teams[1].get("result", {}).get("gameWins", 0)
        result_text = f" ({team1_wins}-{team2_wins})"

    st.markdown(
        f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="color: {status_color}; font-weight: 600; font-size: 12px;">
                    {status}
                </span>
                <span style="color: rgba(255, 255, 255, 0.5); font-size: 11px;">
                    BO{best_of}
                </span>
            </div>
            <div style="font-size: 18px; font-weight: bold; color: rgba(255, 255, 255, 0.95); margin: 10px 0;">
                {team1} vs {team2}{result_text}
            </div>
            <div style="color: rgba(255, 255, 255, 0.6); font-size: 12px; margin-bottom: 10px;">
                {league}
            </div>
            <div style="color: rgba(255, 255, 255, 0.7); font-size: 13px;">
                {datetime_str if datetime_str else "Horário não definido"}
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button(
        "📊 Analisar Jogo", key=f"analyze_{game_index}", use_container_width=True
    ):
        st.session_state.current_page = "game_analysis"
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


def render_game_analysis():
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
        st.title(f"🎮 {game_data['team1']} vs {game_data['team2']}")
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

    # Obter draft
    blue_team, red_team = get_draft(game_data["match_id"], current_map)

    if blue_team and red_team:
        render_betting_analysis(blue_team, red_team, game_data)
        render_draft_display(blue_team, red_team)
    else:
        st.info(f"📊 Draft do Mapa {current_map} não disponível no momento.")


def render_betting_analysis(blue_team, red_team, game_data):
    model, model_info = load_betting_model()
    champion_impacts = load_champion_impacts()
    team_averages = load_team_averages()

    if not model or not model_info:
        return

    # DEBUG: Imprimir dados do jogo
    print("\n" + "=" * 60)
    print("DEBUG: DADOS DO JOGO SELECIONADO")
    print("=" * 60)
    print(f"Liga: {game_data['league']}")
    print(f"Time 1: {game_data['team1']}")
    print(f"Time 2: {game_data['team2']}")
    print(f"Data/Hora: {game_data.get('datetime', 'N/A')}")
    print(f"Estado: {game_data['state']}")
    print(f"Best of: {game_data['best_of']}")
    print(f"Match ID: {game_data['match_id']}")
    print(f"Mapa selecionado: {st.session_state.get('selected_map', 1)}")

    print("\nDRAFT - BLUE TEAM:")
    for i, player in enumerate(blue_team[:5]):
        roles = ["TOP", "JUNG", "MID", "ADC", "SUP"]
        role = roles[i] if i < len(roles) else "ROLE"
        summoner = player.get("summonerName", "Unknown")
        champion = player.get("championId", "N/A")
        print(f"  {role}: {summoner} - {champion}")

    print("\nDRAFT - RED TEAM:")
    for i, player in enumerate(red_team[:5]):
        roles = ["TOP", "JUNG", "MID", "ADC", "SUP"]
        role = roles[i] if i < len(roles) else "ROLE"
        summoner = player.get("summonerName", "Unknown")
        champion = player.get("championId", "N/A")
        print(f"  {role}: {summoner} - {champion}")
    print("=" * 60 + "\n")

    st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🤖 Análise de Apostas</div>', unsafe_allow_html=True
    )

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
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🔮 Prever", type="primary", use_container_width=True):
            features = draft_to_features(
                blue_team,
                red_team,
                champion_impacts,
                team_averages,
                bet_line,
                game_data["team1"],
                game_data["team2"],
                game_data["league"],
            )

            print("\n" + "=" * 60)
            print("DEBUG: FEATURES ENVIADAS AO MODELO")
            print("=" * 60)
            print(f"Time 1: {game_data['team1']} (Liga: {game_data['league']})")
            print(f"Time 2: {game_data['team2']}")
            print(f"Linha de apostas: {bet_line}")
            print("\nFeatures completas:")
            for key, value in features.items():
                print(f"  {key}: {value}")

            print("\nDETALHE DOS CAMPEÕES:")
            print("BLUE TEAM:")
            for i, player in enumerate(blue_team[:5]):
                if i < 5:
                    roles = ["top", "jung", "mid", "adc", "sup"]
                    role = roles[i]
                    champion = player.get("championId", "")
                    normalized = normalize_champion_name(champion)
                    impact = champion_impacts.get(normalized, 0.0)
                    print(
                        f"  {role.upper()}: {champion} -> {normalized} (Impact: {impact})"
                    )

            print("RED TEAM:")
            for i, player in enumerate(red_team[:5]):
                if i < 5:
                    roles = ["top", "jung", "mid", "adc", "sup"]
                    role = roles[i]
                    champion = player.get("championId", "")
                    normalized = normalize_champion_name(champion)
                    impact = champion_impacts.get(normalized, 0.0)
                    print(
                        f"  {role.upper()}: {champion} -> {normalized} (Impact: {impact})"
                    )

            print("=" * 60)

            prediction = predict_betting_line(model, model_info, features)
            st.session_state.prediction_result = prediction
            st.session_state.prediction_line = bet_line
            st.session_state.team_info = {
                "team1": game_data["team1"],
                "team2": game_data["team2"],
                "team1_avg": get_team_average(
                    game_data["team1"], game_data["league"], team_averages
                ),
                "team2_avg": get_team_average(
                    game_data["team2"], game_data["league"], team_averages
                ),
            }

    if "prediction_result" in st.session_state:
        pred = st.session_state.prediction_result
        pred_line = st.session_state.prediction_line
        team_info = st.session_state.team_info

        st.markdown("---")
        st.markdown("### 📊 Resultado da Previsão")

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown(
                f'<div class="team-info">📊 {team_info["team1"]}: {team_info["team1_avg"]:.1f} kills/jogo</div>',
                unsafe_allow_html=True,
            )
        with col_info2:
            st.markdown(
                f'<div class="team-info">📊 {team_info["team2"]}: {team_info["team2_avg"]:.1f} kills/jogo</div>',
                unsafe_allow_html=True,
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Probabilidade OVER")
            st.markdown(
                f'<div style="font-size: 32px; font-weight: bold; color: #4CAF50;">{pred["probability"]:.1%}</div>',
                unsafe_allow_html=True,
            )

        with col2:
            rec_color = {"OVER": "#4CAF50", "UNDER": "#f44336", "NEUTRO": "#888"}.get(
                pred["recommendation"], "#888"
            )
            rec_emoji = {"OVER": "🟢", "UNDER": "🔴", "NEUTRO": "⚫"}.get(
                pred["recommendation"], "⚫"
            )
            st.markdown("#### Recomendação")
            st.markdown(
                f'<div style="font-size: 28px; font-weight: bold; color: {rec_color};">{rec_emoji} {pred["recommendation"]}</div>',
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown("#### Ação Sugerida")
            if pred["recommendation"] == "OVER":
                action_text = f"OVER {pred_line}"
                action_color = "#4CAF50"
            elif pred["recommendation"] == "UNDER":
                action_text = f"UNDER {pred_line}"
                action_color = "#f44336"
            else:
                action_text = "Não apostar"
                action_color = "#888"

            st.markdown(
                f'<div style="font-size: 24px; font-weight: bold; color: {action_color};">{action_text}</div>',
                unsafe_allow_html=True,
            )

        if pred["confidence"] == "ALTA":
            if pred["recommendation"] == "OVER":
                st.success(
                    f"✅ **Modelo tem alta confiança que será OVER {pred_line}**"
                )
            elif pred["recommendation"] == "UNDER":
                st.success(
                    f"✅ **Modelo tem alta confiança que será UNDER {pred_line}**"
                )
        elif pred["confidence"] == "MODERADA":
            st.warning(
                f"⚠️ **Modelo tem confiança moderada na recomendação {pred['recommendation']}**"
            )
        else:
            st.info("ℹ️ **Modelo sugere não apostar - probabilidade na zona neutra**")

    st.markdown("</div>", unsafe_allow_html=True)

def render_draft_display(blue_team, red_team):
    st.markdown("### 📋 Draft dos Times")

    col_blue, col_red = st.columns(2)
    champion_impacts = load_champion_impacts()

    with col_blue:
        st.markdown("#### 🔵 BLUE SIDE")
        for i, p in enumerate(blue_team[:5]):
            roles = ["TOP", "JUNG", "MID", "ADC", "SUP"]
            role = roles[i] if i < len(roles) else "ROLE"
            summoner = p.get("summonerName", "Unknown")
            champion = p.get("championId", "Não selecionado")

            # Buscar impact pelo nome normalizado
            normalized_champion = normalize_champion_name(champion)
            impact = champion_impacts.get(normalized_champion, None)

            # Formatar exibição do impact
            if impact is not None:
                impact_color = (
                    "#4CAF50" if impact > 0 else "#f44336" if impact < 0 else "#888"
                )
                impact_text = f'<span style="color: {impact_color}; font-weight: bold;">{impact:+.2f}</span>'
                debug_info = ""
            else:
                impact_text = (
                    '<span style="color: #FF9800; font-weight: bold;">N/A</span>'
                )
                debug_info = f'<div style="color: #FF5722; font-size: 10px; font-style: italic;">Missing: {normalized_champion}</div>'

            st.markdown(
                f"""
                <div class="champion-entry blue-champion">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-weight: 600; font-size: 12px; color: #3498db;">
                            {role}
                        </div>
                        <div style="font-size: 14px;">
                            Impact: {impact_text}
                        </div>
                    </div>
                    <div style="font-size: 14px; margin: 3px 0; font-weight: 500;">
                        {summoner}
                    </div>
                    <div style="color: rgba(255, 255, 255, 0.7); font-size: 12px;">
                        → {champion}
                    </div>
                    {debug_info}
                </div>
            """,
                unsafe_allow_html=True,
            )

    with col_red:
        st.markdown("#### 🔴 RED SIDE")
        for i, p in enumerate(red_team[:5]):
            roles = ["TOP", "JUNG", "MID", "ADC", "SUP"]
            role = roles[i] if i < len(roles) else "ROLE"
            summoner = p.get("summonerName", "Unknown")
            champion = p.get("championId", "Não selecionado")

            # Buscar impact pelo nome normalizado
            normalized_champion = normalize_champion_name(champion)
            impact = champion_impacts.get(normalized_champion, None)

            # Formatar exibição do impact
            if impact is not None:
                impact_color = (
                    "#4CAF50" if impact > 0 else "#f44336" if impact < 0 else "#888"
                )
                impact_text = f'<span style="color: {impact_color}; font-weight: bold;">{impact:+.2f}</span>'
                debug_info = ""
            else:
                impact_text = (
                    '<span style="color: #FF9800; font-weight: bold;">N/A</span>'
                )
                debug_info = f'<div style="color: #FF5722; font-size: 10px; font-style: italic;">Missing: {normalized_champion}</div>'

            st.markdown(
                f"""
                <div class="champion-entry red-champion">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-weight: 600; font-size: 12px; color: #e74c3c;">
                            {role}
                        </div>
                        <div style="font-size: 14px;">
                            Impact: {impact_text}
                        </div>
                    </div>
                    <div style="font-size: 14px; margin: 3px 0; font-weight: 500;">
                        {summoner}
                    </div>
                    <div style="color: rgba(255, 255, 255, 0.7); font-size: 12px;">
                        → {champion}
                    </div>
                    {debug_info}
                </div>
            """,
                unsafe_allow_html=True,
            )


def main():
    # Inicializar estado da página
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"

    # Roteamento
    if st.session_state.current_page == "home":
        render_home_page()
    elif st.session_state.current_page == "game_analysis":
        render_game_analysis()


def render_home_page():
    st.title("🎮 League of Legends Esports Schedule")

    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("🔄", help="Atualizar dados"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Carregando jogos..."):
        events = get_schedule()

    if not events:
        st.warning("⚠️ Nenhum jogo encontrado no momento.")
        return

    live_games = [g for g in events if g.get("state") == "inProgress"]
    scheduled_games = [g for g in events if g.get("state") == "unstarted"][:20]
    completed_games = [g for g in events if g.get("state") == "completed"][:10]

    if live_games:
        st.header("🔴 Jogos Ao Vivo")
        render_game_cards(live_games, "live")
        st.markdown("---")

    if scheduled_games:
        st.header("📅 Próximos Jogos")
        render_game_cards(scheduled_games, "scheduled")
        st.markdown("---")

    if completed_games:
        with st.expander("✅ Jogos Finalizados", expanded=False):
            render_game_cards(completed_games, "completed")

    st.caption("🕐 Horários exibidos no fuso horário de Brasília (GMT-3)")


if __name__ == "__main__":
    main()
