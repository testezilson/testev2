#!/usr/bin/env python3
"""
Exemplo Prático de Uso do Modelo LoL UNDER/OVER
Demonstra como usar o modelo na prática com exemplos reais
"""

import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime

def carregar_modelo_completo():
    """Carrega todos os componentes do modelo"""
    print("🔄 Carregando modelo LoL UNDER/OVER...")
    
    try:
        # Carregar modelos treinados
        with open('lol_under_over_model/trained_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        # Carregar scaler
        with open('lol_under_over_model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Carregar impactos dos campeões
        with open('lol_under_over_model/champion_impacts.pkl', 'rb') as f:
            champion_impacts = pickle.load(f)
        
        # Carregar estatísticas das ligas
        with open('lol_under_over_model/league_stats.pkl', 'rb') as f:
            league_stats = pickle.load(f)
        
        # Carregar colunas das features
        with open('lol_under_over_model/feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        
        # Carregar classificação dos thresholds
        with open('lol_under_over_model/threshold_classification.json', 'r') as f:
            threshold_classification = json.load(f)
        
        print("✅ Modelo carregado com sucesso!")
        print(f"📊 {len(models)} modelos disponíveis (linhas 25.5 a 32.5)")
        print(f"🌍 {len(league_stats)} ligas suportadas")
        
        return models, scaler, champion_impacts, league_stats, feature_cols, threshold_classification
    
    except FileNotFoundError as e:
        print(f"❌ Erro: Arquivo não encontrado - {e}")
        print("💡 Certifique-se de que a pasta 'lol_under_over_model' está no diretório atual")
        return None, None, None, None, None, None

def criar_features_jogo(jogo_data, champion_impacts, league_stats):
    """Cria features para um jogo específico"""
    league = jogo_data['league']
    
    # Verificar se a liga existe
    if league not in league_stats:
        print(f"⚠️ Liga '{league}' não encontrada. Usando média geral.")
        league_avg = 28.5  # Média aproximada do dataset
        league_std = 7.99
    else:
        league_avg = league_stats[league]
        league_std = 7.99  # Valor padrão
    
    # Posições dos times
    positions_t1 = ['top_t1', 'jung_t1', 'mid_t1', 'adc_t1', 'sup_t1']
    positions_t2 = ['top_t2', 'jung_t2', 'mid_t2', 'adc_t2', 'sup_t2']
    
    # Calcular impactos dos times
    team1_impacts = []
    team2_impacts = []
    
    for pos in positions_t1:
        champion = jogo_data[pos]
        if league in champion_impacts and champion in champion_impacts[league]:
            impact = champion_impacts[league][champion]
        else:
            impact = 0.0  # Campeão não encontrado
        team1_impacts.append(impact)
    
    for pos in positions_t2:
        champion = jogo_data[pos]
        if league in champion_impacts and champion in champion_impacts[league]:
            impact = champion_impacts[league][champion]
        else:
            impact = 0.0  # Campeão não encontrado
        team2_impacts.append(impact)
    
    # Criar features
    features = {
        'league_encoded': hash(league) % 100,
        'mean_league_kills': league_avg,
        'std_league_kills': league_std,
        'mean_impact_team1': np.mean(team1_impacts),
        'mean_impact_team2': np.mean(team2_impacts),
        'total_impact': np.mean(team1_impacts) + np.mean(team2_impacts),
        'impact_diff': np.mean(team1_impacts) - np.mean(team2_impacts),
    }
    
    # Adicionar impactos individuais
    for i, impact in enumerate(team1_impacts):
        features[f'impact_t1_pos{i+1}'] = impact
    
    for i, impact in enumerate(team2_impacts):
        features[f'impact_t2_pos{i+1}'] = impact
    
    return features

def prever_jogo(jogo_data, models, scaler, champion_impacts, league_stats, feature_cols, threshold=0.55):
    """Faz predição para um jogo específico"""
    # Criar features
    features = criar_features_jogo(jogo_data, champion_impacts, league_stats)
    
    # Converter para DataFrame
    features_df = pd.DataFrame([features])
    features_df = features_df[feature_cols]  # Garantir ordem correta
    
    # Normalizar features
    features_scaled = scaler.transform(features_df)
    
    # Fazer predições para todas as linhas
    predictions = {}
    bet_lines = [25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5]
    
    for bet_line in bet_lines:
        model = models[bet_line]
        prob_under = model.predict_proba(features_scaled)[0, 1]
        
        # Aplicar threshold
        bet_under = prob_under >= threshold
        
        # Determinar confiança
        if prob_under > 0.7 or prob_under < 0.3:
            confidence = 'High'
        elif prob_under > 0.6 or prob_under < 0.4:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        predictions[bet_line] = {
            'probability_under': prob_under,
            'bet_under': bet_under,
            'confidence': confidence
        }
    
    return predictions

def mostrar_resultados(jogo_nome, predictions, threshold_info):
    """Mostra os resultados de forma organizada"""
    print(f"\n🎮 {jogo_nome}")
    print("=" * 60)
    
    # Contar apostas recomendadas
    apostas_recomendadas = sum(1 for pred in predictions.values() if pred['bet_under'])
    
    if apostas_recomendadas == 0:
        print("⚠️ Nenhuma aposta UNDER recomendada com este threshold")
        return
    
    print(f"📊 {apostas_recomendadas} apostas UNDER recomendadas:")
    print()
    
    # Mostrar apenas as apostas recomendadas
    for linha, pred in predictions.items():
        if pred['bet_under']:
            confidence_icon = {
                'High': '🔥',
                'Medium': '⚡',
                'Low': '💡'
            }[pred['confidence']]
            
            print(f"  {confidence_icon} UNDER {linha}: {pred['probability_under']:.1%} "
                  f"({pred['confidence']} confidence)")
    
    print(f"\n💡 Estratégia: {threshold_info}")

def exemplo_jogo_lpl():
    """Exemplo com jogo da LPL (Liga Chinesa)"""
    return {
        'nome': 'JDG vs BLG (LPL)',
        'league': 'LPL',
        'top_t1': 'Aatrox',
        'jung_t1': 'Graves',
        'mid_t1': 'Azir', 
        'adc_t1': 'Jinx',
        'sup_t1': 'Thresh',
        'top_t2': 'Gnar',
        'jung_t2': 'Sejuani',
        'mid_t2': 'Orianna',
        'adc_t2': 'Aphelios',
        'sup_t2': 'Braum'
    }

def exemplo_jogo_lck():
    """Exemplo com jogo da LCK (Liga Coreana)"""
    return {
        'nome': 'T1 vs GenG (LCK)',
        'league': 'LCK',
        'top_t1': 'Jayce',
        'jung_t1': 'Nidalee',
        'mid_t1': 'LeBlanc',
        'adc_t1': 'Caitlyn', 
        'sup_t1': 'Lux',
        'top_t2': 'Ornn',
        'jung_t2': 'Ammu',
        'mid_t2': 'Galio',
        'adc_t2': 'Ezreal',
        'sup_t2': 'Braum'
    }

def exemplo_jogo_lec():
    """Exemplo com jogo da LEC (Liga Europeia)"""
    return {
        'nome': 'G2 vs FNC (LEC)',
        'league': 'LEC',
        'top_t1': 'Fiora',
        'jung_t1': 'Elise',
        'mid_t1': 'Yasuo',
        'adc_t1': 'Kai\'Sa',
        'sup_t1': 'Pyke',
        'top_t2': 'Malphite', 
        'jung_t2': 'Rammus',
        'mid_t2': 'Malzahar',
        'adc_t2': 'Sivir',
        'sup_t2': 'Yuumi'
    }

def comparar_estrategias(jogo_data, models, scaler, champion_impacts, league_stats, feature_cols):
    """Compara as 3 estratégias (BOA, MUITO BOA, EXCELENTE)"""
    print(f"\n📊 COMPARAÇÃO DE ESTRATÉGIAS - {jogo_data['nome']}")
    print("=" * 70)
    
    estrategias = [
        (0.55, "🟢 BOA", "Volume alto, lucro máximo"),
        (0.65, "🔵 MUITO BOA", "Equilíbrio risco/retorno"),
        (0.75, "🟡 EXCELENTE", "Máxima precisão, volume seletivo")
    ]
    
    for threshold, categoria, descricao in estrategias:
        predictions = prever_jogo(jogo_data, models, scaler, champion_impacts, 
                                 league_stats, feature_cols, threshold=threshold)
        
        apostas_recomendadas = sum(1 for pred in predictions.values() if pred['bet_under'])
        apostas_alta_confianca = sum(1 for pred in predictions.values() 
                                   if pred['bet_under'] and pred['confidence'] == 'High')
        
        print(f"{categoria} (T={threshold}):")
        print(f"  📈 {descricao}")
        print(f"  🎯 {apostas_recomendadas} apostas recomendadas")
        print(f"  🔥 {apostas_alta_confianca} com alta confiança")
        print()

def analisar_multiplos_jogos():
    """Analisa múltiplos jogos de diferentes ligas"""
    print("\n🌍 ANÁLISE DE MÚLTIPLOS JOGOS")
    print("=" * 50)
    
    # Carregar modelo
    components = carregar_modelo_completo()
    if components[0] is None:
        return
    
    models, scaler, champion_impacts, league_stats, feature_cols, threshold_classification = components
    
    # Lista de jogos exemplo
    jogos = [
        exemplo_jogo_lpl(),
        exemplo_jogo_lck(), 
        exemplo_jogo_lec()
    ]
    
    # Analisar cada jogo com estratégia BOA (0.55)
    for jogo in jogos:
        predictions = prever_jogo(jogo, models, scaler, champion_impacts,
                                 league_stats, feature_cols, threshold=0.55)
        
        mostrar_resultados(jogo['nome'], predictions, 
                          "🟢 BOA - Estratégia de Volume (ROI ~36%)")
    
    # Comparar estratégias no primeiro jogo
    comparar_estrategias(jogos[0], models, scaler, champion_impacts, 
                        league_stats, feature_cols)

def demonstrar_uso_personalizado():
    """Demonstra como criar e analisar um jogo personalizado"""
    print("\n🎯 EXEMPLO PERSONALIZADO")
    print("=" * 40)
    
    # Carregar modelo
    components = carregar_modelo_completo()
    if components[0] is None:
        return
    
    models, scaler, champion_impacts, league_stats, feature_cols, threshold_classification = components
    
    # Criar jogo personalizado
    jogo_personalizado = {
        'nome': 'Jogo Personalizado (CBLOL)',
        'league': 'CBLOL',
        'top_t1': 'Darius',
        'jung_t1': 'Hecarim',
        'mid_t1': 'Syndra',
        'adc_t1': 'Jhin',
        'sup_t1': 'Leona',
        'top_t2': 'Shen',
        'jung_t2': 'Kindred', 
        'mid_t2': 'Twisted Fate',
        'adc_t2': 'Ashe',
        'sup_t2': 'Morgana'
    }
    
    print("📝 Jogo criado:")
    print(f"  Liga: {jogo_personalizado['league']}")
    print(f"  Time 1: {jogo_personalizado['top_t1']}, {jogo_personalizado['jung_t1']}, "
          f"{jogo_personalizado['mid_t1']}, {jogo_personalizado['adc_t1']}, {jogo_personalizado['sup_t1']}")
    print(f"  Time 2: {jogo_personalizado['top_t2']}, {jogo_personalizado['jung_t2']}, "
          f"{jogo_personalizado['mid_t2']}, {jogo_personalizado['adc_t2']}, {jogo_personalizado['sup_t2']}")
    
    # Analisar com estratégia MUITO BOA
    predictions = prever_jogo(jogo_personalizado, models, scaler, champion_impacts,
                             league_stats, feature_cols, threshold=0.65)
    
    mostrar_resultados(jogo_personalizado['nome'], predictions,
                      "🔵 MUITO BOA - Estratégia Equilibrada (ROI ~48%)")

def mostrar_informacoes_modelo():
    """Mostra informações sobre o modelo"""
    print("🤖 INFORMAÇÕES DO MODELO")
    print("=" * 40)
    print("📊 Performance: ROC-AUC ~0.746")
    print("🎯 Break-even: 54.6% winrate")
    print("💰 Odds: Vitória +0.83, Derrota -1.00")
    print("📈 Resultado: 100% das combinações lucrativas")
    print("🎮 Dataset: 5.320 jogos profissionais")
    print("🌍 Ligas: LPL, LCK, LEC, LCS, CBLOL e outras")
    print("📏 Linhas: 25.5 a 32.5 kills")
    print()
    print("🎯 ESTRATÉGIAS DISPONÍVEIS:")
    print("🟢 BOA (0.55-0.60): Volume alto, lucro máximo")
    print("🔵 MUITO BOA (0.65-0.70): Equilíbrio risco/retorno") 
    print("🟡 EXCELENTE (0.75+): Máxima precisão")

def main():
    """Função principal com menu interativo"""
    print("🎯 MODELO LOL UNDER/OVER - EXEMPLOS PRÁTICOS")
    print("=" * 60)
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    mostrar_informacoes_modelo()
    
    print("\n📋 EXEMPLOS DISPONÍVEIS:")
    print("1. Análise de múltiplos jogos")
    print("2. Exemplo personalizado")
    print("3. Sair")
    
    while True:
        try:
            escolha = input("\n🔢 Escolha uma opção (1-3): ").strip()
            
            if escolha == '1':
                analisar_multiplos_jogos()
            elif escolha == '2':
                demonstrar_uso_personalizado()
            elif escolha == '3':
                print("\n👋 Obrigado por usar o modelo LoL UNDER/OVER!")
                break
            else:
                print("❌ Opção inválida. Escolha 1, 2 ou 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Saindo...")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    main()
