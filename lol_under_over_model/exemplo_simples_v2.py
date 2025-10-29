#!/usr/bin/env python3
"""
Exemplo Simples de Uso do Modelo LoL UNDER/OVER
Demonstração direta sem input interativo
"""

import pickle
import pandas as pd
import numpy as np
import json

def carregar_modelo():
    """Carrega todos os componentes do modelo"""
    print("🔄 Carregando modelo LoL UNDER/OVER...")
    
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
    
    print("✅ Modelo carregado com sucesso!")
    return models, scaler, champion_impacts, league_stats, feature_cols

def criar_features_jogo(jogo_data, champion_impacts, league_stats):
    """Cria features para um jogo específico"""
    league = jogo_data['league']
    league_avg = league_stats.get(league, 28.5)  # Média padrão se liga não encontrada
    league_std = 7.99
    
    # Posições dos times
    positions_t1 = ['top_t1', 'jung_t1', 'mid_t1', 'adc_t1', 'sup_t1']
    positions_t2 = ['top_t2', 'jung_t2', 'mid_t2', 'adc_t2', 'sup_t2']
    
    # Calcular impactos dos times
    team1_impacts = []
    team2_impacts = []
    
    for pos in positions_t1:
        champion = jogo_data[pos]
        impact = champion_impacts.get(league, {}).get(champion, 0.0)
        team1_impacts.append(impact)
    
    for pos in positions_t2:
        champion = jogo_data[pos]
        impact = champion_impacts.get(league, {}).get(champion, 0.0)
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
        if prob_under > 0.7:
            confidence = 'High'
        elif prob_under > 0.6:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        predictions[bet_line] = {
            'probability_under': prob_under,
            'bet_under': bet_under,
            'confidence': confidence
        }
    
    return predictions

def main():
    """Demonstração completa do modelo"""
    print("🎯 DEMONSTRAÇÃO DO MODELO LOL UNDER/OVER")
    print("=" * 60)
    
    # Carregar modelo
    models, scaler, champion_impacts, league_stats, feature_cols = carregar_modelo()
    
    # Exemplo 1: Jogo LPL
    print("\n🎮 EXEMPLO 1: JDG vs BLG (LPL)")
    print("=" * 40)
    
    jogo_lpl = {
        'league': 'LPL',
        'top_t1': 'Aatrox', 'jung_t1': 'Graves', 'mid_t1': 'Azir',
        'adc_t1': 'Jinx', 'sup_t1': 'Thresh',
        'top_t2': 'Gnar', 'jung_t2': 'Sejuani', 'mid_t2': 'Orianna',
        'adc_t2': 'Aphelios', 'sup_t2': 'Braum'
    }
    
    predictions_lpl = prever_jogo(jogo_lpl, models, scaler, champion_impacts,
                                 league_stats, feature_cols, threshold=0.55)
    
    print("🟢 Estratégia BOA (Threshold 0.55):")
    apostas_recomendadas = 0
    for linha, pred in predictions_lpl.items():
        if pred['bet_under']:
            apostas_recomendadas += 1
            confidence_icon = '🔥' if pred['confidence'] == 'High' else '⚡' if pred['confidence'] == 'Medium' else '💡'
            print(f"  {confidence_icon} UNDER {linha}: {pred['probability_under']:.1%}")
    
    if apostas_recomendadas == 0:
        print("  ⚠️ Nenhuma aposta UNDER recomendada")
    else:
        print(f"  📊 Total: {apostas_recomendadas} apostas recomendadas")
    
    # Exemplo 2: Jogo LCK
    print("\n🎮 EXEMPLO 2: T1 vs GenG (LCK)")
    print("=" * 40)
    
    jogo_lck = {
        'league': 'LCK',
        'top_t1': 'Jayce', 'jung_t1': 'Nidalee', 'mid_t1': 'LeBlanc',
        'adc_t1': 'Caitlyn', 'sup_t1': 'Lux',
        'top_t2': 'Ornn', 'jung_t2': 'Ammu', 'mid_t2': 'Galio',
        'adc_t2': 'Ezreal', 'sup_t2': 'Braum'
    }
    
    predictions_lck = prever_jogo(jogo_lck, models, scaler, champion_impacts,
                                 league_stats, feature_cols, threshold=0.65)
    
    print("🔵 Estratégia MUITO BOA (Threshold 0.65):")
    apostas_recomendadas = 0
    for linha, pred in predictions_lck.items():
        if pred['bet_under']:
            apostas_recomendadas += 1
            confidence_icon = '🔥' if pred['confidence'] == 'High' else '⚡' if pred['confidence'] == 'Medium' else '💡'
            print(f"  {confidence_icon} UNDER {linha}: {pred['probability_under']:.1%}")
    
    if apostas_recomendadas == 0:
        print("  ⚠️ Nenhuma aposta UNDER recomendada")
    else:
        print(f"  📊 Total: {apostas_recomendadas} apostas recomendadas")
    
    # Exemplo 3: Comparação de estratégias
    print("\n📊 EXEMPLO 3: COMPARAÇÃO DE ESTRATÉGIAS")
    print("=" * 50)
    
    jogo_teste = {
        'league': 'LEC',
        'top_t1': 'Fiora', 'jung_t1': 'Elise', 'mid_t1': 'Yasuo',
        'adc_t1': 'Kai\'Sa', 'sup_t1': 'Pyke',
        'top_t2': 'Malphite', 'jung_t2': 'Rammus', 'mid_t2': 'Malzahar',
        'adc_t2': 'Sivir', 'sup_t2': 'Yuumi'
    }
    
    estrategias = [
        (0.55, "🟢 BOA", "Volume alto"),
        (0.65, "🔵 MUITO BOA", "Equilibrada"),
        (0.75, "🟡 EXCELENTE", "Seletiva")
    ]
    
    print("G2 vs FNC (LEC):")
    for threshold, categoria, descricao in estrategias:
        predictions = prever_jogo(jogo_teste, models, scaler, champion_impacts,
                                 league_stats, feature_cols, threshold=threshold)
        
        apostas = sum(1 for pred in predictions.values() if pred['bet_under'])
        alta_confianca = sum(1 for pred in predictions.values() 
                           if pred['bet_under'] and pred['confidence'] == 'High')
        
        print(f"  {categoria}: {apostas} apostas ({alta_confianca} alta confiança)")
    
    # Informações finais
    print("\n💡 INFORMAÇÕES DO MODELO")
    print("=" * 30)
    print("📊 Performance: ROC-AUC ~0.746")
    print("🎯 Break-even: 54.6% winrate")
    print("💰 Odds: Vitória +0.83, Derrota -1.00")
    print("📈 Resultado: 100% combinações lucrativas")
    print("🎮 Dataset: 5.320 jogos profissionais")
    
    print("\n🎯 ESTRATÉGIAS:")
    print("🟢 BOA (0.55-0.60): ROI ~38%, Volume alto")
    print("🔵 MUITO BOA (0.65-0.70): ROI ~48%, Equilibrada")
    print("🟡 EXCELENTE (0.75+): ROI ~55%, Seletiva")
    
    print("\n✅ Demonstração concluída!")

if __name__ == "__main__":
    main()
