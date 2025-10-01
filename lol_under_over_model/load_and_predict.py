#!/usr/bin/env python3
"""
Script de exemplo para carregar e usar o modelo LoL UNDER/OVER
"""

import pickle
import pandas as pd
import numpy as np
import json

def load_model_components():
    """Carrega todos os componentes do modelo"""
    print("🔄 Carregando componentes do modelo...")
    
    # Carregar modelos
    with open('trained_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    # Carregar scaler
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Carregar impactos
    with open('champion_impacts.pkl', 'rb') as f:
        champion_impacts = pickle.load(f)
    
    # Carregar stats das ligas
    with open('league_stats.pkl', 'rb') as f:
        league_stats = pickle.load(f)
    
    # Carregar colunas das features
    with open('feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Carregar classificação dos thresholds
    with open('threshold_classification.json', 'r') as f:
        threshold_classification = json.load(f)
    
    print("✅ Todos os componentes carregados!")
    
    return models, scaler, champion_impacts, league_stats, feature_cols, threshold_classification

def create_features_for_game(game_data, champion_impacts, league_stats):
    """Cria features para um novo jogo"""
    league = game_data['league']
    league_avg = league_stats[league]
    league_std = 7.99  # Valor médio do dataset original
    
    # Calcular impactos dos times
    positions_t1 = ['top_t1', 'jung_t1', 'mid_t1', 'adc_t1', 'sup_t1']
    positions_t2 = ['top_t2', 'jung_t2', 'mid_t2', 'adc_t2', 'sup_t2']
    
    team1_impacts = []
    team2_impacts = []
    
    for pos in positions_t1:
        champion = game_data[pos]
        impact = champion_impacts[league].get(champion, 0.0)
        team1_impacts.append(impact)
    
    for pos in positions_t2:
        champion = game_data[pos]
        impact = champion_impacts[league].get(champion, 0.0)
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
    
    # Impactos individuais
    for i, impact in enumerate(team1_impacts):
        features[f'impact_t1_pos{i+1}'] = impact
    
    for i, impact in enumerate(team2_impacts):
        features[f'impact_t2_pos{i+1}'] = impact
    
    return features

def predict_game(game_data, models, scaler, champion_impacts, league_stats, feature_cols, threshold=0.55):
    """Faz predição para um jogo específico"""
    # Criar features
    features = create_features_for_game(game_data, champion_impacts, league_stats)
    
    # Converter para DataFrame
    features_df = pd.DataFrame([features])
    features_df = features_df[feature_cols]  # Garantir ordem correta
    
    # Normalizar
    features_scaled = scaler.transform(features_df)
    
    # Fazer predições para todas as linhas
    predictions = {}
    bet_lines = [25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5]
    
    for bet_line in bet_lines:
        model = models[bet_line]
        prob_under = model.predict_proba(features_scaled)[0, 1]
        
        # Aplicar threshold
        bet_under = prob_under >= threshold
        
        predictions[bet_line] = {
            'probability_under': prob_under,
            'bet_under': bet_under,
            'confidence': 'High' if prob_under > 0.7 or prob_under < 0.3 else 'Medium'
        }
    
    return predictions

# Exemplo de uso
if __name__ == "__main__":
    # Carregar modelo
    models, scaler, champion_impacts, league_stats, feature_cols, threshold_classification = load_model_components()
    
    # Exemplo de jogo
    example_game = {
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
    
    # Fazer predição
    predictions = predict_game(example_game, models, scaler, champion_impacts, league_stats, feature_cols, threshold=0.55)
    
    # Mostrar resultados
    print("\n🎯 PREDIÇÕES PARA O JOGO:")
    print("=" * 50)
    for bet_line, pred in predictions.items():
        print(f"Linha {bet_line}: Prob={pred['probability_under']:.1%}, "
              f"Apostar UNDER: {pred['bet_under']}, Confiança: {pred['confidence']}")
