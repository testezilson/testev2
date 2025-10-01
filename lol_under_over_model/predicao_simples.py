#!/usr/bin/env python3
"""
Script Simplificado para Predição Rápida
Uso direto com input mínimo - CAMINHOS CORRIGIDOS
"""

import pickle
import pandas as pd
import numpy as np
import os

def carregar_modelo():
    """Carrega modelo de forma simplificada com caminhos flexíveis"""
    
    # Detectar se estamos dentro da pasta do modelo ou fora
    if os.path.exists('trained_models.pkl'):
        # Executando de dentro da pasta lol_under_over_model
        base_path = ''
    elif os.path.exists('lol_under_over_model/trained_models.pkl'):
        # Executando de fora da pasta lol_under_over_model
        base_path = 'lol_under_over_model/'
    else:
        raise FileNotFoundError("Arquivos do modelo não encontrados. Certifique-se de estar na pasta correta.")
    
    print(f"📁 Carregando modelo do caminho: {base_path or 'pasta atual'}")
    
    with open(f'{base_path}trained_models.pkl', 'rb') as f:
        models = pickle.load(f)
    with open(f'{base_path}scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{base_path}champion_impacts.pkl', 'rb') as f:
        champion_impacts = pickle.load(f)
    with open(f'{base_path}league_stats.pkl', 'rb') as f:
        league_stats = pickle.load(f)
    with open(f'{base_path}feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    print("✅ Modelo carregado com sucesso!")
    return models, scaler, champion_impacts, league_stats, feature_cols

def prever_jogo(jogo_data):
    """
    Função principal para predição
    
    Input: dicionário com as chaves:
    - league: str (ex: 'LPL')
    - bet_line: float (ex: 28.5)
    - top_t1, jung_t1, mid_t1, adc_t1, sup_t1: str (campeões time 1)
    - top_t2, jung_t2, mid_t2, adc_t2, sup_t2: str (campeões time 2)
    
    Output: dicionário com recomendação de aposta
    """
    
    # Carregar modelo
    models, scaler, champion_impacts, league_stats, feature_cols = carregar_modelo()
    
    # Winrates esperados por categoria (baseado na análise real)
    winrates = {
        'BOA': {'UNDER': 58.6, 'OVER': 61.0},
        'MUITO_BOA': {'UNDER': 67.4, 'OVER': 65.7},
        'EXCELENTE': {'UNDER': 73.0, 'OVER': 65.4}
    }
    
    # Criar features
    league = jogo_data['league']
    league_avg = league_stats.get(league, 28.5)
    
    positions_t1 = ['top_t1', 'jung_t1', 'mid_t1', 'adc_t1', 'sup_t1']
    positions_t2 = ['top_t2', 'jung_t2', 'mid_t2', 'adc_t2', 'sup_t2']
    
    team1_impacts = [champion_impacts.get(league, {}).get(jogo_data[pos], 0.0) for pos in positions_t1]
    team2_impacts = [champion_impacts.get(league, {}).get(jogo_data[pos], 0.0) for pos in positions_t2]
    
    features = {
        'league_encoded': hash(league) % 100,
        'mean_league_kills': league_avg,
        'std_league_kills': 7.99,
        'mean_impact_team1': np.mean(team1_impacts),
        'mean_impact_team2': np.mean(team2_impacts),
        'total_impact': np.mean(team1_impacts) + np.mean(team2_impacts),
        'impact_diff': np.mean(team1_impacts) - np.mean(team2_impacts),
    }
    
    for i, impact in enumerate(team1_impacts):
        features[f'impact_t1_pos{i+1}'] = impact
    for i, impact in enumerate(team2_impacts):
        features[f'impact_t2_pos{i+1}'] = impact
    
    # Processar features
    features_df = pd.DataFrame([features])[feature_cols]
    features_scaled = scaler.transform(features_df)
    
    # Encontrar modelo
    bet_line = jogo_data['bet_line']
    if bet_line not in models:
        bet_line_original = bet_line
        bet_line = min(models.keys(), key=lambda x: abs(x - bet_line))
        print(f"⚠️ Linha {bet_line_original} não encontrada, usando modelo da linha {bet_line}")
    
    # Fazer predição
    prob_under = models[bet_line].predict_proba(features_scaled)[0, 1]
    prob_over = 1 - prob_under
    
    # Classificar apostas
    def classificar(prob):
        if prob >= 0.75: return 'EXCELENTE'
        elif prob >= 0.65: return 'MUITO_BOA'
        elif prob >= 0.55: return 'BOA'
        return None
    
    resultado = {
        'prob_under': prob_under,
        'prob_over': prob_over,
        'recomendacao': None
    }
    
    # Verificar UNDER
    if prob_under > 0.55:
        categoria = classificar(prob_under)
        if categoria:
            roi_under = (winrates[categoria]['UNDER']/100 * 0.83) - ((1 - winrates[categoria]['UNDER']/100) * 1.00)
            resultado['recomendacao'] = {
                'tipo': 'UNDER',
                'categoria': categoria,
                'probabilidade': prob_under,
                'winrate_esperado': winrates[categoria]['UNDER'],
                'roi_esperado': roi_under
            }
    
    # Verificar OVER (se melhor que UNDER)
    if prob_over > 0.55:
        categoria = classificar(prob_over)
        if categoria:
            roi_over = (winrates[categoria]['OVER']/100 * 0.83) - ((1 - winrates[categoria]['OVER']/100) * 1.00)
            if resultado['recomendacao'] is None or roi_over > resultado['recomendacao']['roi_esperado']:
                resultado['recomendacao'] = {
                    'tipo': 'OVER',
                    'categoria': categoria,
                    'probabilidade': prob_over,
                    'winrate_esperado': winrates[categoria]['OVER'],
                    'roi_esperado': roi_over
                }
    
    return resultado

def validar_input(jogo_data):
    """Valida se o input está no formato correto"""
    campos_obrigatorios = [
        'league', 'bet_line', 
        'top_t1', 'jung_t1', 'mid_t1', 'adc_t1', 'sup_t1',
        'top_t2', 'jung_t2', 'mid_t2', 'adc_t2', 'sup_t2'
    ]
    
    for campo in campos_obrigatorios:
        if campo not in jogo_data:
            raise ValueError(f"Campo obrigatório '{campo}' não encontrado no input")
    
    if not isinstance(jogo_data['bet_line'], (int, float)):
        raise ValueError("Campo 'bet_line' deve ser um número")
    
    print("✅ Input validado com sucesso!")

# Exemplo de uso
if __name__ == "__main__":
    print("🎯 PREDIÇÃO SIMPLES - EXEMPLO (CAMINHOS CORRIGIDOS)")
    print("=" * 50)
    
    # Exemplo de jogo
    jogo_exemplo = {
        'league': 'LPL',
        'bet_line': 29.5,
        'top_t1': 'Aatrox', 'jung_t1': 'Graves', 'mid_t1': 'Azir', 'adc_t1': 'Jinx', 'sup_t1': 'Thresh',
        'top_t2': 'Gnar', 'jung_t2': 'Sejuani', 'mid_t2': 'Orianna', 'adc_t2': 'Aphelios', 'sup_t2': 'Braum'
    }
    
    print("📊 Input:")
    print(f"  Liga: {jogo_exemplo['league']}")
    print(f"  Linha: {jogo_exemplo['bet_line']}")
    print(f"  Time 1: {jogo_exemplo['top_t1']}, {jogo_exemplo['jung_t1']}, {jogo_exemplo['mid_t1']}, {jogo_exemplo['adc_t1']}, {jogo_exemplo['sup_t1']}")
    print(f"  Time 2: {jogo_exemplo['top_t2']}, {jogo_exemplo['jung_t2']}, {jogo_exemplo['mid_t2']}, {jogo_exemplo['adc_t2']}, {jogo_exemplo['sup_t2']}")
    
    try:
        # Validar input
        validar_input(jogo_exemplo)
        
        # Fazer predição
        resultado = prever_jogo(jogo_exemplo)
        
        print(f"\n📈 Output:")
        print(f"  Prob UNDER: {resultado['prob_under']:.1%}")
        print(f"  Prob OVER: {resultado['prob_over']:.1%}")
        
        if resultado['recomendacao']:
            rec = resultado['recomendacao']
            
            # Emojis por categoria
            emojis = {'BOA': '🟢', 'MUITO_BOA': '🔵', 'EXCELENTE': '🟡'}
            emoji_categoria = emojis.get(rec['categoria'], '⚪')
            
            print(f"\n🎯 RECOMENDAÇÃO:")
            print(f"  ✅ Apostar {rec['tipo']}")
            print(f"  {emoji_categoria} Categoria: {rec['categoria']}")
            print(f"  📊 Probabilidade: {rec['probabilidade']:.1%}")
            print(f"  🎯 Winrate esperado: {rec['winrate_esperado']:.1f}%")
            print(f"  💰 ROI esperado: {rec['roi_esperado']*100:+.1f}%")
            
            # Nível de confiança
            if rec['probabilidade'] >= 0.75:
                confianca = "ALTA 🔥"
            elif rec['probabilidade'] >= 0.65:
                confianca = "MÉDIA ⚡"
            else:
                confianca = "BAIXA 💡"
            
            print(f"  🎖️ Confiança: {confianca}")
        else:
            print(f"\n❌ Nenhuma aposta recomendada (probabilidades < 55%)")
        
        print(f"\n✅ Predição concluída!")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("💡 Verifique se todos os arquivos do modelo estão na pasta correta")
