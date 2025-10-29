#!/usr/bin/env python3
"""
Script de Exemplo para Uso em Produção - CAMINHOS CORRIGIDOS
Mostra formato de entrada, processamento e saída completa
"""

import pickle
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def carregar_modelo_completo():
    """Carrega todos os componentes do modelo com caminhos flexíveis"""
    
    # Detectar se estamos dentro da pasta do modelo ou fora
    if os.path.exists('trained_models.pkl'):
        # Executando de dentro da pasta lol_under_over_model
        base_path = ''
    elif os.path.exists('lol_under_over_model/trained_models.pkl'):
        # Executando de fora da pasta lol_under_over_model
        base_path = 'lol_under_over_model/'
    else:
        raise FileNotFoundError("Arquivos do modelo não encontrados. Certifique-se de estar na pasta correta.")
    
    print(f"🔄 Carregando modelo LoL UNDER/OVER do caminho: {base_path or 'pasta atual'}")
    
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

def carregar_winrates_esperados():
    """Carrega winrates esperados por subcategoria baseado na análise"""
    # Baseado na análise detalhada realizada
    winrates_esperados = {
        'BOA': {
            'UNDER': 0.586,  # 58.6%
            'OVER': 0.610    # 61.0%
        },
        'MUITO_BOA': {
            'UNDER': 0.674,  # 67.4%
            'OVER': 0.657    # 65.7%
        },
        'EXCELENTE': {
            'UNDER': 0.730,  # 73.0%
            'OVER': 0.654    # 65.4%
        }
    }
    return winrates_esperados

def mostrar_formato_entrada():
    """Mostra o formato exato de entrada dos dados"""
    print("📋 FORMATO DE ENTRADA DOS DADOS")
    print("=" * 50)
    print("O input deve ser um dicionário Python com as seguintes chaves OBRIGATÓRIAS:")
    print()
    
    formato_exemplo = {
        'league': 'str',           # Liga do jogo (ex: 'LPL', 'LCK', 'LEC')
        'bet_line': 'float',       # Linha de aposta oferecida (ex: 28.5)
        'top_t1': 'str',          # Campeão top do time 1
        'jung_t1': 'str',         # Campeão jungle do time 1
        'mid_t1': 'str',          # Campeão mid do time 1
        'adc_t1': 'str',          # Campeão ADC do time 1
        'sup_t1': 'str',          # Campeão support do time 1
        'top_t2': 'str',          # Campeão top do time 2
        'jung_t2': 'str',         # Campeão jungle do time 2
        'mid_t2': 'str',          # Campeão mid do time 2
        'adc_t2': 'str',          # Campeão ADC do time 2
        'sup_t2': 'str'           # Campeão support do time 2
    }
    
    print("📝 ESTRUTURA:")
    for chave, tipo in formato_exemplo.items():
        print(f"  '{chave}': {tipo}")
    
    print(f"\n⚠️ IMPORTANTE:")
    print(f"- Todas as 12 chaves são OBRIGATÓRIAS")
    print(f"- Nomes dos campeões devem estar corretos (case-sensitive)")
    print(f"- Liga deve existir no dataset de treinamento")
    print(f"- bet_line deve ser um número (float)")

def criar_features_jogo(jogo_data, champion_impacts, league_stats):
    """Cria features para um jogo específico"""
    league = jogo_data['league']
    league_avg = league_stats.get(league, 28.5)
    league_std = 7.99
    
    positions_t1 = ['top_t1', 'jung_t1', 'mid_t1', 'adc_t1', 'sup_t1']
    positions_t2 = ['top_t2', 'jung_t2', 'mid_t2', 'adc_t2', 'sup_t2']
    
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
    
    features = {
        'league_encoded': hash(league) % 100,
        'mean_league_kills': league_avg,
        'std_league_kills': league_std,
        'mean_impact_team1': np.mean(team1_impacts),
        'mean_impact_team2': np.mean(team2_impacts),
        'total_impact': np.mean(team1_impacts) + np.mean(team2_impacts),
        'impact_diff': np.mean(team1_impacts) - np.mean(team2_impacts),
    }
    
    for i, impact in enumerate(team1_impacts):
        features[f'impact_t1_pos{i+1}'] = impact
    
    for i, impact in enumerate(team2_impacts):
        features[f'impact_t2_pos{i+1}'] = impact
    
    return features

def classificar_aposta(probabilidade):
    """Classifica aposta baseada na probabilidade"""
    if 0.55 <= probabilidade < 0.65:
        return 'BOA', '🟢'
    elif 0.65 <= probabilidade < 0.75:
        return 'MUITO_BOA', '🔵'
    elif probabilidade >= 0.75:
        return 'EXCELENTE', '🟡'
    else:
        return None, None

def analisar_jogo(jogo_data, models, scaler, champion_impacts, league_stats, feature_cols, winrates_esperados):
    """Analisa um jogo e retorna recomendações de aposta"""
    print(f"\n🎮 ANALISANDO JOGO")
    print("=" * 40)
    
    # Mostrar dados de entrada
    print(f"🌍 Liga: {jogo_data['league']}")
    print(f"📏 Linha de aposta: {jogo_data['bet_line']}")
    print(f"🔵 Time 1: {jogo_data['top_t1']}, {jogo_data['jung_t1']}, {jogo_data['mid_t1']}, {jogo_data['adc_t1']}, {jogo_data['sup_t1']}")
    print(f"🔴 Time 2: {jogo_data['top_t2']}, {jogo_data['jung_t2']}, {jogo_data['mid_t2']}, {jogo_data['adc_t2']}, {jogo_data['sup_t2']}")
    
    # Criar features
    features = criar_features_jogo(jogo_data, champion_impacts, league_stats)
    features_df = pd.DataFrame([features])
    features_df = features_df[feature_cols]
    features_scaled = scaler.transform(features_df)
    
    # Encontrar modelo para a linha de aposta
    bet_line = jogo_data['bet_line']
    if bet_line not in models:
        linhas_disponiveis = list(models.keys())
        bet_line_proxima = min(linhas_disponiveis, key=lambda x: abs(x - bet_line))
        modelo = models[bet_line_proxima]
        print(f"⚠️ Linha {bet_line} não encontrada, usando modelo da linha {bet_line_proxima}")
    else:
        modelo = models[bet_line]
        print(f"✅ Modelo encontrado para linha {bet_line}")
    
    # Fazer predição
    prob_under = modelo.predict_proba(features_scaled)[0, 1]
    prob_over = 1 - prob_under
    
    print(f"\n📊 PROBABILIDADES DO MODELO:")
    print(f"  🔽 UNDER: {prob_under:.3f} ({prob_under:.1%})")
    print(f"  🔼 OVER: {prob_over:.3f} ({prob_over:.1%})")
    
    # Analisar apostas possíveis
    apostas_recomendadas = []
    
    # Verificar UNDER
    if prob_under > 0.55:
        categoria, cor = classificar_aposta(prob_under)
        if categoria:
            winrate_esperado = winrates_esperados[categoria]['UNDER']
            apostas_recomendadas.append({
                'tipo': 'UNDER',
                'probabilidade': prob_under,
                'categoria': categoria,
                'cor': cor,
                'winrate_esperado': winrate_esperado,
                'emoji': '🔽'
            })
    
    # Verificar OVER
    if prob_over > 0.55:
        categoria, cor = classificar_aposta(prob_over)
        if categoria:
            winrate_esperado = winrates_esperados[categoria]['OVER']
            apostas_recomendadas.append({
                'tipo': 'OVER',
                'probabilidade': prob_over,
                'categoria': categoria,
                'cor': cor,
                'winrate_esperado': winrate_esperado,
                'emoji': '🔼'
            })
    
    # Mostrar resultados
    print(f"\n🎯 ANÁLISE DE APOSTAS:")
    print("=" * 40)
    
    if not apostas_recomendadas:
        print("❌ NENHUMA APOSTA RECOMENDADA")
        print("   Probabilidades abaixo do threshold mínimo (0.55)")
        return None
    
    print(f"✅ {len(apostas_recomendadas)} APOSTA(S) RECOMENDADA(S):")
    print()
    
    for i, aposta in enumerate(apostas_recomendadas, 1):
        print(f"{i}. {aposta['emoji']} APOSTAR {aposta['tipo']} na linha {bet_line}")
        print(f"   {aposta['cor']} Classificação: {aposta['categoria']}")
        print(f"   📊 Probabilidade do modelo: {aposta['probabilidade']:.1%}")
        print(f"   🎯 Winrate esperado: {aposta['winrate_esperado']:.1%}")
        
        # Calcular ROI esperado baseado no winrate
        roi_esperado = (aposta['winrate_esperado'] * 0.83) - ((1 - aposta['winrate_esperado']) * 1.00)
        roi_percentual = (roi_esperado * 100)
        
        print(f"   📈 ROI esperado: {roi_percentual:+.1f}%")
        
        # Nível de confiança
        if aposta['probabilidade'] >= 0.75:
            confianca = "ALTA 🔥"
        elif aposta['probabilidade'] >= 0.65:
            confianca = "MÉDIA ⚡"
        else:
            confianca = "BAIXA 💡"
        
        print(f"   🎖️ Nível de confiança: {confianca}")
        print()
    
    # Recomendação final
    if len(apostas_recomendadas) == 1:
        melhor_aposta = apostas_recomendadas[0]
        print(f"🏆 RECOMENDAÇÃO FINAL:")
        print(f"   Apostar {melhor_aposta['emoji']} {melhor_aposta['tipo']} - {melhor_aposta['cor']} {melhor_aposta['categoria']}")
    else:
        # Se há múltiplas apostas, recomendar a de maior ROI
        melhor_aposta = max(apostas_recomendadas, key=lambda x: x['winrate_esperado'])
        print(f"🏆 RECOMENDAÇÃO PRINCIPAL:")
        print(f"   Apostar {melhor_aposta['emoji']} {melhor_aposta['tipo']} - {melhor_aposta['cor']} {melhor_aposta['categoria']}")
        print(f"   (Maior winrate esperado: {melhor_aposta['winrate_esperado']:.1%})")
    
    return apostas_recomendadas

def exemplo_jogo_lpl():
    """Exemplo de jogo da LPL"""
    return {
        'league': 'LPL',
        'bet_line': 29.5,
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
    """Exemplo de jogo da LCK"""
    return {
        'league': 'LCK',
        'bet_line': 26.5,
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
    """Exemplo de jogo da LEC"""
    return {
        'league': 'LEC',
        'bet_line': 32.5,
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

def mostrar_informacoes_modelo():
    """Mostra informações sobre o modelo e classificações"""
    print("🤖 INFORMAÇÕES DO MODELO")
    print("=" * 40)
    print("📊 Performance: ROC-AUC ~0.746")
    print("🎯 Break-even: 54.6% winrate")
    print("💰 Odds: Vitória +0.83, Derrota -1.00")
    print("📈 Resultado: 100% das combinações lucrativas")
    print("🎮 Dataset: 5.320 jogos + 389 jogos reais testados")
    print()
    
    print("🎯 SISTEMA DE CLASSIFICAÇÃO:")
    print("🟢 BOA (0.55-0.64): Winrate 58.6%-61.0%, ROI 7.3%-11.7%")
    print("🔵 MUITO BOA (0.65-0.74): Winrate 65.7%-67.4%, ROI 20.3%-23.4%")
    print("🟡 EXCELENTE (≥0.75): Winrate 65.4%-73.0%, ROI 19.7%-33.5%")
    print()
    
    print("📋 COMO INTERPRETAR:")
    print("- Probabilidade: Confiança do modelo na predição")
    print("- Classificação: Qualidade da oportunidade de aposta")
    print("- Winrate esperado: Taxa de acerto histórica da categoria")
    print("- ROI esperado: Retorno médio por aposta da categoria")

def main():
    """Função principal - demonstração completa"""
    print("🎯 MODELO LOL UNDER/OVER - EXEMPLO DE USO EM PRODUÇÃO")
    print("=" * 70)
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. Mostrar informações do modelo
        mostrar_informacoes_modelo()
        
        # 2. Mostrar formato de entrada
        mostrar_formato_entrada()
        
        # 3. Carregar modelo e dados
        models, scaler, champion_impacts, league_stats, feature_cols = carregar_modelo_completo()
        winrates_esperados = carregar_winrates_esperados()
        
        # 4. Exemplos práticos
        print(f"\n🎮 EXEMPLOS PRÁTICOS")
        print("=" * 30)
        
        exemplos = [
            ("JDG vs BLG (LPL)", exemplo_jogo_lpl()),
            ("T1 vs GenG (LCK)", exemplo_jogo_lck()),
            ("G2 vs FNC (LEC)", exemplo_jogo_lec())
        ]
        
        for nome_jogo, jogo_data in exemplos:
            print(f"\n" + "="*60)
            print(f"🎮 EXEMPLO: {nome_jogo}")
            print("="*60)
            
            apostas = analisar_jogo(jogo_data, models, scaler, champion_impacts, 
                                   league_stats, feature_cols, winrates_esperados)
        
        # 5. Instruções de uso
        print(f"\n💡 COMO USAR EM PRODUÇÃO:")
        print("=" * 40)
        print("1. Prepare os dados no formato mostrado acima")
        print("2. Chame a função analisar_jogo() com seus dados")
        print("3. Interprete a saída para tomar decisão de aposta")
        print("4. Considere apenas apostas classificadas (BOA+)")
        print("5. Gerencie sua banca baseado no ROI esperado")
        
        print(f"\n✅ DEMONSTRAÇÃO CONCLUÍDA!")
        print("🚀 Modelo pronto para uso em produção!")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("💡 Verifique se todos os arquivos do modelo estão na pasta correta")
        print("📁 Execute este script de dentro da pasta 'lol_under_over_model' ou")
        print("📁 certifique-se de que a pasta 'lol_under_over_model' está no diretório atual")

if __name__ == "__main__":
    main()
