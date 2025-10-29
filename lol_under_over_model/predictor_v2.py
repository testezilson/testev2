#!/usr/bin/env python3
"""
Predictor.py - Modelo LoL UNDER/OVER
Formato de saída padronizado para uso futuro
"""

import pickle
import pandas as pd
import numpy as np
import os

class LoLPredictor:
    """Classe principal para predições LoL UNDER/OVER"""
    
    def __init__(self):
        """Inicializa o predictor carregando o modelo"""
        self.models = None
        self.scaler = None
        self.champion_impacts = None
        self.league_stats = None
        self.feature_cols = None
        self.winrates = {
            'BOA': {'UNDER': 58.6, 'OVER': 61.0},
            'MUITO_BOA': {'UNDER': 67.4, 'OVER': 65.7},
            'EXCELENTE': {'UNDER': 73.0, 'OVER': 65.4}
        }
        self._carregar_modelo()
    
    def _carregar_modelo(self):
        """Carrega todos os componentes do modelo"""
        # Detectar caminho dos arquivos
        if os.path.exists('trained_models.pkl'):
            base_path = ''
        elif os.path.exists('lol_under_over_model/trained_models.pkl'):
            base_path = 'lol_under_over_model/'
        else:
            raise FileNotFoundError("Arquivos do modelo não encontrados")
        
        # Carregar componentes
        with open(f'{base_path}trained_models.pkl', 'rb') as f:
            self.models = pickle.load(f)
        with open(f'{base_path}scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open(f'{base_path}champion_impacts.pkl', 'rb') as f:
            self.champion_impacts = pickle.load(f)
        with open(f'{base_path}league_stats.pkl', 'rb') as f:
            self.league_stats = pickle.load(f)
        with open(f'{base_path}feature_columns.pkl', 'rb') as f:
            self.feature_cols = pickle.load(f)
    
    def _criar_features(self, jogo_data):
        """Cria features para um jogo"""
        league = jogo_data['league']
        league_avg = self.league_stats.get(league, 28.5)
        
        positions_t1 = ['top_t1', 'jung_t1', 'mid_t1', 'adc_t1', 'sup_t1']
        positions_t2 = ['top_t2', 'jung_t2', 'mid_t2', 'adc_t2', 'sup_t2']
        
        team1_impacts = [self.champion_impacts.get(league, {}).get(jogo_data[pos], 0.0) for pos in positions_t1]
        team2_impacts = [self.champion_impacts.get(league, {}).get(jogo_data[pos], 0.0) for pos in positions_t2]
        
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
        
        return features
    
    def _classificar_probabilidade(self, prob):
        """Classifica probabilidade em categoria"""
        if prob >= 0.75:
            return 'EXCELENTE'
        elif prob >= 0.65:
            return 'MUITO_BOA'
        elif prob >= 0.55:
            return 'BOA'
        return None
    
    def predict(self, jogo_data):
        """
        Faz predição para um jogo
        
        Args:
            jogo_data (dict): Dados do jogo com chaves:
                - league: str
                - bet_line: float
                - top_t1, jung_t1, mid_t1, adc_t1, sup_t1: str
                - top_t2, jung_t2, mid_t2, adc_t2, sup_t2: str
        
        Returns:
            dict: Resultado da predição no formato padronizado
        """
        
        # Validar input
        campos_obrigatorios = [
            'league', 'bet_line', 
            'top_t1', 'jung_t1', 'mid_t1', 'adc_t1', 'sup_t1',
            'top_t2', 'jung_t2', 'mid_t2', 'adc_t2', 'sup_t2'
        ]
        
        for campo in campos_obrigatorios:
            if campo not in jogo_data:
                raise ValueError(f"Campo obrigatório '{campo}' não encontrado")
        
        # Criar features
        features = self._criar_features(jogo_data)
        features_df = pd.DataFrame([features])[self.feature_cols]
        features_scaled = self.scaler.transform(features_df)
        
        # Encontrar modelo
        bet_line = jogo_data['bet_line']
        if bet_line not in self.models:
            bet_line = min(self.models.keys(), key=lambda x: abs(x - bet_line))
        
        # Fazer predição
        prob_under = self.models[bet_line].predict_proba(features_scaled)[0, 1]
        prob_over = 1 - prob_under
        
        # Analisar apostas
        recomendacao_under = None
        recomendacao_over = None
        
        # Verificar UNDER
        if prob_under > 0.55:
            categoria = self._classificar_probabilidade(prob_under)
            if categoria:
                winrate = self.winrates[categoria]['UNDER']
                roi = (winrate/100 * 0.83) - ((1 - winrate/100) * 1.00)
                
                recomendacao_under = {
                    'tipo': 'UNDER',
                    'categoria': categoria,
                    'probabilidade': prob_under,
                    'winrate_esperado': winrate,
                    'roi_esperado': roi * 100
                }
        
        # Verificar OVER
        if prob_over > 0.55:
            categoria = self._classificar_probabilidade(prob_over)
            if categoria:
                winrate = self.winrates[categoria]['OVER']
                roi = (winrate/100 * 0.83) - ((1 - winrate/100) * 1.00)
                
                recomendacao_over = {
                    'tipo': 'OVER',
                    'categoria': categoria,
                    'probabilidade': prob_over,
                    'winrate_esperado': winrate,
                    'roi_esperado': roi * 100
                }
        
        # Escolher melhor recomendação
        melhor_recomendacao = None
        if recomendacao_under and recomendacao_over:
            # Escolher por maior ROI
            if recomendacao_under['roi_esperado'] >= recomendacao_over['roi_esperado']:
                melhor_recomendacao = recomendacao_under
            else:
                melhor_recomendacao = recomendacao_over
        elif recomendacao_under:
            melhor_recomendacao = recomendacao_under
        elif recomendacao_over:
            melhor_recomendacao = recomendacao_over
        
        # Formato de saída padronizado
        resultado = {
            'input': {
                'league': jogo_data['league'],
                'bet_line': jogo_data['bet_line'],
                'team1': f"{jogo_data['top_t1']}, {jogo_data['jung_t1']}, {jogo_data['mid_t1']}, {jogo_data['adc_t1']}, {jogo_data['sup_t1']}",
                'team2': f"{jogo_data['top_t2']}, {jogo_data['jung_t2']}, {jogo_data['mid_t2']}, {jogo_data['adc_t2']}, {jogo_data['sup_t2']}"
            },
            'probabilities': {
                'under': round(prob_under, 3),
                'over': round(prob_over, 3)
            },
            'recommendation': melhor_recomendacao,
            'all_options': {
                'under': recomendacao_under,
                'over': recomendacao_over
            }
        }
        
        return resultado
    
    def predict_and_print(self, jogo_data):
        """
        Faz predição e imprime resultado formatado
        
        Args:
            jogo_data (dict): Dados do jogo
        
        Returns:
            dict: Resultado da predição
        """
        resultado = self.predict(jogo_data)
        
        print("📊 Input:")
        print(f"  Liga: {resultado['input']['league']}")
        print(f"  Linha: {resultado['input']['bet_line']}")
        print(f"  Time 1: {resultado['input']['team1']}")
        print(f"  Time 2: {resultado['input']['team2']}")
        
        print(f"\n📈 Output:")
        print(f"  Prob UNDER: {resultado['probabilities']['under']:.1%}")
        print(f"  Prob OVER: {resultado['probabilities']['over']:.1%}")
        
        if resultado['recommendation']:
            rec = resultado['recommendation']
            emojis = {'BOA': '🟢', 'MUITO_BOA': '🔵', 'EXCELENTE': '🟡'}
            emoji_categoria = emojis.get(rec['categoria'], '⚪')
            
            print(f"\n🎯 RECOMENDAÇÃO:")
            print(f"  ✅ Apostar {rec['tipo']}")
            print(f"  {emoji_categoria} Categoria: {rec['categoria']}")
            print(f"  📊 Probabilidade: {rec['probabilidade']:.1%}")
            print(f"  🎯 Winrate esperado: {rec['winrate_esperado']:.1f}%")
            print(f"  💰 ROI esperado: {rec['roi_esperado']:+.1f}%")
            
            if rec['probabilidade'] >= 0.75:
                confianca = "ALTA 🔥"
            elif rec['probabilidade'] >= 0.65:
                confianca = "MÉDIA ⚡"
            else:
                confianca = "BAIXA 💡"
            
            print(f"  🎖️ Confiança: {confianca}")
        else:
            print(f"\n❌ Nenhuma aposta recomendada (probabilidades < 55%)")
        
        return resultado

# Função de conveniência para uso direto
def prever_jogo(jogo_data):
    """
    Função de conveniência para fazer predição
    
    Args:
        jogo_data (dict): Dados do jogo
    
    Returns:
        dict: Resultado da predição
    """
    predictor = LoLPredictor()
    return predictor.predict(jogo_data)

# Exemplo de uso
if __name__ == "__main__":
    print("🎯 PREDICTOR.PY - MODELO LOL UNDER/OVER")
    print("=" * 50)
    
    # Exemplo de jogo
    jogo_exemplo = {
        'league': 'LPL',
        'bet_line': 29.5,
        'top_t1': 'Aatrox', 'jung_t1': 'Graves', 'mid_t1': 'Azir', 
        'adc_t1': 'Jinx', 'sup_t1': 'Thresh',
        'top_t2': 'Gnar', 'jung_t2': 'Sejuani', 'mid_t2': 'Orianna', 
        'adc_t2': 'Aphelios', 'sup_t2': 'Braum'
    }
    
    try:
        # Criar predictor
        predictor = LoLPredictor()
        
        # Fazer predição com saída formatada
        resultado = predictor.predict_and_print(jogo_exemplo)
        
        print(f"\n📋 FORMATO DE SAÍDA PADRONIZADO:")
        print(f"  - Input: Liga, linha, times")
        print(f"  - Probabilities: UNDER e OVER")
        print(f"  - Recommendation: Melhor aposta")
        print(f"  - All_options: Todas as opções disponíveis")
        
        print(f"\n✅ Predictor pronto para uso!")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("💡 Certifique-se de que os arquivos do modelo estão disponíveis")
