#!/usr/bin/env python3
"""
Exemplos Adicionais para Teste do Modelo
2 novos cenários com diferentes características
"""

from predicao_simples import prever_jogo

def exemplo_1_lck_linha_baixa():
    """
    Exemplo 1: LCK com linha baixa (26.5)
    Cenário: Jogo com campeões que podem gerar poucas kills
    """
    print("🎮 EXEMPLO 1: T1 vs DRX (LCK) - Linha Baixa")
    print("=" * 50)
    
    jogo = {
        'league': 'LCK',
        'bet_line': 26.5,
        'top_t1': 'Ornn',        # Tank defensivo
        'jung_t1': 'Sejuani',    # Tank jungle
        'mid_t1': 'Galio',       # Utilitário
        'adc_t1': 'Ezreal',      # ADC seguro
        'sup_t1': 'Braum',       # Support defensivo
        'top_t2': 'Malphite',    # Tank
        'jung_t2': 'Ammu',       # Tank jungle
        'mid_t2': 'Twisted Fate', # Utilitário
        'adc_t2': 'Sivir',       # ADC utilitário
        'sup_t2': 'Alistar'      # Tank support
    }
    
    print("📊 Input:")
    print(f"  Liga: {jogo['league']}")
    print(f"  Linha: {jogo['bet_line']} (BAIXA - jogo defensivo esperado)")
    print(f"  Time 1: {jogo['top_t1']}, {jogo['jung_t1']}, {jogo['mid_t1']}, {jogo['adc_t1']}, {jogo['sup_t1']}")
    print(f"  Time 2: {jogo['top_t2']}, {jogo['jung_t2']}, {jogo['mid_t2']}, {jogo['adc_t2']}, {jogo['sup_t2']}")
    print(f"  💡 Composições defensivas com tanks e utilitários")
    
    resultado = prever_jogo(jogo)
    
    print(f"\n📈 Output:")
    print(f"  Prob UNDER: {resultado['prob_under']:.1%}")
    print(f"  Prob OVER: {resultado['prob_over']:.1%}")
    
    if resultado['recomendacao']:
        rec = resultado['recomendacao']
        emojis = {'BOA': '🟢', 'MUITO_BOA': '🔵', 'EXCELENTE': '🟡'}
        emoji_categoria = emojis.get(rec['categoria'], '⚪')
        
        print(f"\n🎯 RECOMENDAÇÃO:")
        print(f"  ✅ Apostar {rec['tipo']}")
        print(f"  {emoji_categoria} Categoria: {rec['categoria']}")
        print(f"  📊 Probabilidade: {rec['probabilidade']:.1%}")
        print(f"  🎯 Winrate esperado: {rec['winrate_esperado']:.1f}%")
        print(f"  💰 ROI esperado: {rec['roi_esperado']*100:+.1f}%")
        
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

def exemplo_2_lec_linha_alta():
    """
    Exemplo 2: LEC com linha alta (33.5)
    Cenário: Jogo com campeões agressivos que podem gerar muitas kills
    """
    print("\n\n🎮 EXEMPLO 2: G2 vs MAD Lions (LEC) - Linha Alta")
    print("=" * 50)
    
    jogo = {
        'league': 'LEC',
        'bet_line': 33.5,
        'top_t1': 'Fiora',       # Carry agressivo
        'jung_t1': 'Graves',     # Carry jungle
        'mid_t1': 'Yasuo',       # Assassino
        'adc_t1': 'Draven',      # ADC agressivo
        'sup_t1': 'Pyke',        # Support assassino
        'top_t2': 'Riven',       # Carry agressivo
        'jung_t2': 'Kha\'Zix',   # Assassino jungle
        'mid_t2': 'Zed',         # Assassino
        'adc_t2': 'Jinx',        # ADC com reset
        'sup_t2': 'Thresh'       # Support engage
    }
    
    print("📊 Input:")
    print(f"  Liga: {jogo['league']}")
    print(f"  Linha: {jogo['bet_line']} (ALTA - jogo agressivo esperado)")
    print(f"  Time 1: {jogo['top_t1']}, {jogo['jung_t1']}, {jogo['mid_t1']}, {jogo['adc_t1']}, {jogo['sup_t1']}")
    print(f"  Time 2: {jogo['top_t2']}, {jogo['jung_t2']}, {jogo['mid_t2']}, {jogo['adc_t2']}, {jogo['sup_t2']}")
    print(f"  💡 Composições agressivas com carries e assassinos")
    
    resultado = prever_jogo(jogo)
    
    print(f"\n📈 Output:")
    print(f"  Prob UNDER: {resultado['prob_under']:.1%}")
    print(f"  Prob OVER: {resultado['prob_over']:.1%}")
    
    if resultado['recomendacao']:
        rec = resultado['recomendacao']
        emojis = {'BOA': '🟢', 'MUITO_BOA': '🔵', 'EXCELENTE': '🟡'}
        emoji_categoria = emojis.get(rec['categoria'], '⚪')
        
        print(f"\n🎯 RECOMENDAÇÃO:")
        print(f"  ✅ Apostar {rec['tipo']}")
        print(f"  {emoji_categoria} Categoria: {rec['categoria']}")
        print(f"  📊 Probabilidade: {rec['probabilidade']:.1%}")
        print(f"  🎯 Winrate esperado: {rec['winrate_esperado']:.1f}%")
        print(f"  💰 ROI esperado: {rec['roi_esperado']*100:+.1f}%")
        
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

def comparar_exemplos(resultado1, resultado2):
    """Compara os dois exemplos"""
    print(f"\n📊 COMPARAÇÃO DOS EXEMPLOS")
    print("=" * 40)
    
    print(f"🎮 Exemplo 1 (LCK, linha 26.5, defensivo):")
    if resultado1['recomendacao']:
        rec1 = resultado1['recomendacao']
        print(f"  → {rec1['tipo']} {rec1['categoria']} ({rec1['probabilidade']:.1%})")
    else:
        print(f"  → Nenhuma aposta")
    
    print(f"🎮 Exemplo 2 (LEC, linha 33.5, agressivo):")
    if resultado2['recomendacao']:
        rec2 = resultado2['recomendacao']
        print(f"  → {rec2['tipo']} {rec2['categoria']} ({rec2['probabilidade']:.1%})")
    else:
        print(f"  → Nenhuma aposta")
    
    print(f"\n💡 INSIGHTS:")
    print(f"- Linha baixa + comp defensiva: Favorece UNDER")
    print(f"- Linha alta + comp agressiva: Pode favorecer OVER")
    print(f"- Modelo considera contexto da liga e campeões")
    print(f"- Diferentes cenários geram diferentes oportunidades")

def main():
    """Executa os exemplos"""
    print("🎯 EXEMPLOS ADICIONAIS DO MODELO LOL UNDER/OVER")
    print("=" * 60)
    
    try:
        # Exemplo 1: Cenário defensivo
        resultado1 = exemplo_1_lck_linha_baixa()
        
        # Exemplo 2: Cenário agressivo  
        resultado2 = exemplo_2_lec_linha_alta()
        
        # Comparação
        comparar_exemplos(resultado1, resultado2)
        
        print(f"\n✅ EXEMPLOS CONCLUÍDOS!")
        print(f"🎯 Teste diferentes combinações de liga, linha e campeões")
        print(f"📊 O modelo se adapta ao contexto de cada jogo")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("💡 Certifique-se de que o arquivo predicao_simples.py está na mesma pasta")

if __name__ == "__main__":
    main()
