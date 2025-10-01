# LoL UNDER/OVER Model Package

## Descrição
Modelo completo para previsão de apostas UNDER/OVER em League of Legends.

## Arquivos Incluídos

### Modelos e Componentes
- `trained_models.pkl`: Modelos treinados para cada linha de aposta (25.5 a 32.5)
- `scaler.pkl`: StandardScaler para normalização das features
- `champion_impacts.pkl`: Impactos calculados de cada campeão por liga
- `league_stats.pkl`: Estatísticas médias de cada liga
- `feature_columns.pkl`: Lista das colunas de features utilizadas

### Dados e Performance
- `original_dataset.csv`: Dataset original (5.320 jogos)
- `processed_features.csv`: Features processadas prontas para o modelo
- `model_performance.json`: Métricas de performance de cada modelo
- `threshold_classification.json`: Classificação personalizada dos thresholds

### Documentação
- `README.md`: Este arquivo de instruções
- `load_and_predict.py`: Script exemplo para carregar e usar o modelo

## Como Usar

1. Carregue os componentes:
```python
import pickle
import pandas as pd

# Carregar modelos
with open('trained_models.pkl', 'rb') as f:
    models = pickle.load(f)

# Carregar scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Carregar impactos
with open('champion_impacts.pkl', 'rb') as f:
    champion_impacts = pickle.load(f)
```

2. Processe novos dados usando as mesmas features
3. Faça predições para cada linha de aposta
4. Aplique os thresholds conforme a classificação escolhida

## Classificação dos Thresholds

🟢 **BOA (0.55, 0.60)**: Estratégia de Volume
🔵 **MUITO BOA (0.65, 0.70)**: Estratégia Equilibrada  
🟡 **EXCELENTE (0.75+)**: Estratégia Premium

## Métricas de Performance

- ROC-AUC médio: ~0.746
- Break-even: 54.6% winrate
- Todas as combinações são lucrativas
- Dataset: 5.320 jogos, Test: 1.064 jogos

## Data de Criação
2025-10-01 15:21:58
