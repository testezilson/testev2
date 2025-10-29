# main_v2.py
import pickle
import json
import numpy as np
import pandas as pd

def load_components():
    with open('lol_under_over_model/trained_models_v2.pkl', 'rb') as f:
        models = pickle.load(f)
    with open('lol_under_over_model/scaler_v2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('lol_under_over_model/champion_impacts_v2.pkl', 'rb') as f:
        champion_impacts = pickle.load(f)
    with open('lol_under_over_model/league_stats_v2.pkl', 'rb') as f:
        league_stats = pickle.load(f)
    with open('lol_under_over_model/feature_columns_v2.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return models, scaler, champion_impacts, league_stats, feature_cols


def predict_game(game_data, models, scaler, champion_impacts, league_stats, feature_cols, threshold=0.65):
    from lol_under_over_model.load_and_predict_v2 import predict_game as predict_core
    return predict_core(game_data, models, scaler, champion_impacts, league_stats, feature_cols, threshold)


def default_serializer(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray, list)):
        return obj.tolist()
    raise TypeError(f"Tipo não serializável: {type(obj)}")


if __name__ == "__main__":
    models, scaler, champion_impacts, league_stats, feature_cols = load_components()

    game_example = {
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

    predictions = predict_game(
        game_example,
        models,
        scaler,
        champion_impacts,
        league_stats,
        feature_cols,
        threshold=0.65
    )

    print(json.dumps(predictions, indent=2, default=default_serializer))
