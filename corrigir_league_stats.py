import pickle

CAMINHO = "lol_under_over_model/league_stats_v2.pkl"

with open(CAMINHO, "rb") as f:
    data = pickle.load(f)

print("Antes:", data)

corrigido = {}
for k, v in data.items():
    if isinstance(v, dict) and "mean_kills" in v:
        corrigido[k] = v
    else:
        corrigido[k] = {"mean_kills": float(v)}

with open(CAMINHO, "wb") as f:
    pickle.dump(corrigido, f)

print("✅ Corrigido com sucesso!")
print("Depois:", corrigido)
