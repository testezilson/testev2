import pandas as pd
import numpy as np


def analisar_jogos_annie(arquivo_csv):
    df = pd.read_csv(arquivo_csv)

    # Posições onde Annie pode aparecer
    posicoes = [
        "top_t1",
        "jung_t1",
        "mid_t1",
        "adc_t1",
        "sup_t1",
        "top_t2",
        "jung_t2",
        "mid_t2",
        "adc_t2",
        "sup_t2",
    ]

    # Filtrar jogos com Annie
    mask_annie = df[posicoes].eq("Annie").any(axis=1)
    jogos_annie = df[mask_annie]

    # Análise geral de todos os jogos
    total_kills_geral = df["total_kills"]

    # Análise de campeões mid com mais de 50 jogos
    mid_champs = pd.concat([df["mid_t1"], df["mid_t2"]])
    ranking_mid = mid_champs.value_counts()

    # Filtrar apenas campeões com mais de 50 jogos
    mid_50plus = ranking_mid[ranking_mid > 50]

    # Total kills por campeão mid (só +50 jogos)
    mid_kills = []
    for champ in mid_50plus.index:
        mask_mid = (df["mid_t1"] == champ) | (df["mid_t2"] == champ)
        kills_champ = df[mask_mid]["total_kills"]
        mid_kills.append(
            {
                "campeao": champ,
                "jogos": len(kills_champ),
                "media_kills": kills_champ.mean(),
                "min_kills": kills_champ.min(),
                "max_kills": kills_champ.max(),
            }
        )

    mid_stats = pd.DataFrame(mid_kills).sort_values("media_kills", ascending=False)

    resultado = {
        "annie_stats": {
            "total_jogos": len(jogos_annie),
            "media": jogos_annie["total_kills"].mean() if len(jogos_annie) > 0 else 0,
            "minimo": jogos_annie["total_kills"].min() if len(jogos_annie) > 0 else 0,
            "maximo": jogos_annie["total_kills"].max() if len(jogos_annie) > 0 else 0,
            "variancia": jogos_annie["total_kills"].var()
            if len(jogos_annie) > 0
            else 0,
        },
        "geral_stats": {
            "total_jogos": len(df),
            "media": total_kills_geral.mean(),
            "minimo": total_kills_geral.min(),
            "maximo": total_kills_geral.max(),
            "variancia": total_kills_geral.var(),
        },
        "mid_ranking": mid_stats,
        "annie_posicao_mid": None,
    }

    # Posição da Annie no ranking mid
    if "Annie" in mid_stats["campeao"].values:
        annie_pos = mid_stats[mid_stats["campeao"] == "Annie"].index[0] + 1
        resultado["annie_posicao_mid"] = annie_pos

    return resultado


resultado = analisar_jogos_annie("database_transformed.csv")

print("=== ANNIE STATS ===")
print(f"Jogos: {resultado['annie_stats']['total_jogos']}")
print(f"Média kills: {resultado['annie_stats']['media']:.2f}")
print(f"Min: {resultado['annie_stats']['minimo']}")
print(f"Max: {resultado['annie_stats']['maximo']}")
print(f"Variância: {resultado['annie_stats']['variancia']:.2f}")

print("\n=== COMPARAÇÃO GERAL ===")
print(f"Média geral todos os jogos: {resultado['geral_stats']['media']:.2f}")
print(
    f"Diferença Annie vs Geral: {resultado['annie_stats']['media'] - resultado['geral_stats']['media']:.2f}"
)

print(f"\n=== RANKING MID (+50 jogos) ===")
print(resultado["mid_ranking"])

if resultado["annie_posicao_mid"]:
    print(
        f"\nAnnie está na posição {resultado['annie_posicao_mid']} no ranking de mids (+50 jogos)"
    )
else:
    print(f"\nAnnie não está no ranking (menos de 50 jogos ou não jogada no mid)")
