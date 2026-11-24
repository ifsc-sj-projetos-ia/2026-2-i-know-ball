"""Features calculadas só com o que se sabe ANTES de cada jogo (sem vazamento)."""
import pandas as pd

# Valores de partida do Elo de NBA do FiveThirtyEight. Ajustar é etapa de experimento.
ELO_INICIAL = 1500
ELO_K = 20  # quanto um jogo mexe na nota
ELO_MANDO = 100  # bônus de nota do mandante (~64% de chance entre times iguais)
ELO_REVERSAO = 0.25  # na virada de temporada, a nota anda 25% de volta para a média


def prob_elo(diferenca):
    """Chance de vitória de quem tem `diferenca` pontos de Elo a mais. 400 de diferença = 10 para 1."""
    return 1 / (1 + 10 ** (-diferenca / 400))


def _multiplicador_saldo(saldo, diferenca_vencedor):
    # Lavada mexe mais na nota, mas favorito ganhando de muito é menos surpresa (por isso o desconto pela diferença).
    return (abs(saldo) + 3) ** 0.8 / (7.5 + 0.006 * diferenca_vencedor)


def calcular_elo(jogos):
    """Percorre os jogos em ordem de data e devolve, para cada um, o Elo de cada time ANTES do jogo."""
    notas = {}
    temporada_anterior = None
    linhas = []

    for jogo in jogos.itertuples():
        if jogo.temporada != temporada_anterior:
            notas = {t: n + ELO_REVERSAO * (ELO_INICIAL - n) for t, n in notas.items()}
            temporada_anterior = jogo.temporada

        elo_m = notas.get(jogo.mandante, ELO_INICIAL)
        elo_v = notas.get(jogo.visitante, ELO_INICIAL)
        linhas.append((elo_m, elo_v))  # guardado ANTES de atualizar: é o que se sabia na véspera

        diferenca = elo_m + ELO_MANDO - elo_v
        esperado = prob_elo(diferenca)
        diferenca_vencedor = diferenca if jogo.mandante_venceu else -diferenca
        ajuste = ELO_K * _multiplicador_saldo(jogo.saldo, diferenca_vencedor) * (jogo.mandante_venceu - esperado)
        notas[jogo.mandante] = elo_m + ajuste
        notas[jogo.visitante] = elo_v - ajuste

    elo = pd.DataFrame(linhas, columns=["elo_m", "elo_v"], index=jogos.index)
    elo["prob_elo_m"] = prob_elo(elo["elo_m"] + ELO_MANDO - elo["elo_v"])
    return elo, notas


JANELA = 10  # últimos N jogos para forma e eficiência (atravessa a virada de temporada)
DESCANSO_MAX = 3  # 3 dias ou mais de folga contam igual

COLUNAS_MODELO = ["dif_elo", "dif_saldo10", "dif_ataque10", "dif_defesa10",
                  "descanso_m", "descanso_v", "b2b_m", "b2b_v"]


def _por_time(jogos):
    """Transforma 1 linha por jogo em 1 linha por time por jogo, do ponto de vista de cada time."""
    # Posses estimadas: arremessos - rebote ofensivo + perda de bola + 0,44 x lances livres. Média dos dois lados.
    posses = sum(jogos[f"fga_{s}"] - jogos[f"oreb_{s}"] + jogos[f"tov_{s}"] + 0.44 * jogos[f"fta_{s}"]
                 for s in ("m", "v")) / 2
    lados = []
    for eu, ele in (("m", "v"), ("v", "m")):
        lados.append(pd.DataFrame({
            "id_jogo": jogos["id_jogo"],
            "lado": eu,
            "time": jogos["mandante" if eu == "m" else "visitante"],
            "data": jogos["data"],
            "saldo": jogos[f"pts_{eu}"] - jogos[f"pts_{ele}"],
            "ataque": 100 * jogos[f"pts_{eu}"] / posses,  # pontos feitos por 100 posses
            "defesa": 100 * jogos[f"pts_{ele}"] / posses,  # pontos sofridos por 100 posses
        }))
    return pd.concat(lados).sort_values(["time", "data"])


def calcular_janelas(jogos):
    """Forma, eficiência e descanso de cada time ANTES de cada jogo."""
    t = _por_time(jogos)
    g = t.groupby("time")
    for col in ("saldo", "ataque", "defesa"):
        # shift(1): tira o próprio jogo da conta. Sem isso o modelo veria o resultado que deveria prever.
        t[f"{col}10"] = g[col].transform(lambda s: s.shift(1).rolling(JANELA, min_periods=1).mean())
    t["descanso"] = g["data"].diff().dt.days.clip(upper=DESCANSO_MAX).fillna(DESCANSO_MAX)
    t["b2b"] = (t["descanso"] == 1).astype(int)

    cols = ["saldo10", "ataque10", "defesa10", "descanso", "b2b"]
    m = t[t["lado"] == "m"].set_index("id_jogo")[cols].add_suffix("_m")
    v = t[t["lado"] == "v"].set_index("id_jogo")[cols].add_suffix("_v")
    return jogos[["id_jogo"]].join(m, on="id_jogo").join(v, on="id_jogo").drop(columns="id_jogo")


def montar_tabela(jogos):
    """Tabela final do modelo: jogos + Elo + janelas + diferenças mandante - visitante."""
    elo, _ = calcular_elo(jogos)
    tab = jogos.join(elo).join(calcular_janelas(jogos))
    tab["dif_elo"] = tab["elo_m"] - tab["elo_v"]  # sem bônus de mando: o modelo aprende quanto vale jogar em casa
    for col in ("saldo10", "ataque10", "defesa10"):
        tab[f"dif_{col}"] = tab[f"{col}_m"] - tab[f"{col}_v"]
    return tab.dropna(subset=COLUNAS_MODELO)  # só cai o 1º jogo de cada time em 2014-15
