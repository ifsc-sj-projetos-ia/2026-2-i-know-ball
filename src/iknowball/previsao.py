"""Previsão dos jogos de um dia: calendário da nba_api + estado atual de cada time + modelo final."""
from datetime import date, datetime, timedelta
from math import prod

import pandas as pd
from nba_api.stats.endpoints import scoreboardv3

from iknowball import dados, features, modelo

CSV_PREVISOES = dados.RAIZ / "data" / "previsoes.csv"
NAO_COMECOU = 1  # gameStatus: 1 = agendado, 2 = em andamento, 3 = encerrado


def calendario(dia):
    """Jogos marcados para `dia`, com mandante, visitante e status."""
    jogos = scoreboardv3.ScoreboardV3(game_date=dia.isoformat(), league_id="00", timeout=30).get_dict()
    return [{
        "id_jogo": j["gameId"],
        "mandante": j["homeTeam"]["teamTricode"],
        "visitante": j["awayTeam"]["teamTricode"],
        "status": j["gameStatus"],
        "neutro": bool(j.get("isNeutral")),
    } for j in jogos["scoreboard"]["games"]]


def treinar_final(tab):
    # Para prever de verdade, treina com todas as temporadas completas (2015-16 a 2025-26).
    # A nota honesta do modelo continua vindo do teste, medido com um modelo que não viu 2025-26.
    completas = modelo.TREINO + modelo.VALIDACAO + modelo.TESTE
    return modelo.treinar_logistica(tab[tab["temporada"].isin(completas)])


def estado_dos_times(jogos):
    """Elo e forma de cada time depois do último jogo disputado (o 'agora')."""
    _, elo = features.calcular_elo(jogos)
    por_time = features.por_time(jogos)
    forma = por_time.groupby("time")["saldo"].apply(lambda s: s.tail(features.JANELA).mean())
    return elo, forma.to_dict()


def prever_dia(dia):
    jogos = dados.carregar_jogos()
    reg = treinar_final(features.montar_tabela(jogos))
    elo, forma = estado_dos_times(jogos)
    jogaram_ontem = {t for j in calendario(dia - timedelta(days=1)) for t in (j["mandante"], j["visitante"])}

    linhas = []
    for j in calendario(dia):
        if j["status"] != NAO_COMECOU:
            continue  # já começou: prever agora seria trapaça
        m, v = j["mandante"], j["visitante"]
        linhas.append({
            **j,
            "dif_elo": elo[m] - elo[v],
            "b2b_m": int(m in jogaram_ontem),
            "b2b_v": int(v in jogaram_ontem),
            "dif_saldo10": forma[m] - forma[v],
        })
    if not linhas:
        return pd.DataFrame()

    prev = pd.DataFrame(linhas)
    prev["prob_m"] = reg.predict_proba(prev[features.COLUNAS_MODELO])[:, 1]
    prev["favorito"] = prev["mandante"].where(prev["prob_m"] >= 0.5, prev["visitante"])
    prev["prob_favorito"] = prev["prob_m"].where(prev["prob_m"] >= 0.5, 1 - prev["prob_m"])
    prev["data_jogo"] = dia.isoformat()
    prev["previsto_em"] = datetime.now().isoformat(timespec="seconds")
    return prev


def todos_favoritos(prev):
    """Chance de TODOS os favoritos vencerem: produto das chances (supondo jogos independentes)."""
    return prod(prev["prob_favorito"])


def salvar(prev):
    """Guarda a previsão para conferir depois. Se o jogo já foi previsto, mantém a primeira (a feita antes)."""
    cols = ["previsto_em", "data_jogo", "id_jogo", "mandante", "visitante", "neutro",
            *features.COLUNAS_MODELO, "prob_m", "favorito", "prob_favorito"]
    novas = prev[cols]
    if CSV_PREVISOES.exists():
        antigas = pd.read_csv(CSV_PREVISOES, dtype={"id_jogo": str})
        novas = pd.concat([antigas, novas[~novas["id_jogo"].isin(antigas["id_jogo"])]])
    novas.to_csv(CSV_PREVISOES, index=False)


def amanha():
    return date.today() + timedelta(days=1)
