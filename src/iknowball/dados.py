"""Coleta (nba_api) e preparação dos jogos: de 2 linhas por jogo para 1 linha mandante x visitante."""
import time
from datetime import date
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

RAIZ = Path(__file__).resolve().parents[2]
CSV_JOGOS = RAIZ / "data" / "jogos.csv"

PRIMEIRA_TEMPORADA = 2014  # 2014-15: começo da era do arremesso de 3
TIPOS = ["Regular Season", "Playoffs"]

# Estatísticas de box score que vão para a tabela, uma vez para cada lado (_m e _v).
ESTATISTICAS = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf"]


def temporada_atual():
    # A temporada começa em outubro: em novembro de 2026 a atual é a 2026-27.
    hoje = date.today()
    return hoje.year if hoje.month >= 10 else hoje.year - 1


def _baixar_temporada(ano, tipo):
    temporada = f"{ano}-{str(ano + 1)[-2:]}"  # 2014 -> "2014-15"
    df = leaguegamefinder.LeagueGameFinder(
        season_nullable=temporada,
        season_type_nullable=tipo,
        league_id_nullable="00",  # 00 = NBA (exclui WNBA e G League)
        player_or_team_abbreviation="T",  # uma linha por time por jogo
        timeout=60,
    ).get_data_frames()[0]
    df["TEMPORADA"] = temporada
    df["TIPO"] = tipo
    print(f"{temporada} {tipo:<15} {len(df) // 2:>5} jogos")
    return df


def baixar(destino=CSV_JOGOS):
    """Baixa todos os jogos desde 2014-15 e salva o CSV bruto (2 linhas por jogo)."""
    partes = []
    for ano in range(PRIMEIRA_TEMPORADA, temporada_atual() + 1):
        for tipo in TIPOS:
            partes.append(_baixar_temporada(ano, tipo))
            time.sleep(1)  # o stats.nba.com bloqueia quem faz muitas chamadas seguidas

    bruto = pd.concat(partes, ignore_index=True).sort_values(["GAME_DATE", "GAME_ID"])
    bruto = bruto.dropna(subset=["WL"])  # jogo em andamento ainda não tem vencedor
    destino.parent.mkdir(exist_ok=True)
    bruto.to_csv(destino, index=False)
    print(f"\n{len(bruto) // 2} jogos salvos em {destino}")
    return bruto


def atualizar(destino=CSV_JOGOS):
    """Rebaixa só a temporada atual e troca essas linhas no CSV (as temporadas fechadas não mudam)."""
    atual = temporada_atual()
    nova = pd.concat([_baixar_temporada(atual, tipo) for tipo in TIPOS], ignore_index=True)
    nova = nova.dropna(subset=["WL"])
    bruto = pd.read_csv(destino, dtype={"GAME_ID": str})
    bruto = pd.concat([bruto[bruto["TEMPORADA"] != nova["TEMPORADA"].iloc[0]], nova], ignore_index=True)
    bruto.sort_values(["GAME_DATE", "GAME_ID"]).to_csv(destino, index=False)
    print(f"{len(nova) // 2} jogos da temporada atual atualizados em {destino}")


def montar_jogos(bruto):
    """Junta as 2 linhas de cada jogo numa só: colunas _m (mandante) e _v (visitante)."""
    bruto = bruto[bruto["TIPO"] == "Regular Season"].copy()
    bruto.columns = bruto.columns.str.lower()
    em_casa = bruto["matchup"].str.contains(" vs. ")  # "LAL vs. BOS" = LAL em casa; "LAL @ BOS" = fora

    # Jogo em quadra neutra (México, Paris, semifinal da NBA Cup) vem com "@" nos dois lados: sem mandante, fica fora.
    neutro = em_casa.groupby(bruto["game_id"]).transform("sum") != 1
    bruto, em_casa = bruto[~neutro], em_casa[~neutro]

    lado = ["game_id", "team_abbreviation", *ESTATISTICAS]
    jogos = bruto.loc[em_casa, ["game_id", "game_date", "temporada", *lado[1:]]].merge(
        bruto.loc[~em_casa, lado], on="game_id", suffixes=("_m", "_v"), validate="one_to_one")

    jogos = jogos.rename(columns={"game_id": "id_jogo", "game_date": "data",
                                  "team_abbreviation_m": "mandante", "team_abbreviation_v": "visitante"})
    jogos["data"] = pd.to_datetime(jogos["data"])
    jogos["saldo"] = jogos["pts_m"] - jogos["pts_v"]
    jogos["mandante_venceu"] = (jogos["saldo"] > 0).astype(int)
    return jogos.sort_values(["data", "id_jogo"]).reset_index(drop=True)


def carregar_jogos(caminho=CSV_JOGOS):
    """Lê o CSV salvo e devolve a tabela de jogos da temporada regular, 1 linha por jogo."""
    return montar_jogos(pd.read_csv(caminho, dtype={"GAME_ID": str}))
