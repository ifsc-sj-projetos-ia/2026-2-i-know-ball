import pandas as pd

from iknowball.dados import ESTATISTICAS, montar_jogos


def _linha(jogo, time, matchup, wl, pts, tipo="Regular Season"):
    linha = {"GAME_ID": jogo, "GAME_DATE": "2026-11-01", "TEMPORADA": "2026-27", "TIPO": tipo,
             "TEAM_ABBREVIATION": time, "MATCHUP": matchup, "WL": wl}
    linha.update({e.upper(): 0 for e in ESTATISTICAS})
    linha["PTS"] = pts
    return linha


def test_junta_mandante_e_visitante():
    bruto = pd.DataFrame([
        _linha("1", "BOS", "BOS @ LAL", "W", 110),
        _linha("1", "LAL", "LAL vs. BOS", "L", 100),
    ])
    jogo = montar_jogos(bruto).iloc[0]
    assert (jogo.mandante, jogo.visitante) == ("LAL", "BOS")
    assert (jogo.pts_m, jogo.pts_v, jogo.saldo, jogo.mandante_venceu) == (100, 110, -10, 0)


def test_descarta_quadra_neutra():
    bruto = pd.DataFrame([
        _linha("1", "MIA", "MIA @ WAS", "W", 118),
        _linha("1", "WAS", "MIA @ WAS", "L", 98),
    ])
    assert montar_jogos(bruto).empty


def test_descarta_playoffs():
    bruto = pd.DataFrame([
        _linha("1", "LAL", "LAL vs. BOS", "W", 100),
        _linha("1", "BOS", "BOS @ LAL", "L", 90),
        _linha("2", "LAL", "LAL vs. BOS", "W", 100, tipo="Playoffs"),
        _linha("2", "BOS", "BOS @ LAL", "L", 90, tipo="Playoffs"),
    ])
    assert montar_jogos(bruto)["id_jogo"].tolist() == ["1"]


def test_elo_usa_so_o_passado():
    from iknowball.features import ELO_INICIAL, calcular_elo
    bruto = pd.DataFrame([
        _linha("1", "LAL", "LAL vs. BOS", "W", 120), _linha("1", "BOS", "BOS @ LAL", "L", 100),
        _linha("2", "LAL", "LAL vs. BOS", "W", 120), _linha("2", "BOS", "BOS @ LAL", "L", 100),
    ])
    elo, _ = calcular_elo(montar_jogos(bruto))
    assert elo.iloc[0].elo_m == elo.iloc[0].elo_v == ELO_INICIAL  # 1º jogo: ninguém sabe nada ainda
    assert elo.iloc[1].elo_m > ELO_INICIAL > elo.iloc[1].elo_v  # 2º jogo já reflete a vitória do 1º


def test_janela_nao_inclui_o_proprio_jogo():
    from iknowball.features import calcular_janelas
    bruto = pd.DataFrame([
        _linha("1", "LAL", "LAL vs. BOS", "W", 120), _linha("1", "BOS", "BOS @ LAL", "L", 100),
        _linha("2", "LAL", "LAL vs. BOS", "L", 90), _linha("2", "BOS", "BOS @ LAL", "W", 100),
    ])
    jan = calcular_janelas(montar_jogos(bruto))
    assert pd.isna(jan.iloc[0].saldo10_m)  # antes do 1º jogo não há passado
    assert jan.iloc[1].saldo10_m == 20  # no 2º jogo só entra o +20 do 1º, não o -10 do próprio jogo
