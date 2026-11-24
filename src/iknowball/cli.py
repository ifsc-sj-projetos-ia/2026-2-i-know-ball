"""Linha de comando: uv run iknowball <comando>"""
import argparse
from datetime import date

from iknowball import dados, features, modelo, previsao


def cmd_baixar(_args):
    dados.baixar()


def cmd_resumo(_args):
    jogos = dados.carregar_jogos()
    por_temporada = jogos.groupby("temporada").agg(
        jogos=("saldo", "size"),
        mandante_vence=("mandante_venceu", "mean"),
        saldo_medio=("saldo", "mean"),
    )
    print(por_temporada.round(3).to_string())
    print(f"\n{len(jogos)} jogos de temporada regular, de {jogos['data'].min():%d/%m/%Y} a {jogos['data'].max():%d/%m/%Y}")


def cmd_elo(_args):
    jogos = dados.carregar_jogos()
    elo, notas = features.calcular_elo(jogos)
    jogos = jogos.join(elo)
    jogos = jogos[jogos["temporada"] != jogos["temporada"].min()]  # 1ª temporada é aquecimento: todos começam em 1500
    jogos["acerto_elo"] = (jogos["prob_elo_m"] > 0.5) == jogos["mandante_venceu"].astype(bool)
    jogos["brier_elo"] = (jogos["prob_elo_m"] - jogos["mandante_venceu"]) ** 2

    por_temporada = jogos.groupby("temporada").agg(
        mandante_sempre=("mandante_venceu", "mean"),
        acerto_elo=("acerto_elo", "mean"),
        brier_elo=("brier_elo", "mean"),
    )
    print(por_temporada.round(3).to_string())
    print(f"\nGeral: mandante sempre {jogos['mandante_venceu'].mean():.1%} | "
          f"Elo {jogos['acerto_elo'].mean():.1%} | Brier {jogos['brier_elo'].mean():.3f} (0,25 = chute 50%)")

    print("\nElo atual:")
    for time, nota in sorted(notas.items(), key=lambda t: -t[1]):
        print(f"  {time}  {nota:6.0f}")


def cmd_treinar(_args):
    partes = modelo.dividir(features.montar_tabela(dados.carregar_jogos()))
    treino, val = partes["treino"], partes["validacao"]
    print(f"treino {len(treino)} jogos | validação {len(val)} jogos (o teste fica guardado)\n")

    reg = modelo.treinar_logistica(treino)
    candidatos = {
        "mandante sempre": [treino["mandante_venceu"].mean()] * len(val),
        "Elo puro": val["prob_elo_m"],
        "logística": reg.predict_proba(val[features.COLUNAS_MODELO])[:, 1],
    }
    print(f"{'na validação':<16} {'acerto':>7} {'brier':>7} {'log_loss':>9}")
    for nome, prob in candidatos.items():
        m = modelo.avaliar(val["mandante_venceu"], prob)
        print(f"{nome:<16} {m['acerto']:>7.1%} {m['brier']:>7.3f} {m['log_loss']:>9.3f}")

    print("\nPesos da logística (features padronizadas: quanto maior, mais pesa):")
    for nome, peso in modelo.coeficientes(reg).items():
        print(f"  {nome:<14} {peso:+.3f}")


def cmd_ablation(_args):
    partes = modelo.dividir(features.montar_tabela(dados.carregar_jogos()))
    res = modelo.ablation(partes["treino"], partes["validacao"])
    base = res["todas as 8"]
    print(f"{'na validação':<36} {'acerto':>7} {'log_loss':>9} {'Δ log_loss':>11}")
    for nome, m in res.items():
        if nome == "só Elo":
            print()
        print(f"{nome:<36} {m['acerto']:>7.1%} {m['log_loss']:>9.4f} {m['log_loss'] - base['log_loss']:>+11.4f}")
    print("\nΔ positivo = piorou sem o grupo (ele ajudava). Perto de zero = não faz falta.")


def cmd_prever(args):
    dia = date.fromisoformat(args.data) if args.data else previsao.amanha()
    dados.atualizar()
    prev = previsao.prever_dia(dia)
    if prev.empty:
        print(f"Nenhum jogo ainda não começado em {dia:%d/%m/%Y}.")
        return

    print(f"\nJogos de {dia:%d/%m/%Y}:")
    for p in prev.itertuples():
        cansado = [t for t, b in ((p.mandante, p.b2b_m), (p.visitante, p.b2b_v)) if b]
        nota = f"  (jogou ontem: {', '.join(cansado)})" if cansado else ""
        nota += "  (quadra neutra)" if p.neutro else ""
        print(f"  {p.visitante} @ {p.mandante}:  {p.favorito} {p.prob_favorito:.0%}{nota}")

    print(f"\nVitórias esperadas dos favoritos: {prev['prob_favorito'].sum():.1f} de {len(prev)}")
    print(f"Chance de TODOS os favoritos vencerem: {previsao.todos_favoritos(prev):.1%}")
    previsao.salvar(prev)
    print(f"Previsão guardada em {previsao.CSV_PREVISOES.relative_to(dados.RAIZ)}")


def main():
    parser = argparse.ArgumentParser(prog="iknowball", description="Previsão de jogos da NBA")
    sub = parser.add_subparsers(required=True, metavar="comando")
    sub.add_parser("baixar", help="baixa os jogos da nba_api para data/jogos.csv").set_defaults(func=cmd_baixar)
    sub.add_parser("resumo", help="mostra os jogos por temporada e o mando de quadra").set_defaults(func=cmd_resumo)
    sub.add_parser("elo", help="calcula o Elo, mede o acerto por temporada e mostra o ranking atual").set_defaults(func=cmd_elo)
    sub.add_parser("treinar", help="treina a logística e compara com os baselines na validação").set_defaults(func=cmd_treinar)
    sub.add_parser("ablation", help="tira um grupo de features por vez e mede o impacto").set_defaults(func=cmd_ablation)
    p = sub.add_parser("prever", help="prevê os jogos de um dia (padrão: amanhã) e guarda a previsão")
    p.add_argument("--data", help="AAAA-MM-DD")
    p.set_defaults(func=cmd_prever)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
