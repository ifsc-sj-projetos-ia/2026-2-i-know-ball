"""Divisão temporal, treino e avaliação dos modelos."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from iknowball.features import COLUNAS_CANDIDATAS, COLUNAS_MODELO

SEMENTE = 42

# Por tempo, nunca sorteado: na vida real o passado prevê o futuro.
TREINO = [f"{a}-{str(a + 1)[-2:]}" for a in range(2015, 2024)]  # 2015-16 a 2023-24 (2014-15 é aquecimento do Elo)
VALIDACAO = ["2024-25"]  # escolher parâmetros
TESTE = ["2025-26"]  # usar UMA vez, no final
AO_VIVO = ["2026-27"]  # a demo


def dividir(tab):
    return {nome: tab[tab["temporada"].isin(temps)]
            for nome, temps in (("treino", TREINO), ("validacao", VALIDACAO), ("teste", TESTE), ("ao_vivo", AO_VIVO))}


def treinar_logistica(treino, colunas=COLUNAS_MODELO):
    # Padronizar (média 0, desvio 1) deixa os coeficientes comparáveis entre si. O scaler aprende só com o treino.
    modelo = make_pipeline(StandardScaler(), LogisticRegression(random_state=SEMENTE))
    modelo.fit(treino[colunas], treino["mandante_venceu"])
    return modelo


GRUPOS = {
    "forma (saldo10)": ["dif_saldo10"],
    "eficiência (ataque/defesa10)": ["dif_ataque10", "dif_defesa10"],
    "descanso (dias)": ["descanso_m", "descanso_v"],
    "back-to-back": ["b2b_m", "b2b_v"],
}
EXPERIMENTOS = {
    "só Elo": ["dif_elo"],
    "Elo + back-to-back": ["dif_elo", "b2b_m", "b2b_v"],
    "FINAL: Elo + back-to-back + forma": COLUNAS_MODELO,
    "+ eficiência": COLUNAS_MODELO + ["dif_ataque10", "dif_defesa10"],
    "+ descanso": COLUNAS_MODELO + ["descanso_m", "descanso_v"],
}


def ablation(treino, val):
    """Tira um grupo de features por vez, retreina e mede na validação. Piorou muito = o grupo importava."""
    def medir(colunas):
        reg = treinar_logistica(treino, colunas)
        return avaliar(val["mandante_venceu"], reg.predict_proba(val[colunas])[:, 1])

    resultados = {"todas as 8": medir(COLUNAS_CANDIDATAS)}
    for nome, cols in GRUPOS.items():
        resultados[f"8 sem {nome}"] = medir([c for c in COLUNAS_CANDIDATAS if c not in cols])
    resultados.update({nome: medir(cols) for nome, cols in EXPERIMENTOS.items()})
    return resultados


def avaliar(y, prob):
    prob = np.asarray(prob)
    return {
        "acerto": accuracy_score(y, prob > 0.5),
        "brier": brier_score_loss(y, prob),  # 0,25 = chutar 50% em tudo; menor é melhor
        "log_loss": log_loss(y, prob),  # pune caro a confiança errada
    }


def coeficientes(modelo):
    reg = modelo[-1]
    return dict(intercepto=reg.intercept_[0], **dict(zip(modelo.feature_names_in_, reg.coef_[0])))
