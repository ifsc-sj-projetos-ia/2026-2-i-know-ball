"""Divisão temporal, treino e avaliação dos modelos."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from iknowball.features import COLUNAS_MODELO

SEMENTE = 42

# Por tempo, nunca sorteado: na vida real o passado prevê o futuro.
TREINO = [f"{a}-{str(a + 1)[-2:]}" for a in range(2015, 2024)]  # 2015-16 a 2023-24 (2014-15 é aquecimento do Elo)
VALIDACAO = ["2024-25"]  # escolher parâmetros
TESTE = ["2025-26"]  # usar UMA vez, no final
AO_VIVO = ["2026-27"]  # a demo


def dividir(tab):
    return {nome: tab[tab["temporada"].isin(temps)]
            for nome, temps in (("treino", TREINO), ("validacao", VALIDACAO), ("teste", TESTE), ("ao_vivo", AO_VIVO))}


def treinar_logistica(treino):
    # Padronizar (média 0, desvio 1) deixa os coeficientes comparáveis entre si. O scaler aprende só com o treino.
    modelo = make_pipeline(StandardScaler(), LogisticRegression(random_state=SEMENTE))
    modelo.fit(treino[COLUNAS_MODELO], treino["mandante_venceu"])
    return modelo


def avaliar(y, prob):
    prob = np.asarray(prob)
    return {
        "acerto": accuracy_score(y, prob > 0.5),
        "brier": brier_score_loss(y, prob),  # 0,25 = chutar 50% em tudo; menor é melhor
        "log_loss": log_loss(y, prob),  # pune caro a confiança errada
    }


def coeficientes(modelo):
    reg = modelo[-1]
    return dict(intercepto=reg.intercept_[0], **dict(zip(COLUNAS_MODELO, reg.coef_[0])))
