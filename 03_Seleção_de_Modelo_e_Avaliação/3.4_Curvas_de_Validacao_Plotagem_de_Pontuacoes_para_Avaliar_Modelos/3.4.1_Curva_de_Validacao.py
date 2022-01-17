########## 3.4.1. Curva de validação ##########

    # Para validar um modelo, precisamos de uma função de pontuação (consulte Métricas e pontuação: quantificando a qualidade das previsões), por exemplo, precisão para classificadores. A maneira correta de escolher vários hiperparâmetros de um estimador é, obviamente, a pesquisa de grade ou métodos semelhantes (consulte Ajustando os hiperparâmetros de um estimador) que selecionam o hiperparâmetro com a pontuação máxima em um conjunto de validação ou vários conjuntos de validação. Observe que, se otimizarmos os hiperparâmetros com base em uma pontuação de validação, a pontuação de validação é tendenciosa e não é mais uma boa estimativa da generalização. Para obter uma estimativa adequada da generalização, temos que calcular a pontuação em outro conjunto de teste.

    # No entanto, às vezes é útil traçar a influência de um único hiperparâmetro na pontuação de treinamento e na pontuação de validação para descobrir se o estimador está superajustado ou subajustado para alguns valores de hiperparâmetros.

    # A função validation_curve pode ajudar neste caso: 

import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge

np.random.seed(0)
X, y = load_iris(return_X_y=True)
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

train_scores, valid_scores = validation_curve(
    Ridge(), X, y, param_name="alpha", param_range=np.logspace(-7, 3, 3),
    cv=5)
train_scores
valid_scores

    # Se a pontuação de treinamento e a pontuação de validação forem baixas, o estimador estará subajustado. Se a pontuação de treinamento for alta e a pontuação de validação for baixa, o estimador está superajustado e, caso contrário, está funcionando muito bem. Uma pontuação baixa de treinamento e uma pontuação alta de validação geralmente não são possíveis. Underfitting, overfitting e um modelo de trabalho são mostrados no gráfico abaixo, onde variamos o parâmetro \gamma de um SVM no conjunto de dados de dígitos. 

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html