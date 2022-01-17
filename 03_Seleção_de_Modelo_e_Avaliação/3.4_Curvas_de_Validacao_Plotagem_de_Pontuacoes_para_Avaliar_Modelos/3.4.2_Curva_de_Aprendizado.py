########## 3.4.2. Curva de aprendizado ##########

    # Uma curva de aprendizado mostra a validação e a pontuação de treinamento de um estimador para vários números de amostras de treinamento. É uma ferramenta para descobrir o quanto nos beneficiamos da adição de mais dados de treinamento e se o estimador sofre mais com um erro de variância ou um erro de viés. Considere o exemplo a seguir onde traçamos a curva de aprendizado de um classificador Bayes ingênuo e um SVM.

    # Para os Bayes ingênuos, tanto a pontuação de validação quanto a pontuação de treinamento convergem para um valor bastante baixo com o aumento do tamanho do conjunto de treinamento. Assim, provavelmente não nos beneficiaremos muito com mais dados de treinamento.

    # Em contraste, para pequenas quantidades de dados, a pontuação de treinamento do SVM é muito maior que a pontuação de validação. Adicionar mais amostras de treinamento provavelmente aumentará a generalização. 

        
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


    # Podemos usar a função learning_curve para gerar os valores necessários para traçar essa curva de aprendizado (número de amostras que foram usadas, as pontuações médias nos conjuntos de treinamento e as pontuações médias nos conjuntos de validação): 

from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

train_sizes, train_scores, valid_scores = learning_curve(
    SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
train_sizes

train_scores

valid_scores