########## 4.1.2. Gráfico de expectativa condicional individual (ICE) ##########


    # Semelhante a um PDP, um gráfico de expectativa condicional individual (ICE) mostra a dependência entre a função alvo e um recurso de entrada de interesse. No entanto, ao contrário de um PDP, que mostra o efeito médio do recurso de entrada, um gráfico ICE visualiza a dependência da previsão em um recurso para cada amostra separadamente com uma linha por amostra. Devido aos limites da percepção humana, apenas um recurso de entrada de interesse é suportado para gráficos ICE.

    # As figuras abaixo mostram quatro gráficos ICE para o conjunto de dados de habitação da Califórnia, com um HistGradientBoostingRegressor. A segunda figura plota a linha PD correspondente sobreposta nas linhas ICE.

        # https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html

    # Embora os PDPs sejam bons em mostrar o efeito médio dos recursos de destino, eles podem obscurecer uma relação heterogênea criada por interações. Quando as interações estiverem presentes, o gráfico ICE fornecerá muito mais informações. Por exemplo, podemos observar uma relação linear entre a renda mediana e o preço da casa na linha PD. No entanto, as linhas ICE mostram que existem algumas exceções, onde o preço da casa permanece constante em algumas faixas da renda mediana.

    # A função de conveniência PartialDependenceDisplay.from_estimator do módulo sklearn.inspection pode ser usada para criar gráficos ICE definindo kind='individual'. No exemplo abaixo, mostramos como criar uma grade de gráficos ICE: 

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay

X, y = make_hastie_10_2(random_state=0)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X, y)
features = [0, 1]
PartialDependenceDisplay.from_estimator(clf, X, features,
    kind='individual')

    # Em gráficos ICE, pode não ser fácil ver o efeito médio do recurso de entrada de interesse. Portanto, é recomendável usar gráficos de ICE ao lado de PDPs. Eles podem ser plotados juntos com kind='both'. 

PartialDependenceDisplay.from_estimator(clf, X, features,     kind='both')