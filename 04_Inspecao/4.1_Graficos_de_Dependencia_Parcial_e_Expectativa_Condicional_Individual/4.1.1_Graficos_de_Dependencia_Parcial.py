########## 4.1.1. Gráficos de dependência parcial ##########


    # Gráficos de dependência parcial (PDP) mostram a dependência entre a resposta alvo e um conjunto de recursos de entrada de interesse, marginalizando os valores de todos os outros recursos de entrada (os recursos de "complemento"). Intuitivamente, podemos interpretar a dependência parcial como a resposta alvo esperada em função das características de entrada de interesse.

    # Devido aos limites da percepção humana, o tamanho do conjunto de recursos de entrada de interesse deve ser pequeno (geralmente, um ou dois), portanto, os recursos de entrada de interesse geralmente são escolhidos entre os recursos mais importantes.

    # A figura abaixo mostra dois gráficos de dependência parcial de uma via e um de duas vias para o conjunto de dados de habitação da Califórnia, com um HistGradientBoostingRegressor: 

        # https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html

    

    # Os PDPs unidirecionais nos informam sobre a interação entre a resposta do alvo e um recurso de entrada de interesse (por exemplo, linear, não linear). O gráfico da esquerda na figura acima mostra o efeito da ocupação média no preço médio da casa; podemos ver claramente uma relação linear entre eles quando a ocupação média é inferior a 3 pessoas. Da mesma forma, poderíamos analisar o efeito da idade da casa no preço médio da casa (parcela do meio). Assim, essas interpretações são marginais, considerando uma característica de cada vez.

    # PDPs com dois recursos de entrada de interesse mostram as interações entre os dois recursos. Por exemplo, o PDP de duas variáveis ​​na figura acima mostra a dependência do preço médio da casa em relação aos valores conjuntos da idade da casa e dos ocupantes médios por família. Podemos ver claramente uma interação entre as duas características: para uma ocupação média superior a dois, o preço da casa é quase independente da idade da casa, enquanto para valores inferiores a 2 há uma forte dependência da idade.

    # O módulo sklearn.inspection fornece uma função de conveniência from_estimator para criar gráficos de dependência parcial unidirecionais e bidirecionais. No exemplo abaixo, mostramos como criar uma grade de gráficos de dependência parcial: dois PDPs unidirecionais para os recursos 0 e 1 e um PDP bidirecional entre os dois recursos: 

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay

X, y = make_hastie_10_2(random_state=0)
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X, y)
features = [0, 1, (0, 1)]
PartialDependenceDisplay.from_estimator(clf, X, features)


    # Você pode acessar a figura recém-criada e os objetos Axes usando plt.gcf() e plt.gca().

    # Para classificação multiclasse, você precisa definir o rótulo de classe para o qual os PDPs devem ser criados por meio do argumento de destino: 

from sklearn.datasets import load_iris
iris = load_iris()
mc_clf = GradientBoostingClassifier(n_estimators=10,
    max_depth=1).fit(iris.data, iris.target)
features = [3, 2, (3, 2)]
PartialDependenceDisplay.from_estimator(mc_clf, X, features, target=0)


    # O mesmo destino de parâmetro é usado para especificar o destino nas configurações de regressão de várias saídas.

    # Se você precisar dos valores brutos da função de dependência parcial em vez dos gráficos, poderá usar a função sklearn.inspection.partial_dependence: 

from sklearn.inspection import partial_dependence

pdp, axes = partial_dependence(clf, X, [0])
pdp

axes


    # Os valores nos quais a dependência parcial deve ser avaliada são gerados diretamente de X. Para dependência parcial de 2 vias, uma grade 2D de valores é gerada. O campo de valores retornado por sklearn.inspection.partial_dependence fornece os valores reais usados na grade para cada recurso de entrada de interesse. Eles também correspondem ao eixo das parcelas. 