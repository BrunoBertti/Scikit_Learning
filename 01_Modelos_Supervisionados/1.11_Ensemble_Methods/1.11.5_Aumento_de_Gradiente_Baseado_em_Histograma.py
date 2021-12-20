########## 1.11.5 Aumento de gradiente baseado em histograma ##########


    # Scikit-learn 0.21 introduziu duas novas implementações de árvores de aumento de gradiente, ou seja, HistGradientBoostingClassifier e HistGradientBoostingRegressor, inspirado por LightGBM (consulte [LightGBM]).

    # Esses estimadores baseados em histograma podem ser ordens de magnitude mais rápidos do que GradientBoostingClassifier e GradientBoostingRegressor quando o número de amostras é maior do que dezenas de milhares de amostras.

    # Eles também têm suporte integrado para valores ausentes, o que evita a necessidade de um imputador.

    # Esses estimadores rápidos primeiro agrupam as amostras de entrada X em compartimentos de valor inteiro (normalmente 256 compartimentos), o que reduz enormemente o número de pontos de divisão a serem considerados e permite que o algoritmo aproveite estruturas de dados baseadas em inteiros (histogramas) em vez de depender de contínuos classificados valores ao construir as árvores. A API desses estimadores é um pouco diferente e alguns dos recursos de GradientBoostingClassifier e GradientBoostingRegressor ainda não são suportados, por exemplo, algumas funções de perda. 


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html#sphx-glr-auto-examples-inspection-plot-partial-dependence-py



##### 1.11.5.1. Uso

    # A maioria dos parâmetros permanece inalterada em GradientBoostingClassifier e GradientBoostingRegressor. Uma exceção é o parâmetro max_iter que substitui n_estimators e controla o número de iterações do processo de aumento: 

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2

X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]

clf = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
clf.score(X_test, y_test)


        # As perdas disponíveis para regressão são ‘squared_error’, ‘absolute_error’, que é menos sensível a outliers, e ‘poisson’, que é adequado para modelar contagens e frequências. Para classificação, ‘binary_crossentropy’ é usado para classificação binária e ‘categorical_crossentropy’ é usado para classificação multiclasse. Por padrão, a perda é 'automática' e selecionará a perda apropriada dependendo de y passado para ajustar.

        # O tamanho das árvores pode ser controlado por meio dos parâmetros max_leaf_nodes, max_depth e min_samples_leaf.

        # O número de compartimentos usados ​​para agrupar os dados é controlado com o parâmetro max_bins. O uso de menos escaninhos atua como uma forma de regularização. Geralmente, é recomendável usar o máximo de caixas possíveis, que é o padrão.

        # O parâmetro l2_regularization é um regularizador na função de perda e corresponde à equação (2) de [XGBoost].

        # Observe que a parada antecipada é habilitada por padrão se o número de amostras for maior que 10.000. O comportamento de parada antecipada é controlado por meio dos parâmetros de parada antecipada, pontuação, validation_fraction, n_iter_no_change e tol. É possível interromper precocemente usando um marcador arbitrário ou apenas a perda de treinamento ou validação. Observe que, por razões técnicas, usar um marcador é significativamente mais lento do que usar a perda. Por padrão, a parada antecipada é realizada se houver pelo menos 10.000 amostras no conjunto de treinamento, usando a perda de validação.  



##### 1.11.5.2. Suporte a valores ausentes

    # HistGradientBoostingClassifier e HistGradientBoostingRegressor têm suporte integrado para valores ausentes (NaNs).

    # Durante o treinamento, o produtor de árvores aprende em cada ponto de divisão se as amostras com valores ausentes devem ir para a criança esquerda ou direita, com base no ganho potencial. Ao fazer a previsão, as amostras com valores ausentes são atribuídas ao filho esquerdo ou direito, conseqüentemente: 

from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

X = np.array([0, 1, 2, np.nan]).reshape(-1, 1)
y = [0, 0, 1, 1]

gbdt = HistGradientBoostingClassifier(min_samples_leaf=1).fit(X, y)
gbdt.predict(X)


    # Quando o padrão de falta é preditivo, as divisões podem ser feitas se o valor do recurso está faltando ou não: 

X = np.array([0, np.nan, 1, 2, np.nan]).reshape(-1, 1)
y = [0, 1, 0, 0, 1]
gbdt = HistGradientBoostingClassifier(min_samples_leaf=1,
                                      max_depth=2,
                                      learning_rate=1,
                                      max_iter=1).fit(X, y)
gbdt.predict(X)


    # Se nenhum valor ausente for encontrado para um determinado recurso durante o treinamento, as amostras com valores ausentes serão mapeadas para o filho que tiver mais amostras. 




##### 1.11.5.3. Suporte de peso de amostra

   # HistGradientBoostingClassifier e HistGradientBoostingRegressor apoiam pesos de amostra durante o ajuste.

   # O exemplo de brinquedo a seguir demonstra como o modelo ignora as amostras com pesos de amostra zero: 

X = [[1, 0],
     [1, 0],
     [1, 0],
     [0, 1]]
y = [0, 0, 1, 0]
# ignore as 2 primeiras amostras de treinamento definindo seu peso para 0 
sample_weight = [0, 0, 1, 1]
gb = HistGradientBoostingClassifier(min_samples_leaf=1)
gb.fit(X, y, sample_weight=sample_weight)
HistGradientBoostingClassifier(...)
gb.predict([[1, 0]])

gb.predict_proba([[1, 0]])[0, 1]


    # Como você pode ver, [1, 0] é confortavelmente classificado como 1, uma vez que as duas primeiras amostras são ignoradas devido aos seus pesos amostrais.

    # Detalhe de implementação: levar os pesos da amostra em consideração para multiplicar os gradientes (e os hessianos) pelos pesos da amostra. Observe que o estágio de categorização (especificamente o cálculo de quantis) não leva os pesos em consideração. 




##### 1.11.5.4. Suporte para recursos categóricos


    # HistGradientBoostingClassifier e HistGradientBoostingRegressor têm suporte nativo para recursos categóricos: eles podem considerar divisões em dados categóricos não ordenados.

    # Para conjuntos de dados com recursos categóricos, usar o suporte categórico nativo geralmente é melhor do que contar com a codificação one-hot (OneHotEncoder), porque a codificação one-hot requer mais profundidade de árvore para obter divisões equivalentes. Também é geralmente melhor contar com o suporte categórico nativo em vez de tratar características categóricas como contínuas (ordinal), o que acontece para dados categóricos codificados por ordinal, uma vez que categorias são quantidades nominais onde a ordem não importa.

    # Para habilitar o suporte categórico, uma máscara booleana pode ser passada para o parâmetro categorical_features, indicando qual recurso é categórico. A seguir, o primeiro recurso será tratado como categórico e o segundo como numérico: 

gbdt = HistGradientBoostingClassifier(categorical_features=[True, False])

    # De forma equivalente, pode-se passar uma lista de inteiros indicando os índices das características categóricas: 

gbdt = HistGradientBoostingClassifier(categorical_features=[0])


    # A cardinalidade de cada característica categórica deve ser menor que o parâmetro max_bins, e cada característica categórica deve ser codificada em [0, max_bins - 1]. Para esse fim, pode ser útil pré-processar os dados com um OrdinalEncoder como feito em Suporte de recurso categórico no aumento de gradiente.

    # Se houver valores ausentes durante o treinamento, os valores ausentes serão tratados como uma categoria adequada. Se não houver valores ausentes durante o treinamento, então, no momento da previsão, os valores ausentes são mapeados para o nó filho que tem a maioria das amostras (assim como para recursos contínuos). Ao fazer a previsão, as categorias que não foram vistas durante o tempo de ajuste serão tratadas como valores ausentes. 


    # Descoberta de divisão com recursos categóricos: A maneira canônica de considerar divisões categóricas em uma árvore é considerar todas as 2 ^ {K - 1} - 1 partições, onde K é o número de categorias. Isso pode se tornar rapidamente proibitivo quando K é grande. Felizmente, como as árvores de aumento de gradiente são sempre árvores de regressão (mesmo para problemas de classificação), existe uma estratégia mais rápida que pode produzir divisões equivalentes. Primeiro, as categorias de um recurso são classificadas de acordo com a variância do destino, para cada categoria k. Uma vez que as categorias são classificadas, pode-se considerar partições contínuas, ou seja, tratar as categorias como se fossem valores contínuos ordenados (ver Fisher [Fisher1958] para uma prova formal). Como resultado, apenas K - 1 divisões precisam ser consideradas em vez de 2 ^ {K - 1} - 1. A classificação inicial é uma operação \ mathcal {O} (K \ log (K)), levando a uma complexidade total de \ mathcal {O} (K \ log (K) + K), em vez de \ mathcal {O} (2 ^ K)


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_categorical.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-categorical-py



##### 1.11.5.5. Restrições Monotônicas

    # Dependendo do problema em questão, você pode ter conhecimento prévio de que um determinado recurso deve, em geral, ter um efeito positivo (ou negativo) no valor alvo. Por exemplo, se todo o resto for igual, uma pontuação de crédito mais alta deve aumentar a probabilidade de aprovação para um empréstimo. Restrições monotônicas permitem que você incorpore esse conhecimento prévio ao modelo.

    # Uma restrição monotônica positiva é uma restrição da forma:

    # x_1 \ leq x_1 '\ implica F (x_1, x_2) \ leq F (x_1', x_2), onde F é o preditor com duas características

    # Da mesma forma, uma restrição monotônica negativa tem a forma:

        # x_1 \ leq x_1 '\ implica F (x_1, x_2) \ geq F (x_1', x_2)

    # Observe que as restrições monotônicas apenas restringem a saída “todo o resto sendo igual”. De fato, a seguinte relação não é imposta por uma restrição positiva: x_1 \ leq x_1 '\ implica F (x_1, x_2) \ leq F (x_1', x_2 ').

    # Você pode especificar uma restrição monotônica em cada recurso usando o parâmetro monotonic_cst. Para cada recurso, um valor de 0 indica nenhuma restrição, enquanto -1 e 1 indicam uma restrição negativa e positiva, respectivamente:

from sklearn.ensemble import HistGradientBoostingRegressor

# positivo, negativo e sem restrição nos 3 recursos 

gbdt = HistGradientBoostingRegressor(monotonic_cst=[1, -1, 0])


    # Em um contexto de classificação binária, impor uma restrição monotônica significa que o recurso deve ter um efeito positivo / negativo na probabilidade de pertencer à classe positiva. Restrições monotônicas não são suportadas para contexto multiclasse.


    # OBS: Como as categorias são quantidades não ordenadas, não é possível impor restrições monotônicas em características categóricas. 


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_monotonic_constraints.html#sphx-glr-auto-examples-ensemble-plot-monotonic-constraints-py

##### 1.11.5.6. Paralelismo de baixo nível


    # HistGradientBoostingClassifier e HistGradientBoostingRegressor têm implementações que usam OpenMP para paralelização por meio de Cython. Para obter mais detalhes sobre como controlar o número de threads, consulte nossas notas de paralelismo.

    # As seguintes partes são paralelizadas:


        # mapear amostras de valores reais para compartimentos de valor inteiro (encontrar os limites do compartimento é, no entanto, sequencial)

        # a construção de histogramas é paralelizada aos recursos

        # encontrar o melhor ponto de divisão em um nó é paralelizado aos recursos

        # durante o ajuste, o mapeamento de amostras nos filhos esquerdo e direito é colocado em paralelo com as amostras

        # cálculos de gradiente e hessianos são paralelizados sobre as amostras

        # a previsão é paralelizada sobre as amostras     

##### 1.11.5.7. Por que é mais rápido 

    # O gargalo de um procedimento de aumento de gradiente é a construção das árvores de decisão. Construir uma árvore de decisão tradicional (como nos outros GBDTs GradientBoostingClassifier e GradientBoostingRegressor) requer a classificação das amostras em cada nó (para cada recurso). A classificação é necessária para que o ganho potencial de um ponto de divisão possa ser calculado com eficiência. Dividir um único nó tem, portanto, uma complexidade de \ mathcal {O} (n_ \ text {recursos} \ vezes n \ log (n)) onde n é o número de amostras no nó.

    # HistGradientBoostingClassifier e HistGradientBoostingRegressor, em contraste, não exigem a classificação dos valores do recurso e, em vez disso, usam uma estrutura de dados chamada histograma, onde as amostras são implicitamente ordenadas. A construção de um histograma possui uma complexidade \ mathcal {O} (n), então o procedimento de divisão de nós tem uma complexidade \ mathcal {O} (n_ \ text {features} \ times n), muito menor que o anterior. Além disso, em vez de considerar n pontos de divisão, consideramos aqui apenas os pontos de divisão max_bins, que são muito menores.

    # Para construir histogramas, os dados de entrada X precisam ser agrupados em compartimentos de valor inteiro. Este procedimento de categorização requer a classificação dos valores do recurso, mas isso só acontece uma vez no início do processo de boosting (não em cada nó, como em GradientBoostingClassifier e GradientBoostingRegressor).

    # Finalmente, muitas partes da implementação de HistGradientBoostingClassifier e HistGradientBoostingRegressor são paralelizadas. 



    ## Referências:

    ## F1999 Friedmann, Jerome H., 2007, “Stochastic Gradient Boosting” (https://statweb.stanford.edu/~jhf/ftp/stobst.pdf)


    ## R2007 G. Ridgeway, “Generalized Boosted Models: A guide to the gbm package”, 2007


    ## XGBoost Tianqi Chen, Carlos Guestrin, “XGBoost: A Scalable Tree Boosting System” (https://arxiv.org/abs/1603.02754)


    ## LightGBM(1,2)Ke et. al. “LightGBM: A Highly Efficient Gradient BoostingDecision Tree” (https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)

    ## Fisher1958 Walter D. Fisher. “On Grouping for Maximum Homogeneity” (http://www.csiss.org/SPACE/workshops/2004/SAC/files/fisher.pdf)


