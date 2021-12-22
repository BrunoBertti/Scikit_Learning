########## 1.13.2. Seleção univariada de variáveis ##########

    # A seleção univariada de recursos funciona selecionando os melhores recursos com base em testes estatísticos univariados. Pode ser visto como uma etapa de pré-processamento para um estimador. Scikit-learn expõe rotinas de seleção de recursos como objetos que implementam o método de transformação: 


        # SelectKBest remove todos, exceto os k recursos de maior pontuação

        # SelectPercentile remove todos os recursos, exceto a maior porcentagem de pontuação especificada pelo usuário

        # usando testes estatísticos univariados comuns para cada recurso: taxa de falsos positivos SelectFpr, taxa de descoberta falsa SelectFdr ou erro de família SelectFwe.

        # GenericUnivariateSelect permite realizar a seleção univariada de recursos com uma estratégia configurável. Isso permite selecionar a melhor estratégia de seleção univariada com o estimador de busca hiperparâmetro. 

    # Por exemplo, podemos realizar um teste \ chi ^ 2 nas amostras para recuperar apenas os dois melhores recursos da seguinte maneira: 


from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X, y = load_iris(return_X_y=True)
X.shape
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new.shape

    # Esses objetos tomam como entrada uma função de pontuação que retorna pontuações e valores p univariados (ou apenas pontuações para SelectKBest e SelectPercentile):

        # Para regressão: f_regression, mutual_info_regression

        # Para classificação: chi2, f_classif, mutual_info_classif 

    # Os métodos baseados no teste F estimam o grau de dependência linear entre duas variáveis aleatórias. Por outro lado, os métodos de informação mútua podem capturar qualquer tipo de dependência estatística, mas sendo não paramétricos, eles requerem mais amostras para uma estimativa precisa. 



    # Seleção de recursos com dados esparsos

        # Se você usar dados esparsos (ou seja, dados representados como matrizes esparsas), chi2, mutual_info_regression, mutual_info_classif lidará com os dados sem torná-los densos. 

    # Aviso: Cuidado para não usar uma função de pontuação de regressão com um problema de classificação, você obterá resultados inúteis. 



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-py

    ## https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py