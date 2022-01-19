########## 1.5.1. Classificação ##########


    # A classe SGDClassifier implementa uma rotina de aprendizado de descida de gradiente estocástica simples que suporta diferentes funções de perda e penalidades para classificação. Abaixo está o limite de decisão de um SGDClassifier treinado com a perda de dobradiça, equivalente a um SVM linear. 

        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_separating_hyperplane.html

    # Como outros classificadores, o SGD deve ser ajustado com dois arrays: um array X de forma (n_samples, n_features) contendo as amostras de treinamento e um array y de forma (n_samples), contendo os valores alvo (rótulos de classe) para as amostras de treinamento : 


from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X, y)
     
     # Depois de ajustado, o modelo pode ser usado para prever novos valores: 

clf.predict([[2., 2.]])

    # O SGD ajusta um modelo linear aos dados de treinamento. O atributo coef_ contém os parâmetros do modelo: 

clf.coef_

    # O atributo intercept_ contém a interceptação (também conhecida como offset ou bias): 

clf.intercept_

    # Se o modelo deve ou não usar uma interceptação, ou seja, um hiperplano enviesado, é controlado pelo parâmetro fit_intercept.

    # A distância com sinal para o hiperplano (calculada como o produto escalar entre os coeficientes e a amostra de entrada, mais a interceptação) é dada por SGDClassifier.decision_function: 

clf.decision_function([[2., 2.]])

    # A função de perda de concreto pode ser definida através do parâmetro de perda. SGDClassifier suporta as seguintes funções de perda: 

        # loss="hinge": (margem suave) máquina de vetor de suporte linear,

        # loss="modified_huber": perda de dobradiça suavizada,

        # perda="log": regressão logística,

        # e todas as perdas de regressão abaixo. Nesse caso, o destino é codificado como -1 ou 1 e o problema é tratado como um problema de regressão. A classe prevista corresponde então ao sinal do alvo previsto.

    # Por favor, consulte a seção matemática abaixo para fórmulas. As duas primeiras funções de perda são preguiçosas, elas apenas atualizam os parâmetros do modelo se um exemplo violar a restrição de margem, o que torna o treinamento muito eficiente e pode resultar em modelos mais esparsos (ou seja, com mais coeficientes zero), mesmo quando a penalidade L2 é usada. 

clf = SGDClassifier(loss="log", max_iter=5).fit(X, y)
clf.predict_proba([[1., 1.]]) 

    # A penalidade concreta pode ser definida através do parâmetro de penalidade. O SGD suporta as seguintes penalidades: 

        # penalidade="l2": penalidade da norma L2 em coef_.

        # penalidade="l1": penalidade da norma L1 em coef_.

        # penalidade="elasticnet": Combinação convexa de L2 e L1; (1 - l1_ratio) * L2 + l1_ratio * L1.

    # A configuração padrão é penalidade="l2". A penalidade L1 leva a soluções esparsas, levando a maioria dos coeficientes a zero. O Elastic Net 11 resolve algumas deficiências da penalidade L1 na presença de atributos altamente correlacionados. O parâmetro l1_ratio controla a combinação convexa da penalidade L1 e L2. 

    # O SGDClassifier suporta classificação multiclasse combinando vários classificadores binários em um esquema “um contra todos” (OVA). Para cada uma das classes K, é aprendido um classificador binário que discrimina entre aquela e todas as outras classes K-1. No momento do teste, calculamos a pontuação de confiança (ou seja, as distâncias sinalizadas para o hiperplano) para cada classificador e escolhemos a classe com a maior confiança. A Figura abaixo ilustra a abordagem OVA no conjunto de dados da íris. As linhas tracejadas representam os três classificadores OVA; as cores de fundo mostram a superfície de decisão induzida pelos três classificadores. 

        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_iris.html


    # No caso de classificação multiclasse, coef_ é um array bidimensional de forma (n_classes, n_features) e intercept_ é um array unidimensional de shape (n_classes,). A i-ésima linha de coef_ contém o vetor de peso do classificador OVA para a i-ésima classe; as classes são indexadas em ordem crescente (consulte o atributo classes_). Observe que, em princípio, por permitirem criar um modelo de probabilidade, loss="log" e loss="modified_huber" são mais adequados para a classificação um contra todos.

    # O SGDClassifier suporta classes ponderadas e instâncias ponderadas por meio dos parâmetros de ajuste class_weight e sample_weight. Veja os exemplos abaixo e a docstring de SGDClassifier.fit para mais informações.

    # SGDClassifier suporta SGD de média (ASGD) 10. A média pode ser habilitada definindo average=True. O ASGD executa as mesmas atualizações que o SGD regular (consulte Formulação matemática), mas em vez de usar o último valor dos coeficientes como o atributo coef_ (ou seja, os valores da última atualização), coef_ é definido como o valor médio dos coeficientes em todas as atualizações. O mesmo é feito para o atributo intercept_. Ao usar o ASGD, a taxa de aprendizado pode ser maior e até constante, levando em alguns conjuntos de dados a uma aceleração no tempo de treinamento.

    # Para classificação com perda logística, outra variante do SGD com estratégia de média está disponível com algoritmo Stochastic Average Gradient (SAG), disponível como solver em LogisticRegression. 


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_separating_hyperplane.html#sphx-glr-auto-examples-linear-model-plot-sgd-separating-hyperplane-py

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_iris.html#sphx-glr-auto-examples-linear-model-plot-sgd-iris-py

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_weighted_samples.html#sphx-glr-auto-examples-linear-model-plot-sgd-weighted-samples-py

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html#sphx-glr-auto-examples-linear-model-plot-sgd-comparison-py

    ## https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-unbalanced-py