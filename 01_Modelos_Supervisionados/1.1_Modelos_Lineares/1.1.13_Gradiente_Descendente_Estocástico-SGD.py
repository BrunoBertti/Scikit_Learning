########## 1.1.13 Gradiente Descendente Estocástico - SGD ##########

    # A descida gradiente estocástica é uma abordagem simples, mas muito eficiente, para ajustar modelos lineares. É particularmente útil quando o número de amostras (e o número de recursos) é muito grande. O método partial_fit permite o aprendizado online / out-of-core.

    # As classes SGDClassifier e SGDRegressor fornecem funcionalidade para ajustar modelos lineares para classificação e regressão usando diferentes funções de perda (convexa) e diferentes penalidades. Por exemplo, com loss = "log", SGDClassifier se ajusta a um modelo de regressão logística, enquanto com loss = "hinge" ele se ajusta a uma máquina de vetor de suporte linear (SVM).

    ## Referências:

    ## https://scikit-learn.org/stable/modules/sgd.html#sgd