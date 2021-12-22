########## 1.13.3. Eliminação de variável recursivo ##########


    # Dado um estimador externo que atribui pesos aos variáveis (por exemplo, os coeficientes de um modelo linear), o objetivo da eliminação de variáveis recursivos (RFE) é selecionar variáveis considerando recursivamente conjuntos cada vez menores de variáveis. Primeiro, o estimador é treinado no conjunto inicial de variáveis e a importância de cada recurso é obtida por meio de qualquer atributo específico (como coef_, feature_importances_) ou chamável. Em seguida, os variáveis menos importantes são removidos do conjunto atual de variáveis. Esse procedimento é repetido recursivamente no conjunto podado até que o número desejado de variáveis a serem selecionados seja atingido.

    # RFECV executa RFE em um loop de validação cruzada para encontrar o número ideal de variáveis.


    ## Exemplos:

    ## A recursive feature elimination example showing the relevance of pixels in a digit classification task. (https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py)

    ## A recursive feature elimination example with automatic tuning of the number of features selected with cross-validation. (https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py)