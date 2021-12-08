########## 1.1.15 Algorítmo Passivo-Agressivo ##########

    # O algorítmo passivo agressivo são de uma família de algorítmos de larga escala de aprendizagem. Eles são similiar ao Perceptron em que eles não requerem uma tama de aprendizado. Entretanto, contrário ao Perceptron, eles incluem a regularização do parametro C.

    # Para classificação, PassiveAggressiveClassifier pode ser usado com loss='hinge' (PA-I) ou loss='squared_hinge' (PA-II). Para regressão, PassiveAggressiveRegressor pode ser usado com loss = 'epsilon_insensitive' (PA-I) ou loss = 'squared_epsilon_insensitive' (PA-II). 

    ## Referências:

    ## “Online Passive-Aggressive Algorithms” K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR 7 (2006) (http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf)