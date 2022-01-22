############ 1.8. Decomposição cruzada ############

    # O módulo de decomposição cruzada contém estimadores supervisionados para redução e regressão de dimensionalidade, pertencentes à família “Partial Least Squares”. 

        # https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_compare_cross_decomposition.html


    # Algoritmos de decomposição cruzada encontram as relações fundamentais entre duas matrizes (X e Y). São abordagens de variáveis ​​latentes para modelar as estruturas de covariância nesses dois espaços. Eles tentarão encontrar a direção multidimensional no espaço X que explique a direção da variância multidimensional máxima no espaço Y. Em outras palavras, o PLS projeta X e Y em um subespaço de dimensão inferior, de modo que a covariância entre transformado(X) e transformado(Y) seja máxima.

    # O PLS traça semelhanças com a regressão de componentes principais (PCR), onde as amostras são projetadas primeiro em um subespaço de dimensão inferior e os alvos y são previstos usando transformado(X). Um problema com a PCR é que a redução de dimensionalidade não é supervisionada e pode perder algumas variáveis ​​importantes: a PCR manteria os recursos com a maior variação, mas é possível que os recursos com pequenas variações sejam relevantes para prever o alvo. De certa forma, o PLS permite o mesmo tipo de redução de dimensionalidade, mas levando em consideração os alvos y. Uma ilustração deste fato é dada no exemplo a seguir: * Regressão de Componentes Principais vs Regressão de Mínimos Quadrados Parciais.

    # Além do CCA, os estimadores PLS são particularmente adequados quando a matriz de preditores possui mais variáveis ​​do que observações e quando há multicolinearidade entre as características. Por outro lado, a regressão linear padrão falharia nesses casos, a menos que fosse regularizada.

    # As classes incluídas neste módulo são PLSRegression, PLSCanonical, CCA e PLSSVD 