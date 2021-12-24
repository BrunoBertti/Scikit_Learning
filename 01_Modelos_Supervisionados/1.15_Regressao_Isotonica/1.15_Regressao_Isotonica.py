########## 1.15 Regressão_Isotônica ##########
 


    # A classe IsotonicRegression ajusta uma função real não decrescente para dados unidimensionais. Ele resolve o seguinte problema:


        # minimizar: \ sum_i w_i (y_i - \ hat {y} _i) ^ 2


        # sujeito a \ hat {y} _i \ le \ hat {y} _j sempre que X_i \ le X_j,


    # onde os pesos w_i são estritamente positivos, e tanto X quanto y são quantidades reais arbitrárias.

    # O parâmetro crescente muda a restrição para \ hat {y} _i \ ge \ hat {y} _j sempre que X_i \ le X_j. Configurá-lo como 'automático' escolherá automaticamente a restrição com base no coeficiente de correlação de classificação de Spearman.

    # IsotonicRegression produz uma série de previsões \ hat {y} _i para os dados de treinamento que são os mais próximos dos alvos y em termos de erro quadrático médio. Essas previsões são interpoladas para prever dados não vistos. As previsões de IsotonicRegression, portanto, formam uma função linear por partes: 

    # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_isotonic_regression.html