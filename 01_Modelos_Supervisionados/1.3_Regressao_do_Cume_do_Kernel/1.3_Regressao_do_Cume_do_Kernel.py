########## 1.3. Regressão do cume do kernel  ##########
    
    
    
    
    # A regressão de cume do kernel (KRR) [M2012] combina a regressão e a classificação Ridge (mínimos quadrados lineares com regularização de norma l2) com o truque do kernel. Assim, ele aprende uma função linear no espaço induzida pelo respectivo kernel e os dados. Para kernels não lineares, isso corresponde a uma função não linear no espaço original.

    # A forma do modelo aprendido pelo KernelRidge é idêntica à regressão vetorial de suporte (SVR). No entanto, diferentes funções de perda são usadas: KRR usa perda de erro ao quadrado, enquanto a regressão de vetor de suporte usa perda insensível, ambas combinadas com regularização l2. Ao contrário do SVR, o ajuste do KernelRidge pode ser feito de forma fechada e normalmente é mais rápido para conjuntos de dados de tamanho médio. Por outro lado, o modelo aprendido não é esparso e, portanto, mais lento que o SVR, que aprende um modelo esparso para , em tempo de previsão.

    # A figura a seguir compara KernelRidge e SVR em um conjunto de dados artificial, que consiste em uma função de destino senoidal e ruído forte adicionado a cada quinto ponto de dados. O modelo aprendido de KernelRidge e SVR é plotado, onde tanto a complexidade/regularização quanto a largura de banda do kernel RBF foram otimizadas usando grid-search. As funções aprendidas são muito semelhantes; no entanto, ajustar KernelRidge é aproximadamente sete vezes mais rápido do que ajustar SVR (ambos com pesquisa de grade). No entanto, a previsão de 100.000 valores de destino é mais de três vezes mais rápida com o SVR, pois ele aprendeu um modelo esparso usando apenas aproximadamente 1/3 dos 100 pontos de dados de treinamento como vetores de suporte. 

        # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html

    # A próxima figura compara o tempo de ajuste e previsão de KernelRidge e SVR para diferentes tamanhos do conjunto de treinamento. O ajuste do KernelRidge é mais rápido que o SVR para conjuntos de treinamento de tamanho médio (menos de 1.000 amostras); no entanto, para conjuntos de treinamento maiores, o SVR é melhor dimensionado. Com relação ao tempo de previsão, SVR é mais rápido que KernelRidge para todos os tamanhos do conjunto de treinamento devido à solução esparsa aprendida. Observe que o grau de esparsidade e, portanto, o tempo de previsão depende dos parâmetros e do SVR; corresponderia a um modelo denso. 

        # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html

    




    ## Referências:

    ## “Machine Learning: A Probabilistic Perspective” Murphy, K. P. - chapter 14.4.3, pp. 492-493, The MIT Press, 2012