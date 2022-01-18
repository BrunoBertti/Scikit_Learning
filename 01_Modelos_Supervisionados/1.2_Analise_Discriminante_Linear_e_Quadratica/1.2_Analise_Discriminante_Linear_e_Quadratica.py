########## 1.2. Análise Discriminante Linear e Quadrática ##########


    # A Análise Discriminante Linear (LinearDiscriminantAnalysis) e a Análise Discriminante Quadrática (QuadraticDiscriminantAnalysis) são dois classificadores clássicos, com, como seus nomes sugerem, uma superfície de decisão linear e uma quadrática, respectivamente.

    # Esses classificadores são atraentes porque têm soluções de forma fechada que podem ser facilmente computadas, são inerentemente multiclasse, provaram funcionar bem na prática e não têm hiperparâmetros para ajustar. 

        # https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html

    # O gráfico mostra os limites de decisão para Análise Discriminante Linear e Análise Discriminante Quadrática. A linha inferior demonstra que a Análise Discriminante Linear só pode aprender limites lineares, enquanto a Análise Discriminante Quadrática pode aprender limites quadráticos e, portanto, é mais flexível. 


    ## Exemplos:

    ## Linear and Quadratic Discriminant Analysis with covariance ellipsoid: Comparison of LDA and QDA on synthetic data. (https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html#sphx-glr-auto-examples-classification-plot-lda-qda-py)