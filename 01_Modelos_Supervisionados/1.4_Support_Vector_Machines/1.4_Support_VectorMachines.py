########## 1.4 Support Vector Machines ##########

    # Support vector machines (SVMs)  são um conjunto de métodos de aprendizado supervisionado usados para classificação, regressão e detecção de outliers. 

    # As vantagens de SVMs são:

        # Eficaz em espaços dimensionais elevados. 
        # Ainda eficaz nos casos em que o número de dimensões é maior do que o número de amostras.
        # Usa um subconjunto de pontos de treinamento na função de decisão (chamados de vetores de suporte), portanto, também é eficiente em termos de memória.
        # Versátil: diferentes funções do Kernel podem ser especificadas para a função de decisão. Kernels comuns são fornecidos, mas também é possível especificar kernels personalizados. 

    # As desvantagens dos SVMs incluem:

        # Se o número de recursos for muito maior do que o número de amostras, evite o sobreajuste ao escolher as funções do Kernel, pois o termo de regularização é crucial.
        # Os SVMs não fornecem estimativas de probabilidade diretamente, elas são calculadas usando uma validação cruzada quíntupla cara (consulte Pontuações e probabilidades, abaixo). 

    # Os SVMs no scikit-learn suportam vetores de amostra densos (numpy.ndarray e conversível para aquele por numpy.asarray) e esparsos (qualquer scipy.sparse) como entrada. No entanto, para usar um SVM para fazer previsões para dados esparsos, ele deve se ajustar a esses dados. Para obter o desempenho ideal, use numpy.ndarray ordenado por C (denso) ou scipy.sparse.csr_matrix (esparso) com dtype = float64. 