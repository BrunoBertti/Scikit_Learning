########## 2.7.1. Visão geral dos métodos de detecção de outliers ##########


    # Uma comparação dos algoritmos de detecção de valores discrepantes no scikit-learn. O Fator de valor discrepante local (LOF) não mostra um limite de decisão em preto, pois não possui um método de previsão a ser aplicado em novos dados quando é usado para detecção de valores discrepantes. 

        # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html

    # ensemble.IsolationForest e neighbours.LocalOutlierFactor têm um desempenho razoavelmente bom nos conjuntos de dados considerados aqui. O svm.OneClassSVM é conhecido por ser sensível a outliers e, portanto, não funciona muito bem para detecção de outliers. Dito isso, a detecção de valores discrepantes em alta dimensão ou sem quaisquer suposições sobre a distribuição dos dados subjacentes é muito desafiadora. svm.OneClassSVM ainda pode ser usado com detecção de outliers, mas requer um ajuste fino de seu hiperparâmetro nu para lidar com outliers e evitar overfitting. linear_model.SGDOneClassSVM fornece uma implementação de um SVM linear de uma classe com uma complexidade linear no número de amostras. Esta implementação é usada aqui com uma técnica de aproximação de kernel para obter resultados semelhantes a svm.OneClassSVM, que usa um kernel Gaussiano por padrão. Finalmente, covariance.EllipticEnvelope assume que os dados são gaussianos e aprende uma elipse. Para obter mais detalhes sobre os diferentes estimadores, consulte o exemplo Comparando algoritmos de detecção de anomalias para detecção de valores discrepantes em conjuntos de dados de brinquedos e as seções abaixo. 



    ## Exemplos:

    ## See Comparing anomaly detection algorithms for outlier detection on toy datasets for a comparison of the svm.OneClassSVM, the ensemble.IsolationForest, the neighbors.LocalOutlierFactor and covariance.EllipticEnvelope. (https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html#sphx-glr-auto-examples-miscellaneous-plot-anomaly-comparison-py)