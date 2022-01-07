########## 2.7.3. Detecção de outlier ##########

    # A detecção de outliers é semelhante à detecção de novidades no sentido de que o objetivo é separar um núcleo de observações regulares de algumas outras poluentes, chamadas outliers. Ainda assim, no caso de detecção de outlier, não temos um conjunto de dados limpo que representa a população de observações regulares que podem ser usadas para treinar qualquer ferramenta. 



##### 2.7.3.1. Ajustando um envelope elíptico

    # Uma maneira comum de realizar detecção de outlier é assumir que os dados regulares vêm de uma distribuição conhecida (por exemplo, os dados são distribuídos de Gauss). Partindo desse pressuposto, geralmente tentamos definir a “forma” dos dados e podemos definir observações externas como observações que estão longe o suficiente da forma de ajuste.

    # O scikit-learn fornece uma covariância de objeto.EllipticEnvelope que ajusta uma estimativa de covariância robusta aos dados e, portanto, ajusta uma elipse aos pontos de dados centrais, ignorando os pontos fora do modo central.

    # Por exemplo, assumindo que os dados inlier são distribuídos de Gauss, ele estimará a localização inlier e a covariância de uma forma robusta (ou seja, sem ser influenciado por outliers). As distâncias de Mahalanobis obtidas a partir desta estimativa são usadas para derivar uma medida de afastamento. Essa estratégia é ilustrada a seguir. 


        # https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html





    ## Exemplos:

    ## See Robust covariance estimation and Mahalanobis distances relevance for an illustration of the difference between using a standard (covariance.EmpiricalCovariance) or a robust estimate (covariance.MinCovDet) of location and covariance to assess the degree of outlyingness of an observation. (https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html#sphx-glr-auto-examples-covariance-plot-mahalanobis-distances-py)


    ## Referências:

    ## Rousseeuw, P.J., Van Driessen, K. “A fast algorithm for the minimum covariance determinant estimator” Technometrics 41(3), 212 (1999)



##### 2.7.3.2. Floresta de Isolamento

    # Uma maneira eficiente de realizar a detecção de valores discrepantes em conjuntos de dados de alta dimensão é usar florestas aleatórias. O ensemble.IsolationForest "isola" as observações ao selecionar aleatoriamente um recurso e, em seguida, selecionar aleatoriamente um valor dividido entre os valores máximo e mínimo do recurso selecionado.

    # Como o particionamento recursivo pode ser representado por uma estrutura de árvore, o número de divisões necessárias para isolar uma amostra é equivalente ao comprimento do caminho do nó raiz ao nó final.

    # O comprimento do caminho, calculado em uma floresta dessas árvores aleatórias, é uma medida de normalidade e nossa função de decisão.

    # O particionamento aleatório produz caminhos visivelmente mais curtos para anomalias. Portanto, quando uma floresta de árvores aleatórias produz coletivamente comprimentos de caminho mais curtos para amostras particulares, é muito provável que sejam anomalias. 


    # A implementação de ensemble.IsolationForest é baseada em um ensemble de tree.ExtraTreeRegressor. Seguindo o artigo original da Isolation Forest, a profundidade máxima de cada árvore é definida como \ lceil \ log_2 (n) \ rceil onde n é o número de amostras usadas para construir a árvore (ver (Liu et al., 2008) para mais detalhes).


    # Este algoritmo é ilustrado a seguir. 
     
        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html


    # O ensemble.IsolationForest oferece suporte a warm_start = True, que permite adicionar mais árvores a um modelo já ajustado: 


from sklearn.ensemble import IsolationForest
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [0, 0], [-20, 50], [3, 5]])
clf = IsolationForest(n_estimators=10, warm_start=True)
clf.fit(X)  # treina 10 árvores  
clf.set_params(n_estimators=20)  # adiciona mais 10 
clf.fit(X)  # treina as árvores adicionadas 




    ## Exemplos:

    ## See IsolationForest example for an illustration of the use of IsolationForest. (https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html#sphx-glr-auto-examples-ensemble-plot-isolation-forest-py)

    ## See Comparing anomaly detection algorithms for outlier detection on toy datasets for a comparison of ensemble.IsolationForest with neighbors.LocalOutlierFactor, svm.OneClassSVM (tuned to perform like an outlier detection method), linear_model.SGDOneClassSVM, and a covariance-based outlier detection with covariance.EllipticEnvelope. (https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html#sphx-glr-auto-examples-miscellaneous-plot-anomaly-comparison-py)



    ## Referências:

    ## Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. “Isolation forest.” Data Mining, 2008. ICDM’08. Eighth IEEE International Conference on.
     


##### 2.7.3.3. Fator Outlier Local 

    # Outra maneira eficiente de realizar detecção de valores discrepantes em conjuntos de dados dimensionais moderadamente altos é usar o algoritmo Fator de valores discrepantes locais (LOF).

    # O algoritmo neighbours.LocalOutlierFactor (LOF) calcula uma pontuação (chamado fator de outlier local) refletindo o grau de anormalidade das observações. Ele mede o desvio de densidade local de um determinado ponto de dados em relação a seus vizinhos. A ideia é detectar as amostras que possuem densidade substancialmente menor que suas vizinhas.

    # Na prática, a densidade local é obtida a partir dos k-vizinhos mais próximos. A pontuação LOF de uma observação é igual à razão entre a densidade local média de seus k-vizinhos mais próximos e sua própria densidade local: espera-se que uma instância normal tenha uma densidade local semelhante à de seus vizinhos, enquanto os dados anormais são espera-se que tenha uma densidade local muito menor.

    # O número k de vizinhos considerados, (parâmetro alias n_neighbors) é normalmente escolhido 1) maior do que o número mínimo de objetos que um cluster deve conter, de modo que outros objetos podem ser outliers locais em relação a este cluster, e 2) menores do que o máximo número de objetos próximos que podem ser potencialmente discrepantes locais. Na prática, essas informações geralmente não estão disponíveis, e tomar n_neighbours = 20 parece funcionar bem em geral. Quando a proporção de outliers é alta (ou seja, maior que 10%, como no exemplo abaixo), n_neighbors deve ser maior (n_neighbors = 35 no exemplo abaixo).

    # A força do algoritmo LOF é que ele leva em consideração as propriedades locais e globais dos conjuntos de dados: ele pode funcionar bem mesmo em conjuntos de dados onde as amostras anormais têm diferentes densidades subjacentes. A questão não é quão isolada é a amostra, mas quão isolada ela é em relação à vizinhança.

    # Ao aplicar LOF para detecção de outlier, não há métodos de previsão, função de decisão e amostra de pontuação, mas apenas um método de fit_predict. As pontuações de anormalidade das amostras de treinamento são acessíveis por meio do atributo negative_outlier_factor_. Observe que a previsão, a função de decisão e a amostra de pontuação podem ser usadas em novos dados não vistos quando LOF é aplicado para detecção de novidade, ou seja, quando o parâmetro de novidade é definido como Verdadeiro. Consulte Detecção de novidade com fator de outlier local.

    # Essa estratégia é ilustrada a seguir. 

        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html

    


    ## Exemplos:

    ## See Outlier detection with Local Outlier Factor (LOF) for an illustration of the use of neighbors.LocalOutlierFactor. (https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html#sphx-glr-auto-examples-neighbors-plot-lof-outlier-detection-py)

    ## See Comparing anomaly detection algorithms for outlier detection on toy datasets for a comparison with other anomaly detection methods. (https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html#sphx-glr-auto-examples-miscellaneous-plot-anomaly-comparison-py)



    ## Referências:

    ## Breunig, Kriegel, Ng, and Sander (2000) LOF: identifying density-based local outliers. Proc. ACM SIGMOD (http://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf)