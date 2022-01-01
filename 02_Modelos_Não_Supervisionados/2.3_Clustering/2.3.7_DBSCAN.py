########## 2.3.7. DBSCAN ##########

    # O algoritmo DBSCAN vê os clusters como áreas de alta densidade separadas por áreas de baixa densidade. Devido a essa visão um tanto genérica, os clusters encontrados pelo DBSCAN podem ter qualquer formato, ao contrário de k-means, que assume que os clusters têm formato convexo. O componente central do DBSCAN é o conceito de amostras de núcleo, que são amostras que estão em áreas de alta densidade. Um cluster é, portanto, um conjunto de amostras de núcleo, cada uma próxima uma da outra (medida por alguma medida de distância) e um conjunto de amostras não essenciais que estão perto de uma amostra de núcleo (mas não são, elas mesmas, amostras de núcleo). Existem dois parâmetros para o algoritmo, min_samples e eps, que definem formalmente o que queremos dizer quando dizemos denso. Mais min_samples ou eps mais baixos indicam maior densidade necessária para formar um cluster.

    # Mais formalmente, definimos uma amostra de núcleo como sendo uma amostra no conjunto de dados de forma que existam min_samples outras amostras dentro de uma distância de eps, que são definidas como vizinhas da amostra de núcleo. Isso nos diz que a amostra do núcleo está em uma área densa do espaço vetorial. Um cluster é um conjunto de amostras de núcleo que podem ser construídas tomando recursivamente uma amostra de núcleo, encontrando todos os seus vizinhos que são amostras de núcleo, encontrando todos os seus vizinhos que são amostras de núcleo e assim por diante. Um cluster também tem um conjunto de amostras não essenciais, que são amostras vizinhas de uma amostra principal no cluster, mas não são em si mesmas amostras principais. Intuitivamente, essas amostras estão à margem de um cluster.

    # Qualquer amostra principal faz parte de um cluster, por definição. Qualquer amostra que não seja uma amostra de núcleo e esteja pelo menos eps em distância de qualquer amostra de núcleo é considerada um outlier pelo algoritmo.

    # Enquanto o parâmetro min_samples controla principalmente o quão tolerante o algoritmo é em relação ao ruído (em conjuntos de dados grandes e barulhentos, pode ser desejável aumentar este parâmetro), o parâmetro eps é crucial para escolher apropriadamente para o conjunto de dados e função de distância e geralmente não pode ser deixado no valor padrão. Ele controla a vizinhança local dos pontos. Quando escolhido muito pequeno, a maioria dos dados não será agrupada (e rotulada como -1 para “ruído”). Quando escolhido muito grande, ele faz com que os clusters próximos sejam mesclados em um cluster e, eventualmente, todo o conjunto de dados seja retornado como um único cluster. Algumas heurísticas para a escolha deste parâmetro foram discutidas na literatura, por exemplo, com base em um joelho no gráfico de distâncias do vizinho mais próximo (conforme discutido nas referências abaixo).

    # Na figura abaixo, a cor indica a associação do cluster, com grandes círculos indicando as amostras principais encontradas pelo algoritmo. Círculos menores são amostras não essenciais que ainda fazem parte de um cluster. Além disso, os outliers são indicados por pontos pretos abaixo. 

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html

    


    ## Exemplos:

    ## Demo of DBSCAN clustering algorithm (https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py)





    # Implementação

    # O algoritmo DBSCAN é determinístico, gerando sempre os mesmos clusters quando dados os mesmos dados na mesma ordem. No entanto, os resultados podem ser diferentes quando os dados são fornecidos em uma ordem diferente. Primeiro, embora as amostras principais sejam sempre atribuídas aos mesmos clusters, os rótulos desses clusters dependerão da ordem em que essas amostras são encontradas nos dados. Em segundo lugar, e mais importante, os clusters aos quais as amostras não essenciais são designadas podem diferir dependendo da ordem dos dados. Isso aconteceria quando uma amostra não essencial tem uma distância menor do que eps para duas amostras principais em clusters diferentes. Pela desigualdade triangular, essas duas amostras centrais devem estar mais distantes do que eps uma da outra, ou estariam no mesmo cluster. A amostra não principal é atribuída a qualquer cluster gerado primeiro em uma passagem pelos dados e, portanto, os resultados dependerão da ordem dos dados.

    # A implementação atual usa ball trees e kd-trees para determinar a vizinhança dos pontos, o que evita o cálculo da matriz de distância total (como era feito nas versões do scikit-learn antes de 0.14). A possibilidade de usar métricas personalizadas é mantida; para obter detalhes, consulte NearestNeighbors. 



    # Consumo de memória para grandes tamanhos de amostra

    # Esta implementação é, por padrão, não eficiente em termos de memória porque constrói uma matriz de similaridade completa de pares no caso em que kd-trees ou ball-trees não podem ser usados ​​(por exemplo, com matrizes esparsas). Esta matriz consumirá flutuadores. Alguns mecanismos para contornar isso são:

        # Use o agrupamento OPTICS em conjunto com o método extract_dbscan. O agrupamento OPTICS também calcula a matriz completa de pares, mas mantém apenas uma linha na memória por vez (complexidade de memória n).

        # Um gráfico de vizinhança com raio esparso (onde entradas ausentes são presumivelmente fora de eps) pode ser pré-computado de maneira eficiente em memória e dbscan pode ser executado sobre ele com metric = 'pré-computado'. Consulte sklearn.neighbors.NearestNeighbors.radius_neighs_graph.

        # O conjunto de dados pode ser compactado, removendo duplicatas exatas, se ocorrerem em seus dados, ou usando o BIRCH. Então você só tem um número relativamente pequeno de representantes para um grande número de pontos. Você pode então fornecer um sample_weight ao ajustar o DBSCAN. 



    ## Referências:

    ## “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise” Ester, M., H. P. Kriegel, J. Sander, and X. Xu, In Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996

    ## “DBSCAN revisited, revisited: why and how you should (still) use DBSCAN. Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017). In ACM Transactions on Database Systems (TODS), 42(3), 19.