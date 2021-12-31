########## 2.3. Clustering ##########

    # O agrupamento de dados não rotulados pode ser executado com o módulo sklearn.cluster.

    # Cada algoritmo de agrupamento vem em duas variantes: uma classe, que implementa o método de ajuste para aprender os clusters nos dados do treino, e uma função, que, dados os dados do treino, retorna uma matriz de rótulos inteiros correspondentes aos diferentes clusters. Para a classe, os rótulos sobre os dados de treinamento podem ser encontrados no atributo labels_.

    # Dados de entrada

    # Uma coisa importante a notar é que os algoritmos implementados neste módulo podem ter diferentes tipos de matriz como entrada. Todos os métodos aceitam matrizes de dados padrão de forma (n_samples, n_features). Eles podem ser obtidos nas classes do módulo sklearn.feature_extraction. Para AffinityPropagation, SpectralClustering e DBSCAN, também é possível inserir matrizes de similaridade de forma (n_samples, n_samples). Eles podem ser obtidos nas funções do módulo sklearn.metrics.pairwise. 