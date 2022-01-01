########## 2.3.5. Agrupamento espectral ##########

    # O SpectralClustering executa uma incorporação de baixa dimensão da matriz de afinidade entre as amostras, seguida por agrupamento, por exemplo, por KMeans, dos componentes dos autovetores no espaço de baixa dimensão. É especialmente computacionalmente eficiente se a matriz de afinidade for esparsa e o solucionador amg for usado para o problema de autovalor (observe que o solucionador amg requer que o módulo pyamg seja instalado.)

    # A versão atual do SpectralClustering requer que o número de clusters seja especificado com antecedência. Funciona bem para um pequeno número de clusters, mas não é recomendado para muitos clusters.

    # Para dois clusters, o SpectralClustering resolve um relaxamento convexo do problema de cortes normalizados no gráfico de similaridade: cortar o gráfico em dois para que o peso do corte das bordas seja pequeno em comparação com os pesos das bordas dentro de cada cluster. Este critério é especialmente interessante quando se trabalha em imagens, onde os vértices do gráfico são pixels e os pesos das bordas do gráfico de similaridade são calculados usando uma função de gradiente da imagem. 


        # https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html

    # Aviso: distância de transformação em semelhanças bem comportadas
    # Observe que se os valores de sua matriz de similaridade não estiverem bem distribuídos, por exemplo, com valores negativos ou com uma matriz de distância ao invés de uma similaridade, o problema espectral será singular e o problema não terá solução. Nesse caso, é aconselhável aplicar uma transformação às entradas da matriz. Por exemplo, no caso de uma matriz de distância sinalizada, é comum aplicar um kernel de calor:

    # similaridade = np.exp (-beta * distância / distância.std ()) 


    ## Exemplos:

    ## Spectral clustering for image segmentation: Segmenting objects from a noisy background using spectral clustering. (https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html#sphx-glr-auto-examples-cluster-plot-segmentation-toy-py)

    ## Segmenting the picture of greek coins in regions: Spectral clustering to split the image of coins in regions. (https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_segmentation.html#sphx-glr-auto-examples-cluster-plot-coin-segmentation-py)



##### 2.3.5.1. Diferentes estratégias de atribuição de rótulo

    # Diferentes estratégias de atribuição de rótulo podem ser usadas, correspondendo ao parâmetro assign_labels de SpectralClustering. A estratégia "kmeans" pode corresponder a detalhes mais sutis, mas pode ser instável. Em particular, a menos que você controle o random_state, ele pode não ser reproduzível de execução para execução, pois depende da inicialização aleatória. A estratégia alternativa de "discretizar" é 100% reproduzível, mas tende a criar parcelas de forma razoavelmente uniforme e geométrica. 

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_segmentation.html


##### 2.3.5.2. Gráficos de agrupamento espectral 

    # O clustering espectral também pode ser usado para particionar gráficos por meio de seus embeddings espectrais. Nesse caso, a matriz de afinidade é a matriz de adjacência do gráfico e o SpectralClustering é inicializado com afinidade = 'pré-computado': 


from sklearn.cluster import SpectralClustering
sc = SpectralClustering(3, affinity='precomputed', n_init=100,
                        assign_labels='discretize')
sc.fit_predict(adjacency_matrix)  


    ## Referências:

    ## “A Tutorial on Spectral Clustering” Ulrike von Luxburg, 2007 (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323)

    ## “Normalized cuts and image segmentation” Jianbo Shi, Jitendra Malik, 2000 (http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324)

    ## “A Random Walks View of Spectral Segmentation” Marina Meila, Jianbo Shi, 2001 (http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.33.1501)

    ## “On Spectral Clustering: Analysis and an algorithm” Andrew Y. Ng, Michael I. Jordan, Yair Weiss, 2001 (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.8100)

    ## “Preconditioned Spectral Clustering for Stochastic Block Partition Streaming Graph Challenge” David Zhuzhunashvili, Andrew Knyazev (https://arxiv.org/abs/1309.0238)