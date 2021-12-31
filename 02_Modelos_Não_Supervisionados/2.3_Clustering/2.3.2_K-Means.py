########## 2.3.2. K-means ##########

    # O algoritmo KMeans agrupa os dados tentando separar as amostras em n grupos de variância igual, minimizando um critério conhecido como inércia ou soma dos quadrados dentro do cluster (veja abaixo). Este algoritmo requer que o número de clusters seja especificado. Ele se adapta bem a um grande número de amostras e tem sido usado em uma grande variedade de áreas de aplicação em muitos campos diferentes.

    # O algoritmo k-means divide um conjunto de N amostras X em K clusters C disjuntos, cada um descrito pela média \ mu_j das amostras no cluster. Os meios são comumente chamados de “centróides” do cluster; note que eles não são, em geral, pontos de X, embora vivam no mesmo espaço.


    # O algoritmo K-means visa escolher centróides que minimizem a inércia, ou o critério da soma dos quadrados dentro do cluster:


        # \ sum_ {i = 0} ^ {n} \ min _ {\ mu_j \ in C} (|| x_i - \ mu_j || ^ 2)


    # A inércia pode ser reconhecida como uma medida de quão internamente os clusters são coerentes. Ele sofre de várias desvantagens: 


        # A inércia pressupõe que os aglomerados são convexos e isotrópicos, o que nem sempre é o caso. Ele responde mal a aglomerados alongados ou variedades com formas irregulares.

        # A inércia não é uma métrica normalizada: apenas sabemos que valores mais baixos são melhores e zero é o ideal. Mas em espaços de dimensões muito altas, as distâncias euclidianas tendem a se tornar infladas (este é um exemplo da chamada “maldição da dimensionalidade”). Executar um algoritmo de redução de dimensionalidade, como análise de componente principal (PCA) antes do agrupamento k-means pode aliviar esse problema e acelerar os cálculos. 

            # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html



    # K-means é frequentemente referido como algoritmo de Lloyd. Em termos básicos, o algoritmo possui três etapas. A primeira etapa escolhe os centróides iniciais, com o método mais básico sendo escolher k amostras do conjunto de dados X. Após a inicialização, K-means consiste em fazer um loop entre as duas outras etapas. A primeira etapa atribui cada amostra ao seu centróide mais próximo. A segunda etapa cria novos centróides tomando o valor médio de todas as amostras atribuídas a cada centróide anterior. A diferença entre o antigo e o novo centróide é calculada e o algoritmo repete essas duas últimas etapas até que esse valor seja menor que um limite. Em outras palavras, ele se repete até que os centróides não se movam significativamente.


    # K-médias é equivalente ao algoritmo de maximização de expectativa com uma matriz de covariância diagonal pequena e totalmente igual.

    # O algoritmo também pode ser entendido por meio do conceito de diagramas de Voronoi. Primeiro, o diagrama de Voronoi dos pontos é calculado usando os centróides atuais. Cada segmento no diagrama de Voronoi torna-se um cluster separado. Em segundo lugar, os centróides são atualizados para a média de cada segmento. O algoritmo então repete isso até que um critério de parada seja atendido. Normalmente, o algoritmo para quando a diminuição relativa na função objetivo entre as iterações é menor que o valor de tolerância fornecido. Este não é o caso nesta implementação: a iteração para quando os centróides se movem menos do que a tolerância.

    # Com tempo suficiente, o K-médias sempre convergirá, no entanto, isso pode ser para um mínimo local. Isso é altamente dependente da inicialização dos centróides. Como resultado, o cálculo geralmente é feito várias vezes, com inicializações diferentes dos centróides. Um método para ajudar a resolver esse problema é o esquema de inicialização k-means ++, que foi implementado no scikit-learn (use o parâmetro init = 'k-means ++'). Isso inicializa os centróides para estar (geralmente) distantes um do outro, levando a resultados provavelmente melhores do que a inicialização aleatória, como mostrado na referência.

    # K-means ++ também pode ser chamado de forma independente para selecionar sementes para outros algoritmos de agrupamento, consulte sklearn.cluster.kmeans_plusplus para obter detalhes e exemplo de uso.

    # O algoritmo oferece suporte a pesos de amostra, que podem ser fornecidos por um parâmetro sample_weight. Isso permite atribuir mais peso a algumas amostras ao calcular centros de cluster e valores de inércia. Por exemplo, atribuir um peso de 2 a uma amostra é equivalente a adicionar uma duplicata dessa amostra ao conjunto de dados X.

    # As médias K podem ser usadas para quantização vetorial. Isso é obtido usando o método de transformação de um modelo treinado de KMeans. 




##### 2.3.2.1. Paralelismo de baixo nível

    # O KMeans se beneficia do paralelismo baseado em OpenMP por meio do Cython. Pequenos blocos de dados (256 amostras) são processados em paralelo, o que, além disso, gera uma baixa pegada de memória. Para obter mais detalhes sobre como controlar o número de threads, consulte nossas notas de paralelismo. 

    

    ## Exeplos:

    ## Demonstration of k-means assumptions: Demonstrating when k-means performs intuitively and when it does not (https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py)

    ## A demo of K-Means clustering on the handwritten digits data: Clustering handwritten digits (https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py)





    ## Referências:

    ## “k-means++: The advantages of careful seeding” Arthur, David, and Sergei Vassilvitskii, Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms, Society for Industrial and Applied Mathematics (2007) (http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)




##### 2.3.2.2. Mini lote K-médias 


    # O MiniBatchKMeans é uma variante do algoritmo KMeans que usa minilotes para reduzir o tempo de computação, ao mesmo tempo que tenta otimizar a mesma função objetivo. Os minilotes são subconjuntos dos dados de entrada, amostrados aleatoriamente em cada iteração de treinamento. Esses minilotes reduzem drasticamente a quantidade de computação necessária para convergir para uma solução local. Em contraste com outros algoritmos que reduzem o tempo de convergência de k-médias, o minilote k-médias produz resultados que geralmente são apenas ligeiramente piores do que o algoritmo padrão.

    # O algoritmo itera entre duas etapas principais, semelhante ao vanilla k-means. Na primeira etapa, b amostras são retiradas aleatoriamente do conjunto de dados, para formar um minilote. Em seguida, eles são atribuídos ao centróide mais próximo. Na segunda etapa, os centróides são atualizados. Em contraste com k-médias, isso é feito por amostra. Para cada amostra no minilote, o centróide atribuído é atualizado tomando a média de fluxo da amostra e todas as amostras anteriores atribuídas a esse centróide. Isso tem o efeito de diminuir a taxa de alteração de um centróide ao longo do tempo. Essas etapas são executadas até que a convergência ou um número predeterminado de iterações seja alcançado.

    # MiniBatchKMeans converge mais rápido do que KMeans, mas a qualidade dos resultados é reduzida. Na prática, essa diferença de qualidade pode ser bem pequena, conforme mostrado no exemplo e na referência citada. 

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html



    ## Exemplos:

    ## Comparison of the K-Means and MiniBatchKMeans clustering algorithms: Comparison of KMeans and MiniBatchKMeans (https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html#sphx-glr-auto-examples-cluster-plot-mini-batch-kmeans-py)

    ## Clustering text documents using k-means: Document clustering using sparse MiniBatchKMeans (https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py)

    ## Online learning of a dictionary of parts of faces (https://scikit-learn.org/stable/auto_examples/cluster/plot_dict_face_patches.html#sphx-glr-auto-examples-cluster-plot-dict-face-patches-py)




    ## Referências:

    ## “Web Scale K-Means clustering” D. Sculley, Proceedings of the 19th international conference on World wide web (2010) (https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)