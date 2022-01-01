########## 2.3.6. Agrupamento hierárquico ##########

    # O clustering hierárquico é uma família geral de algoritmos de clustering que constroem clusters aninhados mesclando-os ou dividindo-os sucessivamente. Esta hierarquia de clusters é representada como uma árvore (ou dendrograma). A raiz da árvore é o único cluster que reúne todas as amostras, sendo as folhas os aglomerados com apenas uma amostra. Veja a página da Wikipedia para mais detalhes.

    # O objeto AgglomerativeClustering realiza um agrupamento hierárquico usando uma abordagem ascendente: cada observação começa em seu próprio cluster, e os clusters são sucessivamente mesclados. O critério de ligação determina a métrica usada para a estratégia de fusão: 

        # Ward minimiza a soma das diferenças quadradas em todos os clusters. É uma abordagem de minimização de variância e, neste sentido, é semelhante à função objetivo k-médias, mas tratada com uma abordagem hierárquica aglomerativa.

        # A ligação máxima ou completa minimiza a distância máxima entre as observações de pares de agrupamentos.

        # A ligação média minimiza a média das distâncias entre todas as observações de pares de clusters.

        # A ligação única minimiza a distância entre as observações mais próximas de pares de clusters. 

    # AgglomerativeClustering também pode escalar para um grande número de amostras quando é usado em conjunto com uma matriz de conectividade, mas é computacionalmente caro quando nenhuma restrição de conectividade é adicionada entre as amostras: ele considera em cada etapa todas as combinações possíveis.

    # FeatureAgglomeration

        # O FeatureAgglomeration usa clustering aglomerativo para agrupar recursos que parecem muito semelhantes, diminuindo assim o número de recursos. É uma ferramenta de redução de dimensionalidade, consulte Redução de dimensionalidade não supervisionada. 



##### 2.3.6.1. Tipo de ligação diferente: ligação, ligação completa, média e ligação única

    # AgglomerativeClustering oferece suporte a estratégias de vinculação de Ward, simples, média e completa. 

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html

    # O cluster aglomerativo tem um comportamento de “enriquecimento cada vez maior” que leva a tamanhos de cluster desiguais. Nesse sentido, a ligação simples é a pior estratégia e Ward fornece os tamanhos mais regulares. No entanto, a afinidade (ou distância usada no agrupamento) não pode ser variada com Ward, portanto, para métricas não euclidianas, a ligação média é uma boa alternativa. A ligação única, embora não seja robusta para dados ruidosos, pode ser calculada com muita eficiência e, portanto, pode ser útil para fornecer agrupamento hierárquico de conjuntos de dados maiores. A ligação única também pode funcionar bem em dados não globulares. 


    ## Exemplos:

    ## Various Agglomerative Clustering on a 2D embedding of digits: exploration of the different linkage strategies in a real dataset. (https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py)


##### 2.3.6.2. Visualização da hierarquia do cluster

    # É possível visualizar a árvore que representa a fusão hierárquica dos clusters como um dendrograma. A inspeção visual pode muitas vezes ser útil para compreender a estrutura dos dados, embora ainda mais no caso de tamanhos de amostra pequenos. 

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html


##### 2.3.6.3. Adicionando restrições de conectividade

    # Um aspecto interessante do AgglomerativeClustering é que restrições de conectividade podem ser adicionadas a este algoritmo (apenas clusters adjacentes podem ser mesclados), por meio de uma matriz de conectividade que define para cada amostra as amostras vizinhas seguindo uma determinada estrutura de dados. Por exemplo, no exemplo do rolo suíço abaixo, as restrições de conectividade proíbem a fusão de pontos que não são adjacentes ao rolo suíço e, assim, evitam a formação de aglomerados que se estendem por dobras sobrepostas do rolo.

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html


    # Essas restrições são úteis para impor uma determinada estrutura local, mas também tornam o algoritmo mais rápido, principalmente quando o número de amostras é alto.

    # As restrições de conectividade são impostas por meio de uma matriz de conectividade: uma matriz scipy sparse que possui elementos apenas na interseção de uma linha e uma coluna com índices do conjunto de dados que deve ser conectado. Esta matriz pode ser construída a partir de informações a priori: por exemplo, você pode desejar agrupar páginas da web apenas mesclando páginas com um link apontando de uma para outra. Também pode ser aprendido a partir dos dados, por exemplo, usando sklearn.neighbors.kneighbors_graph para restringir a fusão aos vizinhos mais próximos como neste exemplo, ou usando sklearn.feature_extraction.image.grid_to_graph para permitir apenas a fusão de pixels vizinhos em uma imagem, como em o exemplo da moeda. 



    ## Exemplos:

    ## A demo of structured Ward hierarchical clustering on an image of coins: Ward clustering to split the image of coins in regions. (https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_ward_segmentation.html#sphx-glr-auto-examples-cluster-plot-coin-ward-segmentation-py)

    ## Hierarchical clustering: structured vs unstructured ward: Example of Ward algorithm on a swiss-roll, comparison of structured approaches versus unstructured approaches. (https://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html#sphx-glr-auto-examples-cluster-plot-ward-structured-vs-unstructured-py)

    ## Feature agglomeration vs. univariate selection: Example of dimensionality reduction with feature agglomeration based on Ward hierarchical clustering. (https://scikit-learn.org/stable/auto_examples/cluster/plot_feature_agglomeration_vs_univariate_selection.html#sphx-glr-auto-examples-cluster-plot-feature-agglomeration-vs-univariate-selection-py)

    ## Agglomerative clustering with and without structure (https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py)



    # Aviso: Restrições de conectividade com ligação única, média e completa
    # Restrições de conectividade e ligação única, completa ou média podem aprimorar o aspecto de "enriquecimento cada vez mais rico" do agrupamento aglomerativo, especialmente se forem construídos com sklearn.neighbors.kneighbors_graph. No limite de um pequeno número de clusters, tendem a dar alguns clusters ocupados macroscopicamente e outros quase vazios. (veja a discussão em Agrupamento aglomerativo com e sem estrutura). A ligação simples é a opção de ligação mais frágil com relação a esse problema. 

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html



##### 2.3.6.4. Variando a métrica 


    # A ligação única, média e completa pode ser usada com uma variedade de distâncias (ou afinidades), em particular distância euclidiana (l2), distância de Manhattan (ou Cityblock ou l1), distância cosseno ou qualquer matriz de afinidade pré-computada.

        # A distância l1 costuma ser boa para recursos esparsos ou ruídos esparsos: ou seja, muitos dos recursos são zero, como na mineração de texto usando ocorrências de palavras raras.

        # a distância do cosseno é interessante porque é invariante para as escalas globais do sinal.

    # As diretrizes para escolher uma métrica é usar uma que maximize a distância entre as amostras em diferentes classes e a minimize dentro de cada classe. 


        # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering_metrics.html


    ## Exemplos:


    ## Agglomerative clustering with different metrics (https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering_metrics.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-metrics-py)