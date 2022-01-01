########## 2.3.9. BÉTULA ##########

    # O Birch constrói uma árvore chamada Clustering Feature Tree (CFT) para os dados fornecidos. Os dados são essencialmente compactados com perdas em um conjunto de nós de recurso de cluster (nós de CF). Os Nós CF têm vários subclusters chamados subclusters de Clustering Feature (CF Subclusters) e esses Subclusters CF localizados nos CF Nodes não terminais podem ter CF Nodes como filhos.

    # Os subclusters CF contêm as informações necessárias para o armazenamento em cluster, o que evita a necessidade de manter todos os dados de entrada na memória. Essas informações incluem:

        # Número de amostras em um subcluster.

        # Soma linear - um vetor n-dimensional que mantém a soma de todas as amostras

        # Soma quadrada - Soma da norma L2 quadrada de todas as amostras.

        # Centroids - Para evitar recálculo de soma linear / n_samples.

        # Norma quadrada dos centróides.

    # O algoritmo BIRCH tem dois parâmetros, o limite e o fator de ramificação. O fator de ramificação limita o número de subclusters em um nó e o limite limita a distância entre a amostra de entrada e os subclusters existentes.

    # Este algoritmo pode ser visto como uma instância ou método de redução de dados, uma vez que reduz os dados de entrada a um conjunto de subclusters que são obtidos diretamente das folhas do CFT. Esses dados reduzidos podem ser processados ​​posteriormente, alimentando-os em um clusterer global. Este clusterer global pode ser definido por n_clusters. Se n_clusters for definido como None, os subclusters das folhas são lidos diretamente, caso contrário, uma etapa de agrupamento global rotula esses subclusters em clusters globais (rótulos) e as amostras são mapeadas para o rótulo global do subcluster mais próximo. 


    # Descrição do algoritmo:

        # Uma nova amostra é inserida na raiz da Árvore CF, que é um Nó CF. Ele é então mesclado com o subcluster da raiz, que possui o menor raio após a mesclagem, restringido pelas condições de limite e fator de ramificação. Se o subcluster tiver qualquer nó filho, isso será feito repetidamente até atingir uma folha. Depois de localizar o subcluster mais próximo na folha, as propriedades deste subcluster e dos subclusters pai são atualizados recursivamente.

        # Se o raio do subaglomerado obtido pela fusão da nova amostra e o subaglomerado mais próximo for maior do que o quadrado do limite e se o número de subaglomerados for maior do que o fator de ramificação, um espaço é temporariamente alocado para esta nova amostra. Os dois subaglomerados mais distantes são obtidos e os subaglomerados são divididos em dois grupos com base na distância entre esses subaglomerados.

        # Se este nó dividido tiver um subcluster pai e houver espaço para um novo subcluster, o pai será dividido em dois. Se não houver espaço, esse nó será novamente dividido em dois e o processo continuará recursivamente, até atingir a raiz. 



    # BIRCH ou MiniBatchKMeans?

        # O BIRCH não se adapta muito bem a dados dimensionais elevados. Como regra geral, se n_features for maior que vinte, geralmente é melhor usar MiniBatchKMeans.

        # Se o número de instâncias de dados precisa ser reduzido, ou se alguém deseja um grande número de subclusters como uma etapa de pré-processamento ou não, o BIRCH é mais útil do que MiniBatchKMeans. 



    # Como usar o partial_fit?

    # Para evitar o cálculo de clustering global, para cada chamada de partial_fit o usuário é aconselhado

        # Para definir n_clusters = None inicialmente

        # Treine todos os dados por várias chamadas para partial_fit.

        # Defina n_clusters com um valor necessário usando brc.set_params (n_clusters = n_clusters).

        # Chame partial_fit finalmente sem argumentos, ou seja, brc.partial_fit () que executa o agrupamento global. 


    # https://scikit-learn.org/stable/auto_examples/cluster/plot_birch_vs_minibatchkmeans.html




    ## Referências:

    ## Tian Zhang, Raghu Ramakrishnan, Maron Livny BIRCH: An efficient data clustering method for large databases. https://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf (https://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf)

    ## Roberto Perdisci JBirch - Java implementation of BIRCH clustering algorithm https://code.google.com/archive/p/jbirch (https://code.google.com/archive/p/jbirch)