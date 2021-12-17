########## 1.10.5. Dicas de uso prático  ##########

    # As árvores de decisão tendem a se ajustar a dados com um grande número de recursos. Obter a proporção certa de amostras para o número de recursos é importante, uma vez que uma árvore com poucas amostras em um espaço dimensional elevado tem grande probabilidade de se ajustar demais.

    # Considere realizar a redução de dimensionalidade (PCA, ICA ou seleção de recurso) de antemão para dar à sua árvore uma chance melhor de encontrar recursos que são discriminativos.

    # Compreender a estrutura da árvore de decisão ajudará a obter mais insights sobre como a árvore de decisão faz previsões, o que é importante para entender os recursos importantes dos dados.


    # Visualize sua árvore durante o treinamento, usando a função de exportação. Use max_depth = 3 como uma profundidade inicial da árvore para ter uma ideia de como a árvore se ajusta aos seus dados e, a seguir, aumente a profundidade.

    # Lembre-se de que o número de amostras necessárias para preencher a árvore dobra para cada nível adicional até o qual a árvore cresce. Use max_depth para controlar o tamanho da árvore para evitar sobreajuste.

    # Use min_samples_split ou min_samples_leaf para garantir que várias amostras informem todas as decisões na árvore, controlando quais divisões serão consideradas. Um número muito pequeno geralmente significa que a árvore será ajustada demais, enquanto um número grande impedirá que a árvore aprenda os dados. Tente min_samples_leaf = 5 como um valor inicial. Se o tamanho da amostra variar muito, um número flutuante pode ser usado como porcentagem nesses dois parâmetros. Enquanto min_samples_split pode criar folhas arbitrariamente pequenas, min_samples_leaf garante que cada folha tenha um tamanho mínimo, evitando nós de folha de baixa variação e sobreajuste em problemas de regressão. Para classificação com poucas classes, min_samples_leaf = 1 é geralmente a melhor escolha.

    # Observe que min_samples_split considera as amostras diretamente e independente de sample_weight, se fornecido (por exemplo, um nó com m amostras ponderadas ainda é tratado como tendo exatamente m amostras). Considere min_weight_fraction_leaf ou min_impurity_decrease se a contabilização de pesos de amostra for necessária nas divisões.

    # Equilibre seu conjunto de dados antes do treinamento para evitar que a árvore seja influenciada pelas classes dominantes. O balanceamento de classes pode ser feito amostrando um número igual de amostras de cada classe ou, de preferência, normalizando a soma dos pesos da amostra (sample_weight) para cada classe para o mesmo valor. Observe também que os critérios de pré-poda com base no peso, como min_weight_fraction_leaf, serão menos tendenciosos para as classes dominantes do que os critérios que não conhecem os pesos da amostra, como min_samples_leaf.

    # Se as amostras forem ponderadas, será mais fácil otimizar a estrutura da árvore usando o critério de pré-poda baseado em peso, como min_weight_fraction_leaf, que garante que os nós de folha contenham pelo menos uma fração da soma geral dos pesos da amostra.

    # Todas as árvores de decisão usam matrizes np.float32 internamente. Se os dados de treinamento não estiverem neste formato, uma cópia do conjunto de dados será feita.

    # Se a matriz de entrada X for muito esparsa, é recomendado converter para csc_matrix esparsa antes de chamar fit e csr_matrix esparsa antes de chamar Predict. O tempo de treinamento pode ser muito mais rápido para uma entrada de matriz esparsa em comparação com uma matriz densa quando os recursos têm valores zero na maioria das amostras. 
     