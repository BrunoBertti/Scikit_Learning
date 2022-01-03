########## 2.4.3. Avaliação de Biclustering ##########

    # Existem duas maneiras de avaliar um resultado de bicluster: interno e externo. Medidas internas, como estabilidade do cluster, dependem apenas dos dados e dos próprios resultados. Atualmente, não há medidas de bicluster internas no scikit-learn. As medidas externas referem-se a uma fonte externa de informações, como a verdadeira solução. Ao trabalhar com dados reais, a verdadeira solução geralmente é desconhecida, mas o agrupamento de dados artificiais pode ser útil para avaliar algoritmos precisamente porque a verdadeira solução é conhecida.

    # Para comparar um conjunto de biclusters encontrados com o conjunto de biclusters verdadeiros, duas medidas de similaridade são necessárias: uma medida de similaridade para biclusters individuais e uma forma de combinar essas similaridades individuais em uma pontuação geral.

    # Para comparar biclusters individuais, várias medidas têm sido usadas. Por enquanto, apenas o índice Jaccard está implementado: 

        # J(A, B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}

    # onde A e B são biclusters, | A \ cap B | é o número de elementos em sua interseção. O índice Jaccard atinge seu mínimo de 0 quando os biclusters não se sobrepõem e seu máximo de 1 quando são idênticos.

    # Vários métodos foram desenvolvidos para comparar dois conjuntos de biclusters. Por enquanto, apenas consensus_score (Hochreiter et. Al., 2010) está disponível: 


        # Calcule as semelhanças do bicluster para pares de biclusters, um em cada conjunto, usando o índice de Jaccard ou uma medida semelhante.

        # Atribua biclusters de um conjunto a outro de maneira um a um para maximizar a soma de suas semelhanças. Esta etapa é executada usando o algoritmo húngaro.

        # A soma final das semelhanças é dividida pelo tamanho do conjunto maior.

    # A pontuação mínima de consenso, 0, ocorre quando todos os pares de biclusters são totalmente diferentes. A pontuação máxima, 1, ocorre quando os dois conjuntos são idênticos. 




    ## Referências:

    ## Hochreiter, Bodenhofer, et. al., 2010. FABIA: factor analysis for bicluster acquisition. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881408/)