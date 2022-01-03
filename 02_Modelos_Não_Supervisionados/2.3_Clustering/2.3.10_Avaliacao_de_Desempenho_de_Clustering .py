########## 2.3.10. Avaliação de desempenho de clustering ##########

    # Avaliar o desempenho de um algoritmo de agrupamento não é tão trivial quanto contar o número de erros ou a precisão e recuperação de um algoritmo de classificação supervisionado. Em particular, qualquer métrica de avaliação não deve levar em consideração os valores absolutos dos rótulos do cluster, mas sim se esse agrupamento define separações dos dados semelhantes a algum conjunto de classes verdadeiras ou satisfazendo alguma suposição de que os membros pertencem à mesma classe são mais semelhantes do que membros de classes diferentes de acordo com alguma métrica de similaridade. 




##### 2.3.10.1. Índice de Rand

    # Dado o conhecimento das atribuições de classe de verdade fundamental labels_true e nossas atribuições de algoritmo de agrupamento das mesmas amostras labels_pred, o índice Rand (ajustado ou não ajustado) é uma função que mede a similaridade das duas atribuições, ignorando permutações: 

from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]
metrics.rand_score(labels_true, labels_pred)



    # O índice Rand não garante a obtenção de um valor próximo a 0,0 para uma marcação aleatória. O índice Rand ajustado corrige o acaso e fornecerá essa linha de base. 

metrics.adjusted_rand_score(labels_true, labels_pred)


    # Tal como acontece com todas as métricas de agrupamento, pode-se permutar 0 e 1 nos rótulos previstos, renomear 2 para 3 e obter a mesma pontuação: 

labels_pred = [1, 1, 0, 0, 3, 3]
metrics.rand_score(labels_true, labels_pred)
metrics.adjusted_rand_score(labels_true, labels_pred)

    # Além disso, ambos rand_score ajustados_rand_score são simétricos: trocar o argumento não altera as pontuações. Eles podem, portanto, ser usados como medidas de consenso: 

metrics.rand_score(labels_pred, labels_true)

metrics.adjusted_rand_score(labels_pred, labels_true)

    # A rotulagem perfeita é avaliada em 1,0: 

labels_pred = labels_true[:]
metrics.rand_score(labels_true, labels_pred)
metrics.adjusted_rand_score(labels_true, labels_pred)

    # Rótulos com concordância fraca (por exemplo, rótulos independentes) têm pontuações mais baixas, e para o índice Rand ajustado a pontuação será negativa ou próxima de zero. No entanto, para o índice Rand não ajustado, a pontuação, embora inferior, não será necessariamente próxima de zero: 

labels_true = [0, 0, 0, 0, 0, 0, 1, 1]
labels_pred = [0, 1, 2, 3, 4, 5, 5, 6]
metrics.rand_score(labels_true, labels_pred)
metrics.adjusted_rand_score(labels_true, labels_pred)


##### 2.3.10.1.1. Vantagens


    # Interpretabilidade: O índice Rand não ajustado é proporcional ao número de pares de amostra cujos rótulos são iguais em labels_pred e labels_true, ou são diferentes em ambos.

    # As atribuições de rótulos aleatórios (uniformes) têm uma pontuação de índice Rand ajustada próxima a 0,0 para qualquer valor de n_clusters e n_samples (o que não é o caso para o índice Rand não ajustado ou a medida V, por exemplo).

    # Intervalo limitado: valores mais baixos indicam classificações diferentes, agrupamentos semelhantes têm um índice Rand alto (ajustado ou não ajustado), 1,0 é a pontuação de correspondência perfeita. O intervalo de pontuação é [0, 1] para o índice Rand não ajustado e [-1, 1] para o índice Rand ajustado.

    # Nenhuma suposição é feita sobre a estrutura do cluster: O índice Rand (ajustado ou não ajustado) pode ser usado para comparar todos os tipos de algoritmos de agrupamento e pode ser usado para comparar algoritmos de agrupamento, como k-means que assume formas de blob isotrópicas com resultados espectrais algoritmos de agrupamento que podem encontrar agrupamentos com formas “dobradas”. 



##### 2.3.10.1.2. Inconvenientes


    # Ao contrário da inércia, o índice Rand (ajustado ou não ajustado) requer conhecimento das classes de verdade fundamental que quase nunca está disponível na prática ou requer atribuição manual por anotadores humanos (como no ambiente de aprendizado supervisionado).

    # No entanto, o índice Rand (ajustado ou não ajustado) também pode ser útil em um ambiente puramente não supervisionado como um bloco de construção para um Índice de Consenso que pode ser usado para seleção de modelo de agrupamento (TODO).

    # O índice Rand não ajustado costuma ser próximo a 1,0, mesmo se os próprios agrupamentos diferirem significativamente. Isso pode ser entendido ao interpretar o índice de Rand como a precisão da rotulagem do par de elementos resultante dos agrupamentos: Na prática, muitas vezes há uma maioria de pares de elementos que são atribuídos a rótulos de pares diferentes sob o agrupamento predito e de verdade fundamental, resultando em um alta proporção de rótulos de pares que concordam, o que leva subsequentemente a uma pontuação alta. 



    ## Exemplos:

    ## Adjustment for chance in clustering performance evaluation: Analysis of the impact of the dataset size on the value of clustering measures for random assignments. (https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html#sphx-glr-auto-examples-cluster-plot-adjusted-for-chance-measures-py)

##### 2.3.10.1.3. Formulação matemática


    # Se C é uma atribuição de classe de verdade fundamental e K o agrupamento, vamos definir a e b como:


    # a, o número de pares de elementos que estão no mesmo conjunto em C e no mesmo conjunto em K

    # b, o número de pares de elementos que estão em conjuntos diferentes em C e em conjuntos diferentes em K


    # O índice Rand não ajustado é então dado por:

        # \ text {RI} = \ frac {a + b} {C_2 ^ {n_ {amostras}}}


    # onde C_2 ^ {n_ {samples}} é o número total de pares possíveis no conjunto de dados. Não importa se o cálculo é executado em pares ordenados ou pares não ordenados, desde que o cálculo seja executado de forma consistente.


    # No entanto, o índice de Rand não garante que as atribuições de rótulos aleatórios obterão um valor próximo a zero (especialmente se o número de clusters estiver na mesma ordem de magnitude que o número de amostras).

    # Para contrariar este efeito, podemos descontar o RI E esperado [\ text {RI}] de rotulagens aleatórias, definindo o índice Rand ajustado da seguinte forma:


        # \ text {ARI} = \ frac {\ text {RI} - E [\ text {RI}]} {\ max (\ text {RI}) - E [\ text {RI}]}  



    ## Referências:

    ## Comparing Partitions L. Hubert and P. Arabie, Journal of Classification 1985 (https://link.springer.com/article/10.1007%2FBF01908075)

    ## Properties of the Hubert-Arabie adjusted Rand index D. Steinley, Psychological Methods 2004 (https://psycnet.apa.org/record/2004-17801-007)

    ## Wikipedia entry for the Rand index (https://en.wikipedia.org/wiki/Rand_index)

    ## Wikipedia entry for the adjusted Rand index (https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index)





##### 2.3.10.2. Pontuações baseadas em informações mútuas

    # Dado o conhecimento das atribuições de classe de verdade fundamental labels_true e nossas atribuições de algoritmo de agrupamento das mesmas amostras labels_pred, a Informação Mútua é uma função que mede a concordância das duas atribuições, ignorando permutações. Duas versões normalizadas diferentes desta medida estão disponíveis, Normalized Mutual Information (NMI) e Adjusted Mutual Information (AMI). O NMI é frequentemente usado na literatura, enquanto o AMI foi proposto mais recentemente e é normalizado contra o acaso: 

from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

metrics.adjusted_mutual_info_score(labels_true, labels_pred)  


    # Pode-se permutar 0 e 1 nos rótulos previstos, renomear 2 para 3 e obter a mesma pontuação: 

labels_pred = [1, 1, 0, 0, 3, 3]
metrics.adjusted_mutual_info_score(labels_true, labels_pred) 


    # Todos, mutual_info_score, Adjust_mutual_info_score e normalized_mutual_info_score são simétricos: trocar o argumento não altera a pontuação. Assim, eles podem ser usados como uma medida de consenso: 


metrics.adjusted_mutual_info_score(labels_pred, labels_true)  

    # A rotulagem perfeita é avaliada em 1,0: 

labels_pred = labels_true[:]
metrics.adjusted_mutual_info_score(labels_true, labels_pred)  

metrics.normalized_mutual_info_score(labels_true, labels_pred)  

    # Isso não é verdade para mutual_info_score, que, portanto, é mais difícil de julgar: 

metrics.mutual_info_score(labels_true, labels_pred)  


    # Ruim (por exemplo, rótulos independentes) têm pontuações não positivas: 

labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
metrics.adjusted_mutual_info_score(labels_true, labels_pred) 

##### 2.3.10.2.1. Vantagens

    # As atribuições de rótulos aleatórios (uniformes) têm uma pontuação AMI próxima de 0,0 para qualquer valor de n_clusters e n_samples (o que não é o caso para informações mútuas brutas ou a medida V, por exemplo).

    # Limite superior de 1: valores próximos de zero indicam duas atribuições de rótulos que são amplamente independentes, enquanto valores próximos a um indicam concordância significativa. Além disso, um AMI de exatamente 1 indica que as duas atribuições de rótulos são iguais (com ou sem permutação). 


##### 2.3.10.2.2. Inconvenientes

    # Ao contrário da inércia, as medidas baseadas em MI requerem o conhecimento das classes de verdade do solo, embora quase nunca estejam disponíveis na prática, ou requerem atribuição manual por anotadores humanos (como no ambiente de aprendizado supervisionado).
    # No entanto, as medidas baseadas em MI também podem ser úteis em um ambiente puramente não supervisionado como um bloco de construção para um Índice de Consenso que pode ser usado para seleção de modelo de agrupamento.

    # NMI e MI não são ajustados contra o acaso. 



    ## Exemplos:

    ## Adjustment for chance in clustering performance evaluation: Analysis of the impact of the dataset size on the value of clustering measures for random assignments. This example also includes the Adjusted Rand Index. (https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html#sphx-glr-auto-examples-cluster-plot-adjusted-for-chance-measures-py)

##### 2.3.10.2.3. Formulação matemática

    # Assuma duas atribuições de rótulo (dos mesmos N objetos), U e V. Sua entropia é a quantidade de incerteza para um conjunto de partição, definido por:

        # H (U) = - \ sum_ {i = 1} ^ {| U |} P (i) \ log (P (i))

    # onde P (i) = | U_i | / N é a probabilidade de que um objeto escolhido aleatoriamente de T caia na classe U_i. Da mesma forma para V:

        # H (V) = - \ sum_ {j = 1} ^ {| V |} P '(j) \ log (P' (j))


    # Com P '(j) = | V_j | / N. A informação mútua (MI) entre U e V é calculada por:

        # \ text {MI} (U, V) = \ sum_ {i = 1} ^ {| U |} \ sum_ {j = 1} ^ {| V |} P (i, j) \ log \ left (\ frac {P (i, j)} {P (i) P '(j)} \ direita)



    # onde P (i, j) = | U_i \ cap V_j | / N é a probabilidade de que um objeto escolhido aleatoriamente caia nas classes U_i e V_j.
    # Também pode ser expresso na formulação de cardinalidade definida:


        # \ text {MI} (U, V) = \ sum_ {i = 1} ^ {| U |} \ sum_ {j = 1} ^ {| V |} \ frac {| U_i \ cap V_j |} {N} \ log \ left (\ frac {N | U_i \ cap V_j |} {| U_i || V_j |} \ right)


    # A informação mútua normalizada é definida como

        # \ text {NMI} (U, V) = \ frac {\ text {MI} (U, V)} {\ text {média} (H (U), H (V))}

    # Este valor da informação mútua e também da variante normalizada não é ajustado ao acaso e tende a aumentar conforme o número de rótulos diferentes (clusters) aumenta, independentemente da quantidade real de “informações mútuas” entre as atribuições de rótulos.

    # O valor esperado para as informações mútuas pode ser calculado usando a seguinte equação [VEB2009]. Nesta equação, a_i = | U_i | (o número de elementos em U_i) e b_j = | V_j | (o número de elementos em V_j).

        # E [\ text {MI} (U, V)] = \ sum_ {i = 1} ^ {| U |} \ sum_ {j = 1} ^ {| V |} \ sum_ {n_ {ij} = (a_i + b_j-N) ^ +
        # } ^ {\ min (a_i, b_j)} \ frac {n_ {ij}} {N} \ log \ left (\ frac {N.n_ {ij}} {a_i b_j} \ right)
        # \ frac {a_i! b_j! (N-a_i)! (N-b_j)!} {N! n_ {ij}! (a_i-n_ {ij})! (b_j-n_ {ij})!
        # (N-a_i-b_j + n_ {ij})!}

    # Usando o valor esperado, a informação mútua ajustada pode então ser calculada usando um formulário semelhante ao do índice Rand ajustado:

        # \ text {AMI} = \ frac {\ text {MI} - E [\ text {MI}]} {\ text {média} (H (U), H (V)) - E [\ text {MI}] }

    # Para informação mútua normalizada e informação mútua ajustada, o valor de normalização é tipicamente alguma média generalizada das entropias de cada agrupamento. Existem vários meios generalizados, e não existem regras firmes para preferir um sobre os outros. A decisão é amplamente baseada em campo por campo; por exemplo, na detecção de comunidade, a média aritmética é a mais comum. Cada método de normalização fornece “comportamentos qualitativamente semelhantes” [YAT2016]. Em nossa implementação, isso é controlado pelo parâmetro average_method.

    # Vinh et al. (2010) denominaram variantes de NMI e AMI por seu método de média [VEB2010]. Suas médias 'sqrt' e 'soma' são os meios geométricos e aritméticos; usamos esses nomes mais amplamente comuns. 



    ## Referências: 

    ## Strehl, Alexander, and Joydeep Ghosh (2002). “Cluster ensembles – a knowledge reuse framework for combining multiple partitions”. Journal of Machine Learning Research 3: 583–617. doi:10.1162/153244303321897735. (http://strehl.com/download/strehl-jmlr02.pdf)

    ## Wikipedia entry for the (normalized) Mutual Information ( https://en.wikipedia.org/wiki/Mutual_Information)

    ## Wikipedia entry for the Adjusted Mutual Information (https://en.wikipedia.org/wiki/Adjusted_Mutual_Information)

    ## VEB2009 Vinh, Epps, and Bailey, (2009). “Information theoretic measures for clusterings comparison”. Proceedings of the 26th Annual International Conference on Machine Learning - ICML ‘09. doi:10.1145/1553374.1553511. ISBN 9781605585161. ( https://dl.acm.org/citation.cfm?doid=1553374.1553511)

    ## VEB2010 Vinh, Epps, and Bailey, (2010). “Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance”. JMLR <http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf> (http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf)

    ## YAT2016 Yang, Algesheimer, and Tessone, (2016). “A comparative analysis of community detection algorithms on artificial networks”. Scientific Reports 6: 30750. doi:10.1038/srep30750. (https://www.nature.com/articles/srep30750)

##### 2.3.10.3. Homogeneidade, completude e medida V


    # Dado o conhecimento das atribuições da classe de verdade fundamental das amostras, é possível definir algumas métricas intuitivas usando a análise de entropia condicional.

    # Em particular, Rosenberg e Hirschberg (2007) definem os seguintes dois objetivos desejáveis para qualquer atribuição de cluster:

        # homogeneidade: cada cluster contém apenas membros de uma única classe.

        # integridade: todos os membros de uma determinada classe são atribuídos ao mesmo cluster.

    # Podemos transformar esses conceitos em pontuações homogeneity_score e completeness_score. Ambos são limitados abaixo de 0,0 e acima de 1,0 (quanto maior, melhor): 

from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

metrics.homogeneity_score(labels_true, labels_pred)


metrics.completeness_score(labels_true, labels_pred)

    # Sua média harmônica, chamada de medida V, é calculada por v_measure_score: 

metrics.v_measure_score(labels_true, labels_pred)


    # A fórmula desta função é a seguinte:

        # v = \ frac {(1 + \ beta) \ times \ text {homogeneidade} \ times \ text {completude}} {(\ beta \ times \ text {homogeneidade} + \ text {completude})} 


    # O padrão do beta é um valor de 1,0, mas para usar um valor menor que 1 para o beta: 

metrics.v_measure_score(labels_true, labels_pred, beta=0.6)

    # mais peso será atribuído à homogeneidade, e usando um valor maior que 1: 

metrics.v_measure_score(labels_true, labels_pred, beta=1.8)


    # mais peso será atribuído à completude.

    # A medida V é realmente equivalente à informação mútua (NMI) discutida acima, com a função de agregação sendo a média aritmética [B2011].

    # Homogeneidade, integridade e medida V podem ser calculadas de uma só vez usando homogeneidade_completividade_v_medida da seguinte forma: 

metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)

    # A seguinte atribuição de clustering é um pouco melhor, pois é homogênea, mas não completa: 

labels_pred = [0, 0, 0, 1, 2, 2]
metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)


    # Nota: v_measure_score é simétrico: pode ser usado para avaliar a concordância de duas atribuições independentes no mesmo conjunto de dados.
    # Este não é o caso de completeness_score e homogeneity_score: ambos estão vinculados ao relacionamento:


##### 2.3.10.3.1. Vantagens


    # Pontuações limitadas: 0,0 é o máximo que pode ser, 1,0 é uma pontuação perfeita.

    # Interpretação intuitiva: o agrupamento com medição V ruim pode ser analisado qualitativamente em termos de homogeneidade e integridade para sentir melhor que 'tipo' de erros é cometido pela atribuição.

    # Nenhuma suposição é feita sobre a estrutura do cluster: pode ser usado para comparar algoritmos de agrupamento, como k-means, que assume formas de blob isotrópicas com resultados de algoritmos de agrupamento espectral que podem encontrar agrupamentos com formas "dobradas". 

##### 2.3.10.3.2. Inconvenientes

    # As métricas introduzidas anteriormente não são normalizadas com relação à rotulagem aleatória: isso significa que, dependendo do número de amostras, clusters e classes de verdade do solo, uma rotulagem completamente aleatória nem sempre produzirá os mesmos valores de homogeneidade, integridade e, portanto, medida v. Em particular, a rotulagem aleatória não renderá pontuação zero, especialmente quando o número de clusters é grande.

    # Este problema pode ser ignorado com segurança quando o número de amostras for maior que mil e o número de clusters for menor que 10. Para tamanhos de amostra menores ou número maior de clusters, é mais seguro usar um índice ajustado como o Índice Rand Ajustado ( ARI). 


        # https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html

    # Essas métricas requerem o conhecimento das classes de verdade terrestre, embora quase nunca estejam disponíveis na prática, ou exijam atribuição manual por anotadores humanos (como no ambiente de aprendizado supervisionado). 


    ## Exemplos:

    ## Adjustment for chance in clustering performance evaluation: Analysis of the impact of the dataset size on the value of clustering measures for random assignments. (https://scikit-learn.org/stable/auto_examples/cluster/plot_adjusted_for_chance_measures.html#sphx-glr-auto-examples-cluster-plot-adjusted-for-chance-measures-py)

##### 2.3.10.3.3. Formulação matemática

    # As pontuações de homogeneidade e completude são formalmente dadas por:

        # h = 1 - \ frac {H (C | K)} {H (C)}
        # c = 1 - \ frac {H (K | C)} {H (K)}

    # onde H (C | K) é a entropia condicional das classes dadas as atribuições do cluster e é dada por:

        # H (C | K) = - \ sum_ {c = 1} ^ {| C |} \ sum_ {k = 1} ^ {| K |} \ frac {n_ {c, k}} {n}
        # \ cdot \ log \ left (\ frac {n_ {c, k}} {n_k} \ right)


    # e H (C) é a entropia das classes e é dada por:

        # H (C) = - \ sum_ {c = 1} ^ {| C |} \ frac {n_c} {n} \ cdot \ log \ left (\ frac {n_c} {n} \ right)


    # com n o número total de amostras, n_c e n_k o número de amostras pertencentes respectivamente à classe ce cluster k e, finalmente, n_ {c, k} o número de amostras da classe c atribuídas ao cluster k.

    # A entropia condicional dos clusters dada classe H (K | C) e a entropia dos clusters H (K) são definidas de maneira simétrica.

    # Rosenberg e Hirschberg definem ainda mais a medida V como a média harmônica de homogeneidade e integridade:

        # v = 2 \ cdot \ frac {h \ cdot c} {h + c} 


    

    ## Referências:
    ## V-Measure: A conditional entropy-based external cluster evaluation measure Andrew Rosenberg and Julia Hirschberg, 2007  (https://aclweb.org/anthology/D/D07/D07-1043.pdf)

    ## B2011Identication and Characterization of Events in Social Media, Hila Becker, PhD Thesis. (http://www.cs.columbia.edu/~hila/hila-thesis-distributed.pdf)


##### 2.3.10.4. Placares de Fowlkes-Mallows

    # O índice de Fowlkes-Mallows (sklearn.metrics.fowlkes_mallows_score) pode ser usado quando as atribuições de classe de verdade fundamental das amostras são conhecidas. A pontuação de Fowlkes-Mallows FMI é definida como a média geométrica da precisão dos pares e rechamada:

        # \ text {FMI} = \ frac {\ text {TP}} {\ sqrt {(\ text {TP} + \ text {FP}) (\ text {TP} + \ text {FN})}}


    # Onde TP é o número de Verdadeiros Positivos (ou seja, o número de pares de pontos que pertencem aos mesmos clusters em ambos os rótulos verdadeiros e preditos), FP é o número de Falso Positivo (ou seja, o número de pares de pontos que pertencem para os mesmos clusters nos rótulos verdadeiros e não nos rótulos previstos) e FN é o número de Falso Negativo (ou seja, o número de pares de pontos que pertencem aos mesmos clusters nos rótulos previstos e não nos rótulos verdadeiros).

    # A pontuação varia de 0 a 1. Um valor alto indica uma boa semelhança entre dois clusters. 

from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

metrics.fowlkes_mallows_score(labels_true, labels_pred)

    # Pode-se permutar 0 e 1 nos rótulos previstos, renomear 2 para 3 e obter a mesma pontuação: 

labels_pred = [1, 1, 0, 0, 3, 3]

metrics.fowlkes_mallows_score(labels_true, labels_pred)


    # A rotulagem perfeita é avaliada em 1,0: 

labels_pred = labels_true[:]
metrics.fowlkes_mallows_score(labels_true, labels_pred)

    # Ruim (por exemplo, rótulos independentes) têm pontuação zero: 

labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
metrics.fowlkes_mallows_score(labels_true, labels_pred)


##### 2.3.10.4.1. Vantagens

    # As atribuições de rótulos aleatórios (uniformes) têm uma pontuação FMI próxima a 0,0 para qualquer valor de n_clusters e n_samples (o que não é o caso para informações mútuas brutas ou a medida V, por exemplo).

    # Limite superior em 1: valores próximos de zero indicam duas atribuições de rótulos que são amplamente independentes, enquanto valores próximos a um indicam concordância significativa. Além disso, valores de exatamente 0 indicam atribuições de rótulo puramente independentes e um FMI de exatamente 1 indica que as duas atribuições de rótulo são iguais (com ou sem permutação).

    # Nenhuma suposição é feita sobre a estrutura do cluster: pode ser usado para comparar algoritmos de agrupamento, como k-means, que assume formas de blob isotrópicas com resultados de algoritmos de agrupamento espectral que podem encontrar agrupamentos com formas "dobradas"

##### 2.3.10.4.2. Inconvenientes


    # Ao contrário da inércia, as medidas baseadas no FMI requerem o conhecimento das classes de verdade do solo, embora quase nunca estejam disponíveis na prática, ou requerem atribuição manual por anotadores humanos (como no ambiente de aprendizado supervisionado). 


    ## Referências:

    ## E. B. Fowkles and C. L. Mallows, 1983. “A method for comparing two hierarchical clusterings”. Journal of the American Statistical Association. https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478008 (https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478008)

    ## https://en.wikipedia.org/wiki/Fowlkes-Mallows_index



##### 2.3.10.5. Coeficiente de silhueta

    # Se os rótulos de verdade do terreno não forem conhecidos, a avaliação deve ser realizada usando o próprio modelo. O Coeficiente Silhouette (sklearn.metrics.silhouette_score) é um exemplo de tal avaliação, onde uma pontuação mais alta do Coeficiente Silhouette está relacionada a um modelo com clusters mais bem definidos. O Coeficiente de Silhueta é definido para cada amostra e é composto por duas pontuações: 

        # a: A distância média entre uma amostra e todos os outros pontos da mesma classe.

        # b: A distância média entre uma amostra e todos os outros pontos no próximo cluster mais próximo. 

    # O coeficiente de silhueta para uma única amostra é então dado como: 

        # s = \frac{b - a}{max(a, b)}

    # O coeficiente de silhueta para um conjunto de amostras é dado como a média do coeficiente de silhueta para cada amostra. 


from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
X, y = datasets.load_iris(return_X_y=True)

    # Em uso normal, o Coeficiente Silhouette é aplicado aos resultados de uma análise de cluster. 

import numpy as np
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
metrics.silhouette_score(X, labels, metric='euclidean')


    ## Referências:

    ## Peter J. Rousseeuw (1987). “Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis” . Computational and Applied Mathematics 20: 53–65. (https://doi.org/10.1016/0377-0427(87)90125-7)


##### 2.3.10.5.1. Vantagens


    # A pontuação é limitada entre -1 para agrupamento incorreto e +1 para agrupamento altamente denso. Pontuações em torno de zero indicam clusters sobrepostos.

    # A pontuação é maior quando os clusters são densos e bem separados, o que se relaciona a um conceito padrão de cluster. 

##### 2.3.10.5.2. Inconvenientes

    # O Coeficiente Silhouette é geralmente mais alto para clusters convexos do que outros conceitos de clusters, como clusters baseados em densidade, como aqueles obtidos por meio do DBSCAN. 



    ## Exemplos:

    ## Selecting the number of clusters with silhouette analysis on KMeans clustering : In this example the silhouette analysis is used to choose an optimal value for n_clusters. (https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py)


##### 2.3.10.6. Índice Calinski-Harabasz

    # Se os rótulos verdadeiros não forem conhecidos, o índice Calinski-Harabasz (sklearn.metrics.calinski_harabasz_score) - também conhecido como Critério da Razão de Variância - pode ser usado para avaliar o modelo, onde uma pontuação Calinski-Harabasz mais alta se relaciona a um modelo com clusters mais bem definidos.

    # O índice é a razão da soma da dispersão entre os clusters e da dispersão dentro do cluster para todos os clusters (onde a dispersão é definida como a soma das distâncias ao quadrado): 


from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
X, y = datasets.load_iris(return_X_y=True)

    # Em uso normal, o índice Calinski-Harabasz é aplicado aos resultados de uma análise de cluster: 

import numpy as np
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
metrics.calinski_harabasz_score(X, labels)

##### 2.3.10.6.1. Vantagens

    # A pontuação é maior quando os clusters são densos e bem separados, o que se relaciona a um conceito padrão de cluster.

    # A pontuação é rápida de calcular. 



##### 2.3.10.6.2. Inconvenientes


    # O índice Calinski-Harabasz é geralmente mais alto para clusters convexos do que outros conceitos de clusters, como clusters baseados em densidade, como aqueles obtidos por meio do DBSCAN. 

##### 2.3.10.6.3. Formulação matemática

    # Para um conjunto de dados E de tamanho n_E que foi agrupado em k clusters, a pontuação de Calinski-Harabasz é definida como a razão entre a média de dispersão entre clusters e a dispersão dentro do cluster:

        # s = \ frac {\ mathrm {tr} (B_k)} {\ mathrm {tr} (W_k)} \ times \ frac {n_E - k} {k - 1}


    # onde \ mathrm {tr} (B_k) é o traço da matriz de dispersão entre grupos e \ mathrm {tr} (W_k) é o traço da matriz de dispersão dentro do cluster definida por:

        # W_k = \ sum_ {q = 1} ^ k \ sum_ {x \ em C_q} (x - c_q) (x - c_q) ^ T

        # B_k = \ sum_ {q = 1} ^ k n_q (c_q - c_E) (c_q - c_E) ^ T


    # com C_q o conjunto de pontos no cluster q, c_q o centro do cluster q, c_E o centro de E e n_q o número de pontos no cluster q. 



    ## Referências:

    ## Caliński, T., & Harabasz, J. (1974). “A Dendrite Method for Cluster Analysis”. Communications in Statistics-theory and Methods 3: 1-27. (https://www.researchgate.net/publication/233096619_A_Dendrite_Method_for_Cluster_Analysis)


##### 2.3.10.7. Índice Davies-Bouldin

    # Se os rótulos de verdade não forem conhecidos, o índice Davies-Bouldin (sklearn.metrics.davies_bouldin_score) pode ser usado para avaliar o modelo, onde um índice Davies-Bouldin mais baixo se relaciona a um modelo com melhor separação entre os clusters.

    # Este índice significa a "similaridade" média entre os clusters, onde a similaridade é uma medida que compara a distância entre os clusters com o tamanho dos próprios clusters.

    # Zero é a pontuação mais baixa possível. Valores próximos de zero indicam uma partição melhor.

    # Em uso normal, o índice Davies-Bouldin é aplicado aos resultados de uma análise de cluster da seguinte forma: 



from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
kmeans = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans.labels_
davies_bouldin_score(X, labels)



##### 2.3.10.7.1. Vantagens


    # O cálculo de Davies-Bouldin é mais simples do que a pontuação do Silhouette.

    # O índice é baseado exclusivamente em quantidades e recursos inerentes ao conjunto de dados, pois seu cálculo usa apenas distâncias pontuais. 



##### 2.3.10.7.2. Inconvenientes

    # O índice Davies-Boulding é geralmente mais alto para clusters convexos do que outros conceitos de clusters, como clusters baseados em densidade, como os obtidos do DBSCAN.

    # O uso da distância do centroide limita a métrica da distância ao espaço euclidiano. 

##### 2.3.10.7.3. Formulação matemática

    # O índice é definido como a semelhança média entre cada cluster C_i para i = 1, ..., ke seu mais semelhante C_j. No contexto deste índice, a similaridade é definida como uma medida R_ {ij} que negocia: 

        # s_i, a distância média entre cada ponto do cluster i e o centróide desse cluster - também conhecido como diâmetro do cluster.

        # d_ {ij}, a distância entre os centróides do cluster i e j. 

    # Uma escolha simples para construir R_ {ij} de modo que seja não negativo e simétrico é: 

        # R_{ij} = \frac{s_i + s_j}{d_{ij}}

    # Então, o índice Davies-Bouldin é definido como: 

        # DB = \frac{1}{k} \sum_{i=1}^k \max_{i \neq j} R_{ij}



    ## Referências:

    ## Davies, David L.; Bouldin, Donald W. (1979). “A Cluster Separation Measure” IEEE Transactions on Pattern Analysis and Machine Intelligence. PAMI-1 (2): 224-227. (https://doi.org/10.1109/TPAMI.1979.4766909)

    ## Halkidi, Maria; Batistakis, Yannis; Vazirgiannis, Michalis (2001). “On Clustering Validation Techniques” Journal of Intelligent Information Systems, 17(2-3), 107-145. (https://doi.org/10.1023/A:1012801612483)

    ## Wikipedia entry for Davies-Bouldin index. (https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index)



##### 2.3.10.8. Matriz de Contingência

    # A matriz de contingência (sklearn.metrics.cluster.contingency_matrix) relata a cardinalidade de interseção para cada par de cluster verdadeiro / previsto. A matriz de contingência fornece estatísticas suficientes para todas as métricas de agrupamento onde as amostras são independentes e distribuídas de forma idêntica e não é necessário contabilizar algumas instâncias que não estão sendo agrupadas.

    # Aqui está um exemplo: 

from sklearn.metrics.cluster import contingency_matrix
x = ["a", "a", "a", "b", "b", "b"]
y = [0, 0, 1, 1, 2, 2]
contingency_matrix(x, y)


    # A primeira linha da matriz de saída indica que há três amostras cujo verdadeiro cluster é “a”. Destes, dois estão no cluster predito 0, um está em 1 e nenhum está em 2. E a segunda linha indica que há três amostras cujo verdadeiro cluster é “b”. Deles, nenhum está no cluster 0 previsto, um está no 1 e dois estão no 2.

    # Uma matriz de confusão para classificação é uma matriz de contingência quadrada em que a ordem das linhas e colunas corresponde a uma lista de classes. 


##### 2.3.10.8.1. Vantagens

    # Permite examinar a distribuição de cada cluster verdadeiro entre os clusters previstos e vice-versa.

    # A tabela de contingência calculada é normalmente utilizada no cálculo de uma estatística de similaridade (como as outras listadas neste documento) entre os dois agrupamentos. 

##### 2.3.10.8.2. Inconvenientes

    # A matriz de contingência é fácil de interpretar para um pequeno número de clusters, mas torna-se muito difícil de interpretar para um grande número de clusters.

    # Não fornece uma única métrica para usar como um objetivo para a otimização de agrupamento. 



    ## Referências:

    ## Wikipedia entry for contingency matrix (https://en.wikipedia.org/wiki/Contingency_table)

##### 2.3.10.9. Matriz de confusão de pares 

    # The pair confusion matrix (sklearn.metrics.cluster.pair_confusion_matrix) is a 2x2 similarity matrix

        # \begin{split}C = \left[\begin{matrix}
        # C_{00} & C_{01} \\
        # C_{10} & C_{11}
        # \end{matrix}\right]\end{split}

    # entre dois agrupamentos calculados considerando todos os pares de amostras e contando pares que são atribuídos ao mesmo ou em agrupamentos diferentes sob os agrupamentos verdadeiros e preditos.

    # Possui as seguintes entradas: 

        # C_ {00}: número de pares com ambos os agrupamentos tendo as amostras não agrupadas

        # C_ {10}: número de pares com o verdadeiro rótulo de cluster tendo as amostras agrupadas, mas o outro agrupamento não tendo as amostras agrupadas

        # C_ {01}: número de pares com o verdadeiro rótulo de cluster não tendo as amostras agrupadas, mas o outro agrupamento tendo as amostras agrupadas

        # C_ {11}: número de pares com ambos os agrupamentos tendo as amostras agrupadas

        # Considerando um par de amostras agrupadas em um par positivo, então, como na classificação binária, a contagem de verdadeiros negativos é C_ {00}, falsos negativos é C_ {10}, verdadeiros positivos é C_ {11} e falsos positivos é C_ { 01}.

        # Os rótulos de correspondência perfeita têm todas as entradas diferentes de zero na diagonal, independentemente dos valores reais dos rótulos:     


from sklearn.metrics.cluster import pair_confusion_matrix
pair_confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1])


pair_confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0])

    # As marcações que atribuem todos os membros das classes aos mesmos clusters são completas, mas nem sempre podem ser puras, portanto, penalizadas e têm algumas entradas fora da diagonal diferentes de zero: 


pair_confusion_matrix([0, 0, 1, 2], [0, 0, 1, 1])

    # A matriz não é simétrica: 

pair_confusion_matrix([0, 0, 1, 1], [0, 0, 1, 2])


    # Se os membros das classes estão completamente divididos em diferentes clusters, a atribuição é totalmente incompleta, portanto, a matriz tem todas as entradas diagonais zero: 

pair_confusion_matrix([0, 0, 0, 0], [0, 1, 2, 3])


    ## Referências:

    ## L. Hubert and P. Arabie, Comparing Partitions, Journal of Classification 1985 <https://link.springer.com/article/10.1007%2FBF01908075>_ (https://link.springer.com/article/10.1007%2FBF01908075)