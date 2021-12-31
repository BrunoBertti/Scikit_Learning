########## 2.3.3. Propagação de afinidade ##########

    # AffinityPropagation cria clusters enviando mensagens entre pares de amostras até a convergência. Um conjunto de dados é então descrito usando um pequeno número de exemplares, que são identificados como aqueles mais representativos de outras amostras. As mensagens enviadas entre pares representam a adequação de uma amostra a exemplar da outra, que é atualizada em resposta aos valores de outros pares. Essa atualização acontece iterativamente até a convergência, momento em que os exemplares finais são escolhidos e, portanto, o agrupamento final é dado. 


        # https://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html


    # A Propagação de Afinidade pode ser interessante, pois escolhe o número de clusters com base nos dados fornecidos. Para tanto, os dois parâmetros importantes são a preferência, que controla quantos exemplares são usados, e o fator de amortecimento que amortece as mensagens de responsabilidade e disponibilidade para evitar oscilações numéricas na atualização dessas mensagens.

    # A principal desvantagem da Propagação de Afinidade é sua complexidade. O algoritmo possui uma complexidade de tempo da ordem O (N ^ 2 T), onde N é o número de amostras e T é o número de iterações até a convergência. Além disso, a complexidade da memória é da ordem O (N ^ 2) se uma matriz de similaridade densa for usada, mas redutível se uma matriz de similaridade esparsa for usada. Isso torna a Propagação de afinidade mais apropriada para conjuntos de dados de pequeno a médio porte. 



    ## Exemplos:

    ## Demo of affinity propagation clustering algorithm: Affinity Propagation on a synthetic 2D datasets with 3 classes. (https://scikit-learn.org/stable/auto_examples/cluster/plot_affinity_propagation.html#sphx-glr-auto-examples-cluster-plot-affinity-propagation-py)

    ## Visualizing the stock market structure Affinity Propagation on Financial time series to find groups of companies (https://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html#sphx-glr-auto-examples-applications-plot-stock-market-py)






    # Descrição do algoritmo: As mensagens enviadas entre pontos pertencem a uma de duas categorias. A primeira é a responsabilidade r (i, k), que é a evidência acumulada de que a amostra k deve ser o exemplar para a amostra i. A segunda é a disponibilidade a (i, k) que é a evidência acumulada de que a amostra i deve escolher a amostra k como seu exemplar, e considera os valores para todas as outras amostras que k deve ser um exemplar. Desta forma, os exemplares são escolhidos por amostras se forem (1) semelhantes o suficiente a muitas amostras e (2) escolhidos por muitas amostras para serem representativos de si mesmos. 




    # Mais formalmente, a responsabilidade de uma amostra k de ser o exemplar da amostra i é dada por:


        # r (i, k) \ leftarrow s (i, k) - max [a (i, k ') + s (i, k') \ forall k '\ neq k]


    # Onde s (i, k) é a similaridade entre as amostras i e k. A disponibilidade da amostra k para ser o exemplar da amostra i é dada por:


        # a (i, k) \ leftarrow min [0, r (k, k) + \ sum_ {i '~ s.t. ~ i' \ notin \ {i, k \}} {r (i ', k)}]


    # Para começar, todos os valores de r e a são definidos como zero, e o cálculo de cada itera até a convergência. Conforme discutido acima, a fim de evitar oscilações numéricas ao atualizar as mensagens, o fator de amortecimento \ lambda é introduzido no processo de iteração:

        # r_ {t + 1} (i, k) = \ lambda \ cdot r_ {t} (i, k) + (1- \ lambda) \ cdot r_ {t + 1} (i, k)

        # a_ {t + 1} (i, k) = \ lambda \ cdot a_ {t} (i, k) + (1- \ lambda) \ cdot a_ {t + 1} (i, k)

    # onde t indica os tempos de iteração. 