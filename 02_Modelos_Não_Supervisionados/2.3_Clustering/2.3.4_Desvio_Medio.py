########## 2.3.4. Desvio Médio ##########

    # O agrupamento MeanShift visa descobrir blobs em uma densidade uniforme de amostras. É um algoritmo baseado em centróides, que funciona atualizando candidatos a centróides para serem a média dos pontos dentro de uma determinada região. Esses candidatos são filtrados em um estágio de pós-processamento para eliminar quase duplicatas e formar o conjunto final de centróides.

    # Dado um centróide candidato x_i para a iteração t, o candidato é atualizado de acordo com a seguinte equação:

        # x_i ^ {t + 1} = m (x_i ^ t)


    # Onde N (x_i) é a vizinhança das amostras dentro de uma determinada distância em torno de x_i e m é o vetor de deslocamento médio que é calculado para cada centróide que aponta para uma região de aumento máximo na densidade de pontos. Isso é calculado usando a seguinte equação, atualizando efetivamente um centróide para ser a média das amostras dentro de sua vizinhança:


        # m (x_i) = \ frac {\ sum_ {x_j \ in N (x_i)} K (x_j - x_i) x_j} {\ sum_ {x_j \ in N (x_i)} K (x_j - x_i)}


    # O algoritmo define automaticamente o número de clusters, em vez de depender de um parâmetro de largura de banda, que determina o tamanho da região a ser pesquisada. Este parâmetro pode ser definido manualmente, mas pode ser estimado usando a função estimativa_bandwidth fornecida, que é chamada se a largura de banda não for definida.

    # O algoritmo não é altamente escalável, pois requer várias pesquisas pelos vizinhos mais próximos durante a execução do algoritmo. O algoritmo tem garantia de convergência, no entanto, o algoritmo irá parar de iterar quando a mudança nos centróides for pequena.

    # A rotulagem de uma nova amostra é realizada encontrando o centroide mais próximo para uma determinada amostra. 

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html


    ## Exemplos:

    ## A demo of the mean-shift clustering algorithm: Mean Shift clustering on a synthetic 2D datasets with 3 classes. (https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py)




    ## Referências:

    ## “Mean shift: A robust approach toward feature space analysis.” D. Comaniciu and P. Meer, IEEE Transactions on Pattern Analysis and Machine Intelligence (2002) (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.8968&rep=rep1&type=pdf)