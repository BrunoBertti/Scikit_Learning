########## 2.2.7. Alinhamento de espaço tangente local ##########

    # Embora não seja tecnicamente uma variante do LLE, o alinhamento do espaço tangente local (LTSA) é algoritmicamente semelhante ao LLE para poder ser colocado nesta categoria. Em vez de focar na preservação de distâncias de vizinhança como no LLE, o LTSA busca caracterizar a geometria local em cada vizinhança por meio de seu espaço tangente e realiza uma otimização global para alinhar esses espaços tangentes locais para aprender a incorporação. LTSA pode ser executado com a função localmente_linear_embedding ou sua contraparte orientada a objetos LocallyLinearEmbedding, com a palavra-chave method = 'ltsa'. 

        # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html


##### 2.2.7.1. Complexidade 



    # O algoritmo LTSA compreende três estágios:

        # Pesquisa de vizinhos mais próximos. Igual ao LLE padrão

        # Construção da matriz de peso. Aproximadamente O [D N k ^ 3] + O [k ^ 2 d]. O primeiro termo reflete um custo semelhante ao do LLE padrão.

        # Decomposição parcial do valor próprio. Igual ao LLE padrão

    # A complexidade geral do LTSA padrão é O [D \ log (k) N \ log (N)] + O [D N k ^ 3] + O [k ^ 2 d] + O [d N ^ 2].

        # N: número de pontos de dados de treinamento

        # D: dimensão de entrada

        # k: número de vizinhos mais próximos

        # d: dimensão de saída 

    
    ## Referências:

    ## “Principal manifolds and nonlinear dimensionality reduction via tangent space alignment” Zhang, Z. & Zha, H. Journal of Shanghai Univ. 8:406 (2004) (http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.4.3693)

    