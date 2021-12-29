########## 2.2.3. Incorporação Localmente Linear ##########

    # A incorporação localmente linear (LLE) busca uma projeção dimensional dos dados que preserva as distâncias dentro das vizinhanças locais. Pode ser pensado como uma série de análises de componentes principais locais que são comparadas globalmente para encontrar a melhor incorporação não linear.

    # A incorporação localmente linear pode ser realizada com a função localmente_linear_embedding ou sua contraparte orientada a objetos LocallyLinearEmbedding. 

        # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html



##### 2.2.3.1. Complexidade 

    # O algoritmo LLE padrão compreende três estágios:

        # Pesquisa de vizinhos mais próximos. Veja a discussão em Isomap acima.

        # Construção da matriz de peso. O [D N k ^ 3]. A construção da matriz de peso LLE envolve a solução de uma equação linear k \ times k para cada um dos N bairros locais

        # Decomposição parcial do valor próprio. Veja a discussão no Isomap acima 

    # A complexidade geral do LLE padrão é O [D \ log (k) N \ log (N)] + O [D N k ^ 3] + O [d N ^ 2]. 



       # N: número de pontos de dados de treinamento

       # D: dimensão de entrada

       # k: número de vizinhos mais próximos

       # d: dimensão de saída 



    # Referências:

    ## “Nonlinear dimensionality reduction by locally linear embedding” Roweis, S. & Saul, L. Science 290:2323 (2000) (http://www.sciencemag.org/content/290/5500/2323.full)