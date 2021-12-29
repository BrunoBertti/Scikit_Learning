########## 2.2.2. Isomap ##########

    # Uma das primeiras abordagens para aprendizado múltiplo é o algoritmo Isomap, abreviação de Mapeamento Isométrico. O Isomap pode ser visto como uma extensão do Multi-dimensional Scaling (MDS) ou Kernel PCA. O Isomap busca uma incorporação de dimensão inferior que mantém as distâncias geodésicas entre todos os pontos. O Isomap pode ser executado com o objeto Isomap. 

        # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html



##### 2.2.2.1. Complexidade 

    # O algoritmo Isomap compreende três estágios: 

        # Pesquisa de vizinho mais próximo. Isomap usa BallTree para busca eficiente de vizinhos. O custo é de aproximadamente O [D \ log (k) N \ log (N)], para k vizinhos mais próximos de N pontos em D dimensões.


        # Pesquisa de gráfico de caminho mais curto. Os algoritmos conhecidos mais eficientes para isso são o algoritmo de Dijkstra, que é aproximadamente O [N ^ 2 (k + \ log (N))], ou o algoritmo Floyd-Warshall, que é O [N ^ 3]. O algoritmo pode ser selecionado pelo usuário com a palavra-chave path_method do Isomap. Se não for especificado, o código tenta escolher o melhor algoritmo para os dados de entrada.


        # Decomposição parcial de autovalores. A incorporação é codificada nos autovetores correspondentes aos d maiores autovalores do kernel N \ times N isomapa. Para um solucionador denso, o custo é de aproximadamente O [d N ^ 2]. Muitas vezes, esse custo pode ser melhorado com o solucionador ARPACK. O eigensolver pode ser especificado pelo usuário com a palavra-chave eigen_solver do Isomap. Se não for especificado, o código tenta escolher o melhor algoritmo para os dados de entrada. 


    # A complexidade geral do Isomap é O [D \ log (k) N \ log (N)] + O [N ^ 2 (k + \ log (N))] + O [d N ^ 2]


        # N : número de pontos de dados de treinamento

        # D : dimensão de entrada

        # k : número de vizinhos mais próximos

        # d : dimensão de saída


    ## Referências:

    ## “A global geometric framework for nonlinear dimensionality reduction” Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. Science 290 (5500) (http://science.sciencemag.org/content/290/5500/2319.full)

