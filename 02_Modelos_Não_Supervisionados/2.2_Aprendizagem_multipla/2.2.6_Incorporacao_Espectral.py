########## 2.2.6. Incorporação Espectral ##########

    # A incorporação espectral é uma abordagem para calcular uma incorporação não linear. Scikit-learn implementa Laplacian Eigenmaps, que encontra uma representação dimensional baixa dos dados usando uma decomposição espectral do gráfico Laplaciano. O gráfico gerado pode ser considerado como uma aproximação discreta da variedade de baixa dimensão no espaço de alta dimensão. A minimização de uma função de custo com base no gráfico garante que os pontos próximos uns dos outros na variedade sejam mapeados próximos uns dos outros no espaço de baixa dimensão, preservando as distâncias locais. A incorporação espectral pode ser realizada com a função incorporação_espectral ou sua contraparte orientada a objetos SpectralEmbedding. 



##### 2.2.6.1. Complexidade 

    # O algoritmo Spectral Embedding (Laplacian Eigenmaps) compreende três estágios:

        # Construção do gráfico ponderado. Transforme os dados de entrada brutos em representação gráfica usando a representação de matriz de afinidade (adjacência).

        # Construção Laplaciana de Gráfico. O Laplaciano não normalizado é construído como L = D - A para e normalizado como L = D ^ {- \ frac {1} {2}} (D - A) D ^ {- \ frac {1} {2}}.

        # Decomposição parcial do valor próprio. A decomposição do valor próprio é feita no gráfico Laplaciano

    # A complexidade geral da incorporação espectral é O [D \ log (k) N \ log (N)] + O [D N k ^ 3] + O [d N ^ 2].


        # N: número de pontos de dados de treinamento

        # D: dimensão de entrada

        # k: número de vizinhos mais próximos

        # d: dimensão de saída 


    ## Referências:

    ## “Laplacian Eigenmaps for Dimensionality Reduction and Data Representation” M. Belkin, P. Niyogi, Neural Computation, June 2003; 15 (6):1373-1396 (https://web.cse.ohio-state.edu/~mbelkin/papers/LEM_NC_03.pdf)