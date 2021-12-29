########## 2.2.4. Incorporação Linear Localmente Modificada ##########

    # Um problema bem conhecido com o LLE é o problema de regularização. Quando o número de vizinhos é maior do que o número de dimensões de entrada, a matriz que define cada vizinhança local é deficiente em classificação. Para resolver isso, o LLE padrão aplica um parâmetro de regularização arbitrário r, que é escolhido em relação ao traço da matriz de peso local. Embora possa ser mostrado formalmente que como r \ para 0, a solução converge para o embedding desejado, não há garantia de que a solução ótima será encontrada para r> 0. Este problema se manifesta em embeddings que distorcem a geometria subjacente do múltiplo.

    # Um método para resolver o problema de regularização é usar vários vetores de peso em cada bairro. Esta é a essência do embedding localmente linear modificado (MLLE). MLLE pode ser executado com a função localmente_linear_embedding ou sua contraparte orientada a objetos LocallyLinearEmbedding, com a palavra-chave method = 'modificado'. Requer n_neighs> n_components.
     
        # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html



##### 2.2.4.1. Complexidade 

    # O algoritmo MLLE compreende três estágios:

       # Pesquisa de vizinhos mais próximos. Igual ao LLE padrão

       # Construção da matriz de peso. Aproximadamente O [D N k ^ 3] + O [N (k-D) k ^ 2]. O primeiro termo é exatamente equivalente ao do LLE padrão. O segundo termo tem a ver com a construção da matriz de pesos a partir de vários pesos. Na prática, o custo adicional de construção da matriz de peso MLLE é relativamente pequeno em comparação com o custo dos estágios 1 e 3.

       # Decomposição parcial do valor próprio. Igual ao LLE padrão 


    # A complexidade geral de MLLE é O [D \ log (k) N \ log (N)] + O [D N k ^ 3] + O [N (k-D) k ^ 2] + O [d N ^ 2].

        # D: número de pontos de dados de treinamento

        # K: dimensão de entrada

        # n: número de vizinhos mais próximos

        # d: dimensão de saída 

    

    ## Referências:

    ## “MLLE: Modified Locally Linear Embedding Using Multiple Weights” Zhang, Z. & Wang, J. (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382)

    