########## 2.2.5. Hessian Eigenmapping ##########

    # Hessian Eigenmapping (também conhecido como LLE: HLLE baseado em Hessian) é outro método de resolver o problema de regularização de LLE. Ele gira em torno de uma forma quadrática baseada em hessian em cada vizinhança que é usada para recuperar a estrutura linear local. Embora outras implementações observem seu dimensionamento pobre com tamanho de dados, sklearn implementa algumas melhorias algorítmicas que tornam seu custo comparável ao de outras variantes LLE para pequena dimensão de saída. HLLE pode ser executado com a função localmente_linear_embedding ou sua contraparte orientada a objetos LocallyLinearEmbedding, com a palavra-chave method = 'hessian'. Requer n_neighs> n_components * (n_components + 3) / 2. 

        # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html



##### 2.2.5.1. Complexidade 

    # O algoritmo HLLE compreende três estágios:

        # Pesquisa de vizinhos mais próximos. Igual ao LLE padrão

        # Construção da matriz de peso. Aproximadamente O [D N k ^ 3] + O [N d ^ 6]. O primeiro termo reflete um custo semelhante ao do LLE padrão. O segundo termo vem de uma decomposição QR do estimador hessiano local.

        # Decomposição parcial do valor próprio. Igual ao LLE padrão

    # A complexidade geral do HLLE padrão é O [D \ log (k) N \ log (N)] + O [D N k ^ 3] + O [N d ^ 6] + O [d N ^ 2].

        # N: número de pontos de dados de treinamento

        # D: dimensão de entrada

        # k: número de vizinhos mais próximos

        # d: dimensão de saída 



    ## Referências:

    ## “Hessian Eigenmaps: Locally linear embedding techniques for high-dimensional data” Donoho, D. & Grimes, C. Proc Natl Acad Sci USA. 100:5591 (2003) (http://www.pnas.org/content/100/10/5591)

    