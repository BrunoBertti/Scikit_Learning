########## 2.2.9. Incorporação de Vizinho Estocástico com distribuição t (t-SNE) ##########

    # t-SNE (TSNE) converte afinidades de pontos de dados em probabilidades. As afinidades no espaço original são representadas por probabilidades conjuntas gaussianas e as afinidades no espaço incorporado são representadas por distribuições t de Student. Isso permite que o t-SNE seja particularmente sensível à estrutura local e tem algumas outras vantagens sobre as técnicas existentes:

        # Revelando a estrutura em várias escalas em um único mapa

        # Revelando dados que se encontram em múltiplos, diferentes, múltiplos ou clusters

        # Reduzindo a tendência de aglomerar os pontos no centro


    # Enquanto Isomap, LLE e variantes são mais adequados para desdobrar uma única variedade contínua de baixa dimensão, t-SNE se concentrará na estrutura local dos dados e tenderá a extrair grupos locais agrupados de amostras, conforme destacado no exemplo da curva S. Essa capacidade de agrupar amostras com base na estrutura local pode ser benéfica para desemaranhar visualmente um conjunto de dados que compreende vários manifolds de uma vez, como é o caso no conjunto de dados de dígitos.

    # A divergência de Kullback-Leibler (KL) das probabilidades conjuntas no espaço original e no espaço embutido será minimizada pela descida do gradiente. Observe que a divergência KL não é convexa, ou seja, reinícios múltiplos com inicializações diferentes terminarão em mínimos locais da divergência KL. Portanto, às vezes é útil tentar sementes diferentes e selecionar a incorporação com a divergência de KL mais baixa.

    # As desvantagens de usar t-SNE são aproximadamente: 


        # t-SNE é computacionalmente caro e pode levar várias horas em conjuntos de dados de milhões de amostras em que o PCA terminará em segundos ou minutos

        # O método Barnes-Hut t-SNE é limitado a embeddings bidimensionais ou tridimensionais.

        # O algoritmo é estocástico e várias reinicializações com diferentes sementes podem resultar em diferentes embeddings. No entanto, é perfeitamente legítimo escolher a incorporação com o mínimo de erro.

        # A estrutura global não é preservada explicitamente. Este problema é mitigado inicializando pontos com PCA (usando init = 'pca'). 

            # https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html


##### 2.2.9.1. Otimizando t-SNE 

    # O principal objetivo do t-SNE é a visualização de dados de alta dimensão. Portanto, funciona melhor quando os dados são incorporados em duas ou três dimensões.

    # Otimizar a divergência KL pode ser um pouco complicado às vezes. Existem cinco parâmetros que controlam a otimização de t-SNE e, portanto, possivelmente a qualidade da incorporação resultante: 

        # perplexidade

        # fator de exagero precoce

        # taxa de Aprendizagem

        # número máximo de iterações

        # ângulo (não usado no método exato) 

    # A perplexidade é definida como k = 2 ^ {(S)} onde S é a entropia de Shannon da distribuição de probabilidade condicional. A perplexidade de um dado com k é k, de modo que k é efetivamente o número de vizinhos mais próximos que t-SNE considera ao gerar as probabilidades condicionais. Perplexidades maiores levam a vizinhos mais próximos e menos sensíveis a estruturas pequenas. Por outro lado, uma perplexidade menor considera um número menor de vizinhos e, portanto, ignora mais informações globais em favor da vizinhança local. À medida que os tamanhos dos conjuntos de dados ficam maiores, mais pontos serão necessários para obter uma amostra razoável da vizinhança local e, portanto, maiores perplexidades podem ser necessárias. Da mesma forma, conjuntos de dados mais ruidosos exigirão valores de perplexidade maiores para abranger vizinhos locais suficientes para ver além do ruído de fundo.


    # O número máximo de iterações geralmente é alto o suficiente e não precisa de nenhum ajuste. A otimização consiste em duas fases: a fase de exagero inicial e a otimização final. Durante o exagero inicial, as probabilidades conjuntas no espaço original serão aumentadas artificialmente pela multiplicação com um determinado fator. Fatores maiores resultam em lacunas maiores entre os clusters naturais nos dados. Se o fator for muito alto, a divergência KL pode aumentar durante esta fase. Normalmente não precisa ser ajustado. Um parâmetro crítico é a taxa de aprendizagem. Se o gradiente for muito baixo, a descida ficará presa em um mínimo local incorreto. Se for muito alto, a divergência KL aumentará durante a otimização. Uma heurística sugerida em Belkina et al. (2019) é definir a taxa de aprendizado para o tamanho da amostra dividido pelo fator de exagero inicial. Implementamos esta heurística como o argumento learning_rate = 'auto'. Mais dicas podem ser encontradas nas Perguntas frequentes de Laurens van der Maaten (ver referências). O último parâmetro, ângulo, é uma compensação entre desempenho e precisão. Ângulos maiores implicam que podemos aproximar regiões maiores por um único ponto, levando a melhor velocidade, mas resultados menos precisos.

    # “Como usar t-SNE com eficácia” fornece uma boa discussão sobre os efeitos dos vários parâmetros, bem como gráficos interativos para explorar os efeitos de diferentes parâmetros. 

##### 2.2.9.2. Barnes-Hut t-SNE 

    # O Barnes-Hut t-SNE que foi implementado aqui é geralmente muito mais lento do que outros algoritmos de aprendizagem múltiplos. A otimização é bastante difícil e o cálculo do gradiente é O [d N log (N)], onde d é o número de dimensões de saída e N é o número de amostras. O método Barnes-Hut melhora o método exato em que a complexidade t-SNE é O [d N ^ 2], mas tem várias outras diferenças notáveis: 


        # A implementação Barnes-Hut só funciona a dimensionalidade alvo é 3 ou menos. O caso 2D é típico na construção de visualizações.

        # Barnes-Hut só funciona com dados de entrada densos. Matrizes de dados esparsos só podem ser incorporadas com o método exato ou podem ser aproximadas por uma projeção densa de baixa classificação, por exemplo, usando TruncatedSVD

        # Barnes-Hut é uma escolha do método exato. A aproximação é parametrizada com o parâmetro de ângulo, portanto, o parâmetro de ângulo não é usado quando método = ”exato”

        # Barnes-Hut é histórico mais escalonável. Barnes-Hut pode ser usado para incorporar centenas de pontos de pontos de dados, enquanto o método exato pode lidar com os de antes de se tornar intratável computacionalmente 

    # Para fins de visualização (que é o principal caso de uso de t-SNE), o uso do método de Barnes-Hut é altamente recomendado. O método t-SNE exato é útil para verificar as propriedades teoricamente da incorporação, possivelmente em um espaço dimensional superior, mas limita a pequenos conjuntos de dados devido a restrições computacionais.

    # Observe também que os rótulos dos dígitos correspondem aproximadamente ao agrupamento natural encontrado por t-SNE, enquanto a projeção 2D linear do modelo PCA produz uma representação em que as regiões dos rótulos se sobrepõem amplamente. Esta é uma forte pista de que esses dados podem ser bem separados por métodos não lineares que se concentram na estrutura local (por exemplo, um SVM com um kernel RBF gaussiano). No entanto, a falha em visualizar grupos bem separados rotulados homogeneamente com t-SNE em 2D não implica necessariamente que os dados não possam ser classificados corretamente por um modelo supervisionado. Pode ser que 2 dimensões não sejam altas o suficiente para representar com precisão a estrutura interna dos dados.


    ## Referências:

    ## “Visualizing High-Dimensional Data Using t-SNE” van der Maaten, L.J.P.; Hinton, G. Journal of Machine Learning Research (2008) (http://jmlr.org/papers/v9/vandermaaten08a.html)

    ## “t-Distributed Stochastic Neighbor Embedding” van der Maaten, L.J.P. (https://lvdmaaten.github.io/tsne/)

    ## “Accelerating t-SNE using Tree-Based Algorithms” van der Maaten, L.J.P.; Journal of Machine Learning Research 15(Oct):3221-3245, 2014. (https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf)

    ## “Automated optimized parameters for T-distributed stochastic neighbor embedding improve visualization and analysis of large datasets” Belkina, A.C., Ciccolella, C.O., Anno, R., Halpert, R., Spidlen, J., Snyder-Cappione, J.E., Nature Communications 10, 5415 (2019). (https://www.nature.com/articles/s41467-019-13055-y)