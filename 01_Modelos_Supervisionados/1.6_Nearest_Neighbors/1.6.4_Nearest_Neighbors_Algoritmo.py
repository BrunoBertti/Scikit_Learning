########## 1.6.4 Algoritmos do Vizinho Mais Próximo  ##########



##### 1.6.4.1 Força Bruta

    # O cálculo rápido dos vizinhos mais próximos é uma área ativa de pesquisa em aprendizado de máquina. A implementação de pesquisa de vizinho mais ingênua envolve o cálculo de força bruta de distâncias entre todos os pares de pontos no conjunto de dados: para N amostras em dimensões D, essa abordagem é escalonada como O [D N ^ 2]. Pesquisas de vizinhos de força bruta eficientes podem ser muito competitivas para pequenas amostras de dados. No entanto, conforme o número de amostras aumenta, a abordagem da força bruta rapidamente se torna inviável. Nas classes de sklearn.neighbors, as pesquisas de vizinhos de força bruta são especificadas usando a palavra-chave algoritmo = 'brute' e são calculadas usando as rotinas disponíveis em sklearn.metrics.pairwise. 
     
    


##### 1.6.4.2 K-D Tree

    # Para lidar com as ineficiências computacionais da abordagem de força bruta, uma variedade de estruturas de dados baseadas em árvore foram inventadas. Em geral, essas estruturas tentam reduzir o número necessário de cálculos de distância, codificando de forma eficiente as informações de distância agregadas para a amostra. A ideia básica é que se o ponto a está muito distante do ponto B, e o ponto B está muito próximo do ponto C, então sabemos que os pontos A e C estão muito distantes, sem ter que calcular explicitamente sua distância. Desta forma, o custo computacional de uma pesquisa de vizinhos mais próximos pode ser reduzido para O [D N \ log (N)] ou melhor. Esta é uma melhoria significativa em relação à força bruta para N. 

    # Uma abordagem inicial para tirar proveito dessas informações agregadas foi a estrutura de dados da árvore KD (abreviação de árvore K-dimensional), que generaliza árvores quádruplas bidimensionais e árvores octológicas tridimensionais para um número arbitrário de dimensões. A árvore KD é uma estrutura de árvore binária que particiona recursivamente o espaço de parâmetros ao longo dos eixos de dados, dividindo-o em regiões ortotrópicas aninhadas nas quais os pontos de dados são arquivados. A construção de uma árvore KD é muito rápida: como o particionamento é executado apenas ao longo dos eixos de dados, nenhuma distância D-dimensional precisa ser calculada. Uma vez construído, o vizinho mais próximo de um ponto de consulta pode ser determinado apenas com cálculos de distância O [\ log (N)]. Embora a abordagem da árvore KD seja muito rápida para pesquisas de vizinhos de baixa dimensão (D <20), ela se torna ineficiente à medida que D fica muito grande: esta é uma manifestação da chamada “maldição da dimensionalidade”. No scikit-learn, as pesquisas de vizinhos de árvore KD são especificadas usando o algoritmo de palavra-chave = 'kd_tree' e são calculadas usando a classe KDTree. 

    ## Referências:

    ## “Multidimensional binary search trees used for associative searching”, Bentley, J.L., Communications of the ACM (1975) (https://dl.acm.org/citation.cfm?doid=361002.361007)



##### 1.6.4.3 Ball Tree 

    # Para abordar as ineficiências das Árvores KD em dimensões superiores, a estrutura de dados da árvore da bola foi desenvolvida. Enquanto as árvores KD particionam os dados ao longo dos eixos cartesianos, as árvores das bolas particionam os dados em uma série de hiperesferas aninhadas. Isso torna a construção de árvores mais cara do que a árvore KD, mas resulta em uma estrutura de dados que pode ser muito eficiente em dados altamente estruturados, mesmo em dimensões muito altas.

    # Uma árvore de bola divide recursivamente os dados em nós definidos por um centróide C e raio r, de modo que cada ponto no nó esteja dentro da hiperesfera definida por r e C. O número de pontos candidatos para uma pesquisa de vizinho é reduzido com o uso da desigualdade do triângulo:

        # | x + y | \ leq | x | + | y ​​|

    # Com esta configuração, um único cálculo de distância entre um ponto de teste e o centróide é suficiente para determinar um limite inferior e superior na distância de todos os pontos dentro do nó. Por causa da geometria esférica dos nós da árvore de bolas, ele pode superar uma árvore KD em dimensões altas, embora o desempenho real seja altamente dependente da estrutura dos dados de treinamento. No scikit-learn, as pesquisas de vizinhos baseadas em ball-tree são especificadas usando o algoritmo de palavra-chave = 'ball_tree' e são calculadas usando a classe BallTree. Como alternativa, o usuário pode trabalhar com a classe BallTree diretamente. 


    ## Referências:

    ## “Five balltree construction algorithms”, Omohundro, S.M., International Computer Science Institute Technical Report (1989) (http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.91.8209)



##### 1.6.4.4 Algoritmo de escolha de vizinhos mais próximos

    # O algoritmo ideal para um determinado conjunto de dados é uma escolha complicada e depende de uma série de fatores: 

        # número de amostras N (ou seja, n_amostras) e dimensionalidade D (ou seja, n_variáveis).
    
            # O tempo de consulta de força bruta aumenta conforme O [D N]

            # O tempo de consulta da árvore de bolas aumenta aproximadamente O [D \ log (N)]

            # O tempo de consulta da árvore KD muda com D de uma maneira que é difícil de caracterizar com precisão. Para D pequeno (menos de 20 ou mais), o custo é de aproximadamente O [D \ log (N)], e a consulta da árvore KD pode ser muito eficiente. Para D maior, o custo aumenta para quase O [DN], e a sobrecarga devido à estrutura da árvore pode levar a consultas que são mais lentas do que a força bruta.  
            # Para pequenos conjuntos de dados (N menor que 30 ou mais), \ log (N) é comparável a N, e algoritmos de força bruta podem ser mais eficientes do que uma abordagem baseada em árvore. Tanto o KDTree quanto o BallTree tratam disso fornecendo um parâmetro de tamanho de folha: isso controla o número de amostras nas quais uma consulta muda para força bruta. Isso permite que ambos os algoritmos se aproximem da eficiência de um cálculo de força bruta para N. 
        
        # estrutura de dados: dimensionalidade intrínseca dos dados e / ou dispersão dos dados. A dimensionalidade intrínseca se refere à dimensão d <= D de uma variedade na qual os dados se encontram, que pode ser linearmente ou não linearmente embutida no espaço de parâmetros. A dispersão se refere ao grau em que os dados preenchem o espaço do parâmetro (isso deve ser diferenciado do conceito usado em matrizes "esparsas". A matriz de dados pode não ter entradas zero, mas a estrutura ainda pode ser "esparsa" neste senso). 

            # O tempo de consulta de força bruta não é alterado pela estrutura de dados. 

            # Os tempos de consulta da árvore de bolas e da árvore KD podem ser muito influenciados pela estrutura de dados. Em geral, dados mais esparsos com uma dimensionalidade intrínseca menor levam a tempos de consulta mais rápidos. Como a representação interna da árvore KD está alinhada com os eixos dos parâmetros, ela geralmente não mostra tantas melhorias quanto a árvore de bolas para dados estruturados arbitrariamente.
            # Os conjuntos de dados usados no aprendizado de máquina tendem a ser muito estruturados e são muito adequados para consultas baseadas em árvore. 

        # número de vizinhos k solicitados para um ponto de consulta. 

            # O tempo de consulta de força bruta não é afetado pelo valor de k

            # O tempo de consulta da árvore de bolas e da árvore KD se tornará mais lento à medida que k aumenta. Isso se deve a dois efeitos: primeiro, um k maior leva à necessidade de pesquisar uma porção maior do espaço de parâmetros. Em segundo lugar, usar k> 1 requer enfileiramento interno de resultados conforme a árvore é percorrida.
            # À medida que k se torna grande em comparação com N, a capacidade de podar ramos em uma consulta baseada em árvore é reduzida. Nessa situação, as consultas de força bruta podem ser mais eficientes. 

        # número de pontos de consulta. Tanto a árvore de bolas quanto a árvore KD requerem uma fase de construção. O custo desta construção torna-se insignificante quando amortizado em muitas consultas. Se apenas um pequeno número de consultas for executado, no entanto, a construção pode representar uma fração significativa do custo total. Se poucos pontos de consulta forem necessários, a força bruta é melhor do que um método baseado em árvore. 

    # Atualmente, o algoritmo = 'auto' seleciona 'bruto' se qualquer uma das seguintes condições for verificada: 

        # os dados de entrada são esparsos

        # metric = 'pré-computado'

        # D > 15

        # k> = N / 2

        # Effective_metric_ não está na lista VALID_METRICS para 'kd_tree' ou 'ball_tree' 


    # Caso contrário, ele seleciona o primeiro de 'kd_tree' e 'ball_tree' que tem effective_metric_ em sua lista VALID_METRICS. Esta heurística é baseada nas seguintes suposições: 

        # o número de pontos de consulta é pelo menos a mesma ordem que o número de pontos de treinamento

        # leaf_size está próximo de seu valor padrão de 30

        # quando D> 15, a dimensionalidade intrínseca dos dados é geralmente muito alta para métodos baseados em árvore 

##### 1.6.4.5 Efeito do leaf_size

    # Conforme observado acima, para tamanhos de amostra pequenos, uma pesquisa de força bruta pode ser mais eficiente do que uma consulta baseada em árvore. Esse fato é explicado na árvore de bolas e na árvore KD, alternando internamente para pesquisas de força bruta nos nós de folha. O nível dessa opção pode ser especificado com o parâmetro leaf_size. Esta escolha de parâmetro tem muitos efeitos: 

    # hora de construção

        # Um leaf_size maior leva a um tempo de construção de árvore mais rápido, porque menos nós precisam ser criados

    # tempo de consulta

        # Um tamanho de folha grande ou pequeno pode levar a um custo de consulta abaixo do ideal. Para leaf_size se aproximando de 1, a sobrecarga envolvida na passagem de nós pode reduzir significativamente os tempos de consulta. Para leaf_size se aproximando do tamanho do conjunto de treinamento, as consultas tornam-se essencialmente de força bruta. Um bom meio-termo entre eles é leaf_size = 30, o valor padrão do parâmetro.


    # memória

        # À medida que o tamanho da folha aumenta, a memória necessária para armazenar uma estrutura em árvore diminui. Isso é especialmente importante no caso da árvore de bolas, que armazena um centróide dimensional para cada nó. O espaço de armazenamento necessário para BallTree é aproximadamente 1 / leaf_size vezes o tamanho do conjunto de treinamento.

    # leaf_size não é referenciado para consultas de força bruta. 



##### 1.6.4.6 Métricas válidas para algoritmos do vizinho mais próximo 

    # Para obter uma lista de métricas disponíveis, consulte a documentação da classe DistanceMetric.


    # Uma lista de métricas válidas para qualquer um dos algoritmos acima pode ser obtida usando seu atributo valid_metric. Por exemplo, métricas válidas para KDTree podem ser geradas por: 

from sklearn.neighbors import KDTree
print(sorted(KDTree.valid_metrics))