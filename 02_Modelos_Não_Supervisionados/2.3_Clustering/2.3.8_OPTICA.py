########## 2.3.8. ÓPTICA ##########

    # O algoritmo OPTICS compartilha muitas semelhanças com o algoritmo DBSCAN e pode ser considerado uma generalização do DBSCAN que relaxa o requisito de eps de um único valor para um intervalo de valores. A principal diferença entre DBSCAN e OPTICS é que o algoritmo OPTICS constrói um gráfico de alcançabilidade, que atribui a cada amostra uma distância de alcance_ e um ponto dentro do atributo ordering_ de cluster; esses dois atributos são atribuídos quando o modelo é ajustado e são usados para determinar a associação do cluster. Se OPTICS for executado com o valor padrão de inf definido para max_eps, a extração de cluster de estilo DBSCAN pode ser executada repetidamente em tempo linear para qualquer valor eps fornecido usando o método cluster_optics_dbscan. Definir max_eps com um valor inferior resultará em tempos de execução mais curtos e pode ser considerado como o raio máximo de vizinhança de cada ponto para encontrar outros pontos potenciais alcançáveis. 

        # https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html

    # As distâncias de alcançabilidade geradas pelo OPTICS permitem a extração de densidade variável de clusters em um único conjunto de dados. Conforme mostrado no gráfico acima, combinar distâncias de alcançabilidade e ordenação do conjunto de dados produz um gráfico de alcançabilidade, onde a densidade do ponto é representada no eixo Y e os pontos são ordenados de forma que os pontos próximos sejam adjacentes. ‘Cortar’ o gráfico de alcançabilidade em um único valor produz resultados semelhantes ao DBSCAN; todos os pontos acima do ‘corte’ são classificados como ruído, e cada vez que houver uma quebra durante a leitura da esquerda para a direita, significa um novo agrupamento. A extração de cluster padrão com OPTICS examina as encostas íngremes dentro do gráfico para encontrar os clusters, e o usuário pode definir o que conta como uma encosta íngreme usando o parâmetro xi. Existem também outras possibilidades de análise no próprio gráfico, como a geração de representações hierárquicas dos dados por meio de dendrogramas de plotagem de alcançabilidade, e a hierarquia de clusters detectados pelo algoritmo pode ser acessada por meio do parâmetro cluster_hierarchy_. O gráfico acima foi codificado por cores para que as cores do cluster no espaço planar correspondam aos clusters de segmento linear do gráfico de acessibilidade. Observe que os clusters azul e vermelho são adjacentes no gráfico de alcançabilidade e podem ser hierarquicamente representados como filhos de um cluster pai maior. 




    ## Exemplos:

    ## Demo of OPTICS clustering algorithm (https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html#sphx-glr-auto-examples-cluster-plot-optics-py)


    # Comparação com DBSCAN

    # Os resultados do método OPTICS cluster_optics_dbscan e DBSCAN são muito semelhantes, mas nem sempre idênticos; especificamente, rotulagem de periferia e pontos de ruído. Isso ocorre em parte porque as primeiras amostras de cada área densa processada pelo OPTICS têm um grande valor de alcançabilidade enquanto estão perto de outros pontos em sua área e, portanto, às vezes serão marcadas como ruído em vez de periferia. Isso afeta os pontos adjacentes quando são considerados candidatos a serem marcados como periferia ou ruído.

    # Nota: para qualquer valor único de eps, DBSCAN tenderá a ter um tempo de execução mais curto do que OPTICS; entretanto, para execuções repetidas em valores de eps variáveis, uma única execução do OPTICS pode exigir menos tempo de execução cumulativo do que o DBSCAN. Também é importante notar que a saída do OPTICS é próxima à do DBSCAN apenas se eps e max_eps estiverem próximos. 




    # Complexidade computacional

    # Árvores de indexação espacial são usadas para evitar o cálculo da matriz de distância total e permitir o uso eficiente da memória em grandes conjuntos de amostras. Diferentes métricas de distância podem ser fornecidas por meio da palavra-chave metric.

    # Para grandes conjuntos de dados, resultados semelhantes (mas não idênticos) podem ser obtidos via HDBSCAN. A implementação HDBSCAN é multithread e tem melhor complexidade de tempo de execução algorítmica do que OPTICS, ao custo de pior escalonamento de memória. Para conjuntos de dados extremamente grandes que esgotam a memória do sistema usando HDBSCAN, o OPTICS manterá (ao contrário) o dimensionamento da memória; entretanto, o ajuste do parâmetro max_eps provavelmente precisará ser usado para fornecer uma solução em um período de tempo razoável. 



    ## Referências:

    ## “OPTICS: ordering points to identify the clustering structure.” Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel, and Jörg Sander. In ACM Sigmod Record, vol. 28, no. 2, pp. 49-60. ACM, 1999.