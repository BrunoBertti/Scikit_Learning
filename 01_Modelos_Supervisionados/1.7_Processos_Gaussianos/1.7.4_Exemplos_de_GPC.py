########## 1.7.4. Exemplos de GPC ##########



##### 1.7.4.1. Previsões probabilísticas com GPC

    # Este exemplo ilustra a probabilidade prevista de GPC para um kernel RBF com diferentes opções de hiperparâmetros. A primeira figura mostra a probabilidade prevista de GPC com hiperparâmetros escolhidos arbitrariamente e com os hiperparâmetros correspondentes à probabilidade log-marginal máxima (LML).

    # Embora os hiperparâmetros escolhidos pela otimização do LML tenham um LML consideravelmente maior, eles têm um desempenho um pouco pior de acordo com a perda de log nos dados de teste. A figura mostra que isso ocorre porque eles exibem uma mudança acentuada das probabilidades de classe nos limites da classe (o que é bom), mas têm probabilidades previstas próximas a 0,5 longe dos limites da classe (o que é ruim). Aproximação de Laplace usada internamente pelo GPC.

    # A segunda figura mostra a probabilidade log-marginal para diferentes escolhas dos hiperparâmetros do kernel, destacando as duas escolhas dos hiperparâmetros usados na primeira figura por pontos pretos. 


        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc.html





##### 1.7.4.2. Ilustração do GPC no conjunto de dados XOR



    # Este exemplo ilustra o GPC em dados XOR. Comparados são um kernel estacionário, isotrópico (RBF) e um kernel não estacionário (DotProduct). Neste conjunto de dados em particular, o kernel DotProduct obtém resultados consideravelmente melhores porque os limites de classe são lineares e coincidem com os eixos coordenados. Na prática, no entanto, kernels estacionários como o RBF geralmente obtêm melhores resultados. 

        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_xor.html
    






##### 1.7.4.3. Classificação de processo gaussiana (GPC) no conjunto de dados de íris 


    # Este exemplo ilustra a probabilidade prevista de GPC para um kernel RBF isotrópico e anisotrópico em uma versão bidimensional para o conjunto de dados iris. Isso ilustra a aplicabilidade do GPC à classificação não binária. O kernel RBF anisotrópico obtém uma probabilidade log-marginal ligeiramente mais alta, atribuindo diferentes escalas de comprimento às duas dimensões de recursos. 

        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_iris.html