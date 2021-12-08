########## 1.1.16 Regressão de Robustez - Outliers e Erros de Modelagem ##########

    # Regressão de Robustez mira treinar um modelo de regressão na presença de dados corrompidos: ou outliers, ou erros de modelagem.


##### 1.1.16.1 Cenário diferente e conceitos úteis 

    # Há diferentes coisas para manter na cabeça quando trabalhamos com dados corrompidos por outliers:

        # Outliers no X ou no y?
            # Y: dados acima na forma horizontal.

            # x: dados com um minimax extenso e com dados na vertical.

            # https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html

        # Fração de outliers versus amplitude de erro 

            # O número de pontos externos é importante, mas também o quanto eles são discrepantes. 

            # https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html

    # Uma noção importante de ajuste robusto é a de ponto de decomposição: a fração de dados que pode estar ausente para o ajuste começar a perder os dados básicos.

    # Observe que, em geral, o ajuste robusto em configurações de alta dimensão (n_features grandes) é muito difícil. Os modelos robustos aqui provavelmente não funcionarão nessas configurações. 


    ## Trade-offs: Qual estimador?
    ## O Scikit-learn fornece 3 estimadores de regressão robustos: RANSAC, Theil Sen e HuberRegressor. :

        ## HuberRegressor deve ser mais rápido do que RANSAC e Theil Sen, a menos que o número de amostras seja muito grande, ou seja, n_samples >> n_features. Isso ocorre porque RANSAC e Theil Sen se encaixam em subconjuntos menores de dados. No entanto, é improvável que Theil Sen e RANSAC sejam tão robustos quanto HuberRegressor para os parâmetros padrão.

        ## O RANSAC é mais rápido do que Theil Sen e tem uma escala muito melhor com o número de amostras.

        ## O RANSAC lidará melhor com grandes valores discrepantes na direção y (situação mais comum).

        ## Theil Sen lidará melhor com outliers de tamanho médio na direção X, mas essa propriedade desaparecerá em configurações de dimensões altas.

        ## Em caso de dúvida, use RANSAC. 



##### 1.1.16.2 RANSAC: RANdom SAmple Consensus - Consenso de Amostra Aleatória 

    # RANSAC (RANdom SAmple Consensus) ajusta um modelo de subconjuntos aleatórios de inliers do conjunto de dados completo.

    # RANSAC é um algoritmo não determinístico que produz apenas um resultado razoável com uma certa probabilidade, que é dependente do número de iterações (veja o parâmetro max_trials). É normalmente usado para problemas de regressão linear e não linear e é especialmente popular no campo da visão computacional fotogramétrica.

    # O algoritmo divide os dados completos da amostra de entrada em um conjunto de inliers, que pode estar sujeito a ruído e outliers, que são, por exemplo, causados por medições erradas ou hipóteses inválidas sobre os dados. O modelo resultante é então estimado apenas a partir dos inliers determinados. 



##### 1.1.16.2.1 Detalhes do algorítmo

    # Cada ação na iteração tem os seguintes passos:

        # 1 - Selecione min_samples amostras aleatórias dos dados originais e verifique se o conjunto de dados é válido (consulte is_data_valid).

        # 2 - Ajuste um modelo ao subconjunto aleatório (base_estimator.fit) e verifique se o modelo estimado é válido (consulte is_model_valid).
         
        # 3 - Classifique todos os dados como inliers ou outliers calculando os resíduos para o modelo estimado (base_estimator.predict (X) - y) - todas as amostras de dados com resíduos absolutos menores ou iguais ao residual_threshold são considerados inliers.
         
        # 4 - Salve o modelo ajustado como o melhor modelo se o número de amostras inlier for máximo. Caso o modelo atual estimado tenha o mesmo número de inliers, ele só é considerado o melhor modelo se tiver melhor pontuação. 

    # Essas etapas são executadas um número máximo de vezes (max_trials) ou até que um dos critérios de parada especiais seja atendido (consulte stop_n_inliers e stop_score). O modelo final é estimado usando todas as amostras inlier (conjunto de consenso) do melhor modelo previamente determinado. 

    # As funções is_data_valid e is_model_valid permitem identificar e rejeitar combinações degeneradas de subamostras aleatórias. Se o modelo estimado não for necessário para identificar casos degenerados, is_data_valid deve ser usado como é chamado antes de ajustar o modelo e, portanto, levando a um melhor desempenho computacional. 


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html#sphx-glr-auto-examples-linear-model-plot-ransac-py

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html#sphx-glr-auto-examples-linear-model-plot-robust-fit-py



    ## Referências:

    ## https://en.wikipedia.org/wiki/RANSAC

    ## “Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography” Martin A. Fischler and Robert C. Bolles - SRI International (1981) https://www.sri.com/sites/default/files/publications/ransac-publication.pdf

    ## “Performance Evaluation of RANSAC Family” Sunglok Choi, Taemin Kim and Wonpil Yu - BMVC (2009) (http://www.bmva.org/bmvc/2009/Papers/Paper355/Paper355.pdf)


##### 1.1.16.3 Estimador de Theil-Sen: estimador baseado em mediana generalizada 

    # O estimador TheilSenRegressor usa uma generalização da mediana em múltiplas dimensões. É, portanto, robusto para outliers multivariados. Observe, entretanto, que a robustez do estimador diminui rapidamente com a dimensionalidade do problema. Ele perde suas propriedades de robustez e não se torna melhor do que um mínimo de quadrados comuns em alta dimensão. 

    ## Exemplos:

    ##https://scikit-learn.org/stable/auto_examples/linear_model/plot_theilsen.html#sphx-glr-auto-examples-linear-model-plot-theilsen-py

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html#sphx-glr-auto-examples-linear-model-plot-robust-fit-py


    ## Referências:

    ## https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator


##### 1.1.16.3.1 Considerações Teóricas

    # TheilSenRegressor é comparável ao Ordinary Least Squares (OLS) em termos de eficiência assintótica e como um estimador não enviesado. Em contraste com OLS, Theil-Sen é um método não paramétrico, o que significa que não faz suposições sobre a distribuição subjacente dos dados. Como Theil-Sen é um estimador baseado na mediana, ele é mais robusto contra dados corrompidos, também conhecidos como outliers. Na configuração univariada, Theil-Sen tem um ponto de decomposição de cerca de 29,3% no caso de uma regressão linear simples, o que significa que pode tolerar dados corrompidos arbitrariamente de até 29,3%.

    # A implementação de TheilSenRegressor no scikit-learn segue uma generalização para um modelo de regressão linear multivariado 12 usando a mediana espacial que é uma generalização da mediana para dimensões múltiplas 13.

    # Em termos de complexidade de tempo e espaço, Theil-Sen dimensiona de acordo com n_amostras e n_subamostra

    # o que o torna inviável para ser aplicado exaustivamente a problemas com um grande número de amostras e recursos. Portanto, a magnitude de uma subpopulação pode ser escolhida para limitar a complexidade de tempo e espaço, considerando apenas um subconjunto aleatório de todas as combinações possíveis. 

    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_theilsen.html#sphx-glr-auto-examples-linear-model-plot-theilsen-py

    ## Referências:

    ## Xin Dang, Hanxiang Peng, Xueqin Wang and Heping Zhang: Theil-Sen Estimators in a Multiple Linear Regression Model. (http://home.olemiss.edu/~xdang/papers/MTSE.pdf)

    ## Kärkkäinen and S. Äyrämö: On Computation of Spatial Median for Robust Data Mining. (http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf)


##### 1.1.16.4 Regressão Huber

    # O HuberRegressor é diferente de Ridge porque aplica uma perda linear a amostras classificadas como outliers. Uma amostra é classificada como inlier se o erro absoluto dessa amostra for menor que um certo limite. Ele difere de TheilSenRegressor e RANSACRegressor porque não ignora o efeito dos outliers, mas atribui um peso menor a eles. 

    # The loss function that HuberRegressor minimizes is given by

        # min Somatório (sigma + H (Xiw - ui / sigma)sigma) + alpha||w|| 2^2

    # onde

        # H(z) = z^2, se |z| < e,
        #        2e|z| - e^2, senão

    
    # É aconselhável definir o parâmetro epsilon para 1,35 para atingir 95% de eficiência estatística. 

##### 1.1.16.5 Notas

    # O HuberRegressor difere de usar SGDRegressor com perda definida para huber das seguintes maneiras.

        # HuberRegressor é invariante de escala. Depois que o épsilon é definido, dimensionar X e y para baixo ou para cima em valores diferentes produziria a mesma robustez para outliers de antes. em comparação com SGDRegressor, onde épsilon deve ser definido novamente quando X e y são escalados.

        # O HuberRegressor deve ser mais eficiente para usar em dados com pequeno número de amostras, enquanto o SGDRegressor precisa de várias passagens nos dados de treinamento para produzir a mesma robustez. 

    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_huber_vs_ridge.html#sphx-glr-auto-examples-linear-model-plot-huber-vs-ridge-py

    ## Referências:

    ## Peter J. Huber, Elvezio M. Ronchetti: Robust Statistics, Concomitant scale estimates, pg 172


    # Observe que este estimador é diferente da implementação de R de regressão robusta (http://www.ats.ucla.edu/stat/r/dae/rreg.htm) porque a implementação de R faz uma implementação de mínimos quadrados ponderados com pesos dados a cada amostra com base em quanto o residual é maior do que um certo limite. 