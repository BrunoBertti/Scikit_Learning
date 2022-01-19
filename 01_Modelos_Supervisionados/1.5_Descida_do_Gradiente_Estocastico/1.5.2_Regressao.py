########## 1.5.2. Regressão ##########

    # A classe SGDRegressor implementa uma rotina de aprendizado de descida de gradiente estocástica simples que suporta diferentes funções de perda e penalidades para ajustar modelos de regressão linear. O SGDRegressor é adequado para problemas de regressão com um grande número de amostras de treinamento (> 10.000), para outros problemas recomendamos Ridge, Lasso ou ElasticNet.

    # A função de perda de concreto pode ser definida através do parâmetro de perda. SGDRegressor suporta as seguintes funções de perda:

        # loss="squared_error": mínimos quadrados comuns,

        # loss="huber": perda de Huber para regressão robusta,

        # loss="epsilon_insensitive": regressão linear do vetor de suporte.

    # Por favor, consulte a seção matemática abaixo para fórmulas. As funções de perda insensíveis a Huber e epsilon podem ser usadas para regressão robusta. A largura da região insensível deve ser especificada através do parâmetro epsilon. Este parâmetro depende da escala das variáveis ​​de destino.

    # O parâmetro de penalidade determina a regularização a ser utilizada (veja a descrição acima na seção de classificação).

    # O SGDRegressor também suporta SGD médio de 10 (aqui novamente, veja a descrição acima na seção de classificação).

    # Para regressão com uma perda quadrada e uma penalidade l2, outra variante do SGD com uma estratégia de média está disponível com o algoritmo Stochastic Average Gradient (SAG), disponível como um solver em Ridge. 