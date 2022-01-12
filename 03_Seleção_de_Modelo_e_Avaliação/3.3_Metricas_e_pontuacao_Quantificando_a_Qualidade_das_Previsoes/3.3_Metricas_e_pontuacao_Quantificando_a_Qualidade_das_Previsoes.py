########## 3.3. Métricas e pontuação: quantificando a qualidade das previsões ##########


    # Existem 3 APIs diferentes para avaliar a qualidade das previsões de um modelo:

        # Método de pontuação do estimador: Os estimadores têm um método de pontuação que fornece um critério de avaliação padrão para o problema que eles foram projetados para resolver. Isso não é discutido nesta página, mas na documentação de cada estimador.

        # Parâmetro de pontuação: As ferramentas de avaliação de modelo usando validação cruzada (como model_selection.cross_val_score e model_selection.GridSearchCV) contam com uma estratégia de pontuação interna. Isso é discutido na seção O parâmetro de pontuação: definindo regras de avaliação de modelo.

        # Funções métricas: O módulo sklearn.metrics implementa funções que avaliam erros de previsão para fins específicos. Essas métricas são detalhadas em seções sobre Métricas de classificação, Métricas de classificação Multilabel, Métricas de regressão e Métricas de agrupamento.

    # Por fim, os estimadores fictícios são úteis para obter um valor de linha de base dessas métricas para previsões aleatórias. 

    # Consulte também: Para métricas “em pares”, entre amostras e não estimadores ou previsões, consulte a seção Métricas, afinidades e kernels em pares. 