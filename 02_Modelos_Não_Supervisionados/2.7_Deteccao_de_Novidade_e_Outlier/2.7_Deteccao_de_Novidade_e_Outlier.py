########## 2.7. Detecção de novidade e outlier ##########

    # Muitas aplicações requerem a capacidade de decidir se uma nova observação pertence à mesma distribuição das observações existentes (é um inlier) ou deve ser considerada diferente (é um outlier). Freqüentemente, essa capacidade é usada para limpar conjuntos de dados reais. Duas importantes distinções devem ser feitas:

        # detecção de outlier
        # Os dados de treinamento contêm outliers que são definidos como observações distantes umas das outras. Os estimadores de detecção de outliers, portanto, tentam ajustar as regiões onde os dados de treinamento estão mais concentrados, ignorando as observações de desvio.

        # detecção de novidade
        # Os dados de treinamento não estão poluídos por outliers e estamos interessados ​​em detectar se uma nova observação é outlier. Nesse contexto, um outlier também é chamado de novidade.

    # A detecção de outlier e a detecção de novidade são usadas para detecção de anomalias, onde se está interessado em detectar observações anormais ou incomuns. A detecção de outliers é também conhecida como detecção de anomalia não supervisionada e detecção de novidade como detecção de anomalia semissupervisionada. No contexto da detecção de outliers, os outliers / anomalias não podem formar um cluster denso, pois os estimadores disponíveis assumem que os outliers / anomalias estão localizados em regiões de baixa densidade. Ao contrário, no contexto de detecção de novidades, novidades / anomalias podem formar um cluster denso, desde que estejam em uma região de baixa densidade dos dados de treinamento, considerada normal neste contexto.

    # O projeto scikit-learn fornece um conjunto de ferramentas de aprendizado de máquina que podem ser usadas para detecção de novidades ou outliers. Esta estratégia é implementada com objetos aprendendo de forma não supervisionada a partir dos dados: 


        # estimator.fit(X_train)

    # novas observações podem ser classificadas como inliers ou outliers com um método de previsão: 

        # estimator.predict(X_test)  


    # Inliers são rotulados como 1, enquanto outliers são rotulados como -1. O método de previsão faz uso de um limite na função de pontuação bruta calculada pelo estimador. Essa função de pontuação é acessível por meio do método score_samples, enquanto o limite pode ser controlado pelo parâmetro de contaminação.

    # O método de função de decisão também é definido a partir da função de pontuação, de forma que os valores negativos sejam outliers e os não negativos sejam inliers: 

        # estimator.decision_function(X_test)
    
    # Observe que neighbours.LocalOutlierFactor não suporta os métodos predizer, decision_function e score_samples por padrão, mas apenas um método fit_predict, já que este estimador foi originalmente concebido para ser aplicado para detecção de outlier. As pontuações de anormalidade das amostras de treinamento são acessíveis por meio do atributo negative_outlier_factor_.

    # Se você realmente deseja usar neighbours.LocalOutlierFactor para detecção de novidades, ou seja, prever rótulos ou calcular a pontuação de anormalidade de novos dados não vistos, você pode instanciar o estimador com o parâmetro de novidade definido como True antes de ajustar o estimador. Nesse caso, fit_predict não está disponível. 


    # Aviso: detecção de novidade com fator de outlier local
    # Quando a novidade é definida como True, esteja ciente de que você só deve usar predizer, decision_function e score_samples em novos dados não vistos e não nas amostras de treinamento, pois isso levaria a resultados errados. As pontuações de anormalidade das amostras de treinamento estão sempre acessíveis por meio do atributo negative_outlier_factor_. 


    # O comportamento de neighbours.LocalOutlierFactor é resumido na tabela a seguir. 

#   Método         Detecção de outlier             Detecção de novidade 
#   fit_predict             ok                              Não disponível 
#   predict                 Não disponível                  Use apenas em novos dados 
#   decision_function       Não disponível                  Use apenas em novos dados 
#   score_samples           Use negative_outlier_factor_    Use apenas em novos dados 
