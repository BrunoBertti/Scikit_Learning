########## 1.16.2. Calibrando um classificador ##########


    # Calibrar um classificador consiste em ajustar um regressor (denominado calibrador) que mapeia a saída do classificador (conforme dado por função_de_decisão ou previsão_proba) para uma probabilidade calibrada em [0, 1]. Denotando a saída do classificador para uma determinada amostra por f_i, o calibrador tenta prever p (y_i = 1 | f_i).


    # As amostras usadas para ajustar o calibrador não devem ser as mesmas usadas para ajustar o classificador, pois isso introduziria viés. Isso ocorre porque o desempenho do classificador em seus dados de treinamento seria melhor do que em dados novos. Usar a saída do classificador de dados de treinamento para ajustar o calibrador resultaria, portanto, em um calibrador enviesado que mapeia probabilidades mais próximas de 0 e 1 do que deveria. 