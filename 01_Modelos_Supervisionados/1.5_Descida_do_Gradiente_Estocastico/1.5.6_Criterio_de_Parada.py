########## 1.5.6. Critério de parada ##########

    # As classes SGDClassifier e SGDRegressor fornecem dois critérios para parar o algoritmo quando um determinado nível de convergência é alcançado:

        # Com early_stopping=True, os dados de entrada são divididos em um conjunto de treinamento e um conjunto de validação. O modelo é então ajustado no conjunto de treinamento e o critério de parada é baseado na pontuação de previsão (usando o método de pontuação) calculada no conjunto de validação. O tamanho do conjunto de validação pode ser alterado com o parâmetro validation_fraction.

        # Com early_stopping=False, o modelo é ajustado em todos os dados de entrada e o critério de parada é baseado na função objetivo calculada nos dados de treinamento.

    # Em ambos os casos, o critério é avaliado uma vez por época e o algoritmo para quando o critério não melhora n_iter_no_change vezes seguidas. A melhoria é avaliada com tolerância absoluta tol, e o algoritmo para em qualquer caso após um número máximo de iteração max_iter. 