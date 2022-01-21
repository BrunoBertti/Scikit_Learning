########## 1.7.1. Regressão de Processo Gaussiana (GPR) ##########

    # O GaussianProcessRegressor implementa processos gaussianos (GP) para fins de regressão. Para isso, o anterior do GP precisa ser especificado. A média anterior é considerada constante e zero (para normalize_y=False) ou a média dos dados de treinamento (para normalize_y=True). A covariância do prior é especificada passando um objeto kernel. Os hiperparâmetros do kernel são otimizados durante o ajuste de GaussianProcessRegressor, maximizando a probabilidade de log-marginal (LML) com base no otimizador passado. Como o LML pode ter vários ótimos locais, o otimizador pode ser iniciado repetidamente especificando n_restarts_optimizer. A primeira execução é sempre realizada a partir dos valores iniciais dos hiperparâmetros do kernel; execuções subsequentes são realizadas a partir de valores de hiperparâmetros que foram escolhidos aleatoriamente a partir do intervalo de valores permitidos. Se os hiperparâmetros iniciais devem ser mantidos fixos, Nenhum pode ser passado como otimizador.

    # O nível de ruído nos alvos pode ser especificado passando-o por meio do parâmetro alfa, globalmente como escalar ou por ponto de dados. Observe que um nível de ruído moderado também pode ser útil para lidar com problemas numéricos durante o ajuste, pois é efetivamente implementado como regularização de Tikhonov, ou seja, adicionando-o à diagonal da matriz do kernel. Uma alternativa para especificar o nível de ruído explicitamente é incluir um componente WhiteKernel no kernel, que pode estimar o nível de ruído global a partir dos dados (veja o exemplo abaixo).

    # A implementação é baseada no Algoritmo 2.1 de [RW2006]. Além da API de estimadores scikit-learn padrão, GaussianProcessRegressor:

        # permite a previsão sem ajuste prévio (com base no GP anterior)

        # fornece um método adicional sample_y(X), que avalia amostras extraídas do GPR (anterior ou posterior) em determinadas entradas

        # expõe um método log_marginal_likelihood(theta), que pode ser usado externamente para outras formas de selecionar hiperparâmetros, por exemplo, via cadeia de Markov Monte Carlo. 