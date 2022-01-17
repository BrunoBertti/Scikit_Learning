########## 3.4. Curvas de validação: plotando pontuações para avaliar modelos ##########

    # Todo estimador tem suas vantagens e desvantagens. Seu erro de generalização pode ser decomposto em termos de viés, variância e ruído. O viés de um estimador é seu erro médio para diferentes conjuntos de treinamento. A variância de um estimador indica o quão sensível ele é à variação dos conjuntos de treinamento. O ruído é uma propriedade dos dados.

    # No gráfico a seguir, vemos uma função f(x) = \cos (\frac{3}{2} \pi x) e algumas amostras ruidosas dessa função. Usamos três estimadores diferentes para ajustar a função: regressão linear com características polinomiais de grau 1, 4 e 15. Vemos que o primeiro estimador pode, na melhor das hipóteses, fornecer apenas um ajuste ruim para as amostras e a função verdadeira porque é muito simples ( alto viés), o segundo estimador aproxima-o quase perfeitamente e o último estimador aproxima os dados de treinamento perfeitamente, mas não se ajusta muito bem à função verdadeira, ou seja, é muito sensível à variação dos dados de treinamento (alta variância). 

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html
    
    # O viés e a variância são propriedades inerentes dos estimadores e geralmente temos que selecionar algoritmos de aprendizado e hiperparâmetros para que tanto o viés quanto a variância sejam os mais baixos possíveis (consulte Dilema de viés-variância). Outra maneira de reduzir a variação de um modelo é usar mais dados de treinamento. No entanto, você só deve coletar mais dados de treinamento se a função verdadeira for muito complexa para ser aproximada por um estimador com uma variância menor.

    # No problema unidimensional simples que vimos no exemplo, é fácil ver se o estimador sofre de viés ou variância. No entanto, em espaços de alta dimensão, os modelos podem se tornar muito difíceis de visualizar. Por esse motivo, geralmente é útil usar as ferramentas descritas abaixo. 



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py

    ##  https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py

    ##  https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py