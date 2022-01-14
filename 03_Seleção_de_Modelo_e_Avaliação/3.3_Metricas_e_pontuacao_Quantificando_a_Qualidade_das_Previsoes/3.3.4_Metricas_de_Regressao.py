##########  3.3.4. Métricas de regressão ##########

    # O módulo sklearn.metrics implementa várias funções de perda, pontuação e utilidade para medir o desempenho da regressão. Alguns deles foram aprimorados para lidar com o caso de saída múltipla: mean_squared_error, mean_absolute_error, Explain_variance_score, r2_score e mean_pinball_loss.

    # Essas funções têm um argumento de palavra-chave de saída múltipla que especifica a forma como as pontuações ou perdas para cada alvo individual devem ser calculadas em média. O padrão é 'uniform_average', que especifica uma média ponderada uniformemente sobre as saídas. Se um ndarray de forma (n_outputs,) for passado, suas entradas serão interpretadas como pesos e uma média ponderada correspondente será retornada. Se multioutput for 'raw_values' for especificado, todas as pontuações ou perdas individuais inalteradas serão retornadas em uma matriz de formato (n_outputs,).

    # O r2_score e o Explain_variance_score aceitam um valor adicional 'variance_weighted' para o parâmetro multioutput. Esta opção leva a uma ponderação de cada pontuação individual pela variância da variável alvo correspondente. Essa configuração quantifica a variação não dimensionada capturada globalmente. Se as variáveis-alvo forem de escala diferente, essa pontuação dá mais importância à explicação das variáveis ​​de maior variância. multioutput='variance_weighted' é o valor padrão para r2_score para compatibilidade com versões anteriores. Isso será alterado para uniform_average no futuro. 





##### 3.3.4.1. Pontuação de variação explicada

    # O Explain_variance_score calcula a pontuação de regressão de variância explicada.

    # Se \hat{y} for o output alvo estimado, y o output alvo correspondente (correto) e Var for Variação, o quadrado do desvio padrão, então a variação explicada é estimada da seguinte forma:

        # explicado\_{}variância(y, \hat{y}) = 1 - \frac{Var\{ y - \hat{y}\}}{Var\{y\}}

    # A melhor pontuação possível é 1,0, valores mais baixos são piores.

    # Aqui está um pequeno exemplo de uso da função Explain_variance_score: 


from sklearn.metrics import explained_variance_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
explained_variance_score(y_true, y_pred)
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
explained_variance_score(y_true, y_pred, multioutput='raw_values')
explained_variance_score(y_true, y_pred, multioutput=[0.3, 0.7])





##### 3.3.4.2. Erro máximo


    # A função max_error calcula o erro residual máximo , uma métrica que captura o erro de pior caso entre o valor previsto e o valor verdadeiro. Em um modelo de regressão de saída única perfeitamente ajustado, max_error seria 0 no conjunto de treinamento e, embora isso fosse altamente improvável no mundo real, essa métrica mostra a extensão do erro que o modelo tinha quando foi ajustado.

    # Se \hat{y}_i for o valor previsto da i-ésima amostra e y_i for o valor verdadeiro correspondente, então o erro máximo é definido como

        # \text{Max Error}(y, \hat{y}) = max(| y_i - \hat{y}_i |)

    # Aqui está um pequeno exemplo de uso da função max_error: 

from sklearn.metrics import max_error
y_true = [3, 2, 7, 1]
y_pred = [9, 2, 7, 1]
max_error(y_true, y_pred)

    # O max_error não suporta saída múltipla. 

##### 3.3.4.3. Erro absoluto médio

    # A função mean_absolute_error calcula o erro absoluto médio, uma métrica de risco correspondente ao valor esperado da perda de erro absoluta ou perda da norma l1.

    # Se \hat{y}_i é o valor previsto da i-ésima amostra, e y_i é o valor verdadeiro correspondente, então o erro absoluto médio (MAE) estimado sobre n_{\text{amostras}} é definido como

        # \text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{amostras}}} \sum_{i=0}^{n_{\text{amostras}}-1} \esquerda| y_i - \hat{y}_i \right|.

    # Aqui está um pequeno exemplo de uso da função mean_absolute_error: 

from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_absolute_error(y_true, y_pred)
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
mean_absolute_error(y_true, y_pred)
mean_absolute_error(y_true, y_pred, multioutput='raw_values')
mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])



##### 3.3.4.4. Erro quadrático médio

    # A função mean_squared_error calcula o erro quadrático médio, uma métrica de risco correspondente ao valor esperado do erro ou perda quadrática (quadrática).

    # Se \hat{y}_i é o valor previsto da i-ésima amostra, e y_i é o valor verdadeiro correspondente, então o erro quadrático médio (MSE) estimado sobre n_{\text{amostras}} é definido como


        # \text{MSE}(y, \hat{y}) = \frac{1}{n_\text{amostras}} \sum_{i=0}^{n_\text{amostras} - 1} (y_i - \ chapéu{y}_i)^2.

    # Aqui está um pequeno exemplo de uso da função mean_squared_error: 

from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
mean_squared_error(y_true, y_pred)



    ## Exemplos:

    ## See Gradient Boosting regression for an example of mean squared error usage to evaluate gradient boosting regression. (https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py)



##### 3.3.4.5. Erro logarítmico quadrado médio

    # A função mean_squared_log_error calcula uma métrica de risco correspondente ao valor esperado do erro ou perda logarítmica quadrada (quadrática).

    # Se \hat{y}_i é o valor previsto da i-ésima amostra, e y_i é o valor verdadeiro correspondente, então o erro logarítmico quadrático médio (MSLE) estimado em n_{\text{amostras}} é definido como 

        # \text{MSLE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2.

    # Onde \log_e (x) significa o logaritmo natural de x. Essa métrica é melhor para usar quando as metas têm crescimento exponencial, como contagens de população, vendas médias de uma mercadoria ao longo de um período de anos etc. Observe que essa métrica penaliza uma estimativa subestimada maior do que uma estimativa superestimada.

    # Aqui está um pequeno exemplo de uso da função mean_squared_log_error: 

from sklearn.metrics import mean_squared_log_error
y_true = [3, 5, 2.5, 7]
y_pred = [2.5, 5, 4, 8]
mean_squared_log_error(y_true, y_pred)
y_true = [[0.5, 1], [1, 2], [7, 6]]
y_pred = [[0.5, 2], [1, 2.5], [8, 8]]
mean_squared_log_error(y_true, y_pred)





##### 3.3.4.6. Erro percentual absoluto médio


    # O mean_absolute_percentage_error (MAPE), também conhecido como desvio percentual absoluto médio (MAPD), é uma métrica de avaliação para problemas de regressão. A ideia dessa métrica é ser sensível a erros relativos. Por exemplo, não é alterado por uma escala global da variável de destino.

    # Se \hat{y}_i é o valor previsto da i-ésima amostra e y_i é o valor verdadeiro correspondente, então o erro percentual médio absoluto (MAPE) estimado sobre n_{\text{amostras}} é definido como 


        # \text{MAPE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \frac{{}\left| y_i - \hat{y}_i \right|}{max(\epsilon, \left| y_i \right|)}



    # onde \epsilon é um número arbitrário pequeno mas estritamente positivo para evitar resultados indefinidos quando y é zero.

    # A função mean_absolute_percentage_error suporta saída múltipla.

    # Aqui está um pequeno exemplo de uso da função mean_absolute_percentage_error: 

from sklearn.metrics import mean_absolute_percentage_error
y_true = [1, 10, 1e6]
y_pred = [0.9, 15, 1.2e6]
mean_absolute_percentage_error(y_true, y_pred)


    # No exemplo acima, se tivéssemos usado mean_absolute_error, ele teria ignorado os valores de pequena magnitude e refletido apenas o erro na previsão do valor de maior magnitude. Mas esse problema é resolvido no caso do MAPE porque calcula o erro percentual relativo em relação à saída real. 



##### 3.3.4.7. Erro absoluto mediano


    # O median_absolute_error é particularmente interessante porque é robusto a valores discrepantes. A perda é calculada tomando a mediana de todas as diferenças absolutas entre a meta e a previsão.

    # Se \hat{y}_i é o valor previsto da i-ésima amostra e y_i é o valor verdadeiro correspondente, então o erro médio absoluto (MedAE) estimado sobre n_{\text{amostras}} é definido como 

        # \text{MedAE}(y, \hat{y}) = \text{median}(\mid y_1 - \hat{y}_1 \mid, \ldots, \mid y_n - \hat{y}_n \mid).

    # O median_absolute_error não suporta saídas múltiplas.

    # Aqui está um pequeno exemplo de uso da função median_absolute_error: 

from sklearn.metrics import median_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
median_absolute_error(y_true, y_pred)




##### 3.3.4.8. R² score, o coeficiente de determinação












##### 3.3.4.9. Desvios médios de Poisson, Gamma e Tweedie












##### 3.3.4.10. Pontuação D², o coeficiente de determinação












##### 3.3.4.11. Perda de pinball 




