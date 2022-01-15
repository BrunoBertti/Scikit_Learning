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

    # A função r2_score calcula o coeficiente de determinação, geralmente denotado como R².

    # Representa a proporção da variância (de y) que foi explicada pelas variáveis independentes no modelo. Ele fornece uma indicação da qualidade do ajuste e, portanto, uma medida de quão bem as amostras não vistas provavelmente serão previstas pelo modelo, por meio da proporção da variância explicada.

    # Como essa variação depende do conjunto de dados, o R² pode não ser significativamente comparável em diferentes conjuntos de dados. A melhor pontuação possível é 1,0 e pode ser negativa (porque o modelo pode ser arbitrariamente pior). Um modelo constante que sempre prevê o valor esperado de y, desconsiderando os recursos de entrada, obteria uma pontuação R² de 0,0.

    # Se \hat{y}_i for o valor previsto da i-ésima amostra e y_i for o valor verdadeiro correspondente para o total de n amostras, o R² estimado é definido como: 


        # R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}


    # onde \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i e \sum_{i=1}^{n} (y_i - \hat{y}_i)^ 2 = \sum_{i=1}^{n} \epsilon_i^2.

    # Observe que r2_score calcula R² não ajustado sem corrigir o viés na variância amostral de y.

    # Aqui está um pequeno exemplo de uso da função r2_score: 

from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
r2_score(y_true, y_pred)
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
r2_score(y_true, y_pred, multioutput='variance_weighted')
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
r2_score(y_true, y_pred, multioutput='uniform_average')
r2_score(y_true, y_pred, multioutput='raw_values')
r2_score(y_true, y_pred, multioutput=[0.3, 0.7])


    ## Exemplos:

    ## See Lasso and Elastic Net for Sparse Signals for an example of R² score usage to evaluate Lasso and Elastic Net on sparse signals. (https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py)





##### 3.3.4.9. Desvios médios de Poisson, Gamma e Tweedie


    # A função mean_tweedie_deviance calcula o erro médio de desvio de Tweedie com um parâmetro de potência (P). Essa é uma métrica que obtém valores de expectativa previstos de alvos de regressão.

    # Existem casos especiais a seguir,

        # quando power=0 é equivalente a mean_squared_error.

        # quando power=1 é equivalente a mean_poisson_deviance.

        # quando power=2 é equivalente a mean_gamma_deviance.

    # Se \hat{y}_i é o valor previsto da i-ésima amostra, e y_i é o valor verdadeiro correspondente, então o erro médio de desvio de Tweedie (D) para potência p, estimado sobre n_{\text{amostras}} é definido como 


        # \begin{split}\text{D}(y, \hat{y}) = \frac{1}{n_\text{samples}}
        # \sum_{i=0}^{n_\text{samples} - 1}
        # \begin{cases}
        # (y_i-\hat{y}_i)^2, & \text{for }p=0\text{ (Normal)}\\
        # 2(y_i \log(y/\hat{y}_i) + \hat{y}_i - y_i),  & \text{for}p=1\text{ (Poisson)}\\
        # 2(\log(\hat{y}_i/y_i) + y_i/\hat{y}_i - 1),  & \text{for}p=2\text{ (Gamma)}\\
        # 2\left(\frac{\max(y_i,0)^{2-p}}{(1-p)(2-p)}-
        # \frac{y\,\hat{y}^{1-p}_i}{1-p}+\frac{\hat{y}^{2-p}_i}{2-p}\right),
        # & \text{otherwise}
        # \end{cases}\end{split}


    # O desvio de Tweedie é uma função homogênea do poder de grau 2. Assim, a distribuição Gamma com poder=2 significa que escalar simultaneamente y_true e y_pred não tem efeito sobre o desvio. Para a distribuição Poisson poder=1 a deviance escala linearmente, e para a distribuição Normal (potência=0), quadrática. Em geral, quanto maior a potência, menos peso é dado aos desvios extremos entre os alvos verdadeiros e previstos.

    # Por exemplo, vamos comparar as duas previsões 1.0 e 100 que são 50% de seu valor verdadeiro correspondente.

    # O erro quadrático médio (potência = 0) é muito sensível à diferença de previsão do segundo ponto: 

from sklearn.metrics import mean_tweedie_deviance
mean_tweedie_deviance([1.0], [1.5], power=0)

mean_tweedie_deviance([100.], [150.], power=0)

    # Se aumentarmos a potência para 1,: 

mean_tweedie_deviance([1.0], [1.5], power=1)

mean_tweedie_deviance([100.], [150.], power=1)

    # a diferença de erros diminui. Finalmente, definindo, power=2: 

mean_tweedie_deviance([1.0], [1.5], power=2)

mean_tweedie_deviance([100.], [150.], power=2)


    # obteríamos erros idênticos. O desvio quando power=2 é, portanto, apenas sensível a erros relativos. 





##### 3.3.4.10. Pontuação D², o coeficiente de determinação


    # A função d2_tweedie_score calcula a porcentagem de desvio explicada. É uma generalização do R², onde o erro quadrático é substituído pelo desvio de Tweedie. D², também conhecido como índice de razão de verossimilhança de McFadden, é calculado como

        # D^2(y, \hat{y}) = 1 - \frac{\text{D}(y, \hat{y})}{\text{D}(y, \bar{y})} \ ,.

 
    # O poder do argumento define o poder do Tweedie como para mean_tweedie_deviance. Observe que para power=0, d2_tweedie_score é igual a r2_score (para alvos únicos).

    # Assim como o R², a melhor pontuação possível é 1,0 e pode ser negativa (porque o modelo pode ser arbitrariamente pior). Um modelo constante que sempre prevê o valor esperado de y, desconsiderando os recursos de entrada, obteria uma pontuação D² de 0,0.

    # Um objeto marcador com uma escolha específica de poder pode ser construído por: 

from sklearn.metrics import d2_tweedie_score, make_scorer
d2_tweedie_score_15 = make_scorer(d2_tweedie_score, power=1.5)







##### 3.3.4.11. Perda de pinball 


    # A função mean_pinball_loss é usada para avaliar o desempenho preditivo de modelos de regressão quantílica. A perda de pinball é equivalente a mean_absolute_error quando o parâmetro de quantil alfa é definido como 0,5. 

        # \text{pinball}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1}  \alpha \max(y_i - \hat{y}_i, 0) + (1 - \alpha) \max(\hat{y}_i - y_i, 0)


from sklearn.metrics import mean_pinball_loss
y_true = [1, 2, 3]
mean_pinball_loss(y_true, [0, 2, 3], alpha=0.1)
mean_pinball_loss(y_true, [1, 2, 4], alpha=0.1)
mean_pinball_loss(y_true, [0, 2, 3], alpha=0.9)
mean_pinball_loss(y_true, [1, 2, 4], alpha=0.9)
mean_pinball_loss(y_true, y_true, alpha=0.1)
mean_pinball_loss(y_true, y_true, alpha=0.9)

    # É possível construir um objeto scorer com uma escolha específica de alfa: 

from sklearn.metrics import make_scorer
mean_pinball_loss_95p = make_scorer(mean_pinball_loss, alpha=0.95)

    # Esse marcador pode ser usado para avaliar o desempenho de generalização de um regressor quantil por meio de validação cruzada: 

from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
X, y = make_regression(n_samples=100, random_state=0)
estimator = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.95,
    random_state=0,
)
cross_val_score(estimator, X, y, cv=5, scoring=mean_pinball_loss_95p)

    # Também é possível construir objetos de pontuação para ajuste de hiperparâmetros. O sinal da perda deve ser trocado para garantir que maior significa melhor, conforme explicado no exemplo vinculado abaixo. 



    ## Exemplos:

    ## See Prediction Intervals for Gradient Boosting Regression for an example of using a the pinball loss to evaluate and tune the hyper-parameters of quantile regression models on data with non-symmetric noise and outliers. (https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-quantile-py)