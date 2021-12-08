########## 1.1.17 Regressão Quantil ##########

    # A regressão de quantis estima a mediana ou outros quantis de y condicionais em X, enquanto os mínimos quadrados ordinários (OLS) estimam a média condicional.

    # Como um modelo linear, o QuantileRegressor fornece previsões lineares y ^ (w, X) = Xw para o q-ésimo quantil, q pertence (0,1). Os pesos ou coeficientes w são então encontrados pelo seguinte problema de minimização: 

        # min 1/nsamples Somatório PBq(yi - Xiw) + alpha||w||1

    # Isso consiste na perda de pinball (também conhecida como perda linear), consulte também mean_pinball_loss, 

        # PBq(t) = qmax(t,0) + (1 - q) max(-t, 0) = qt, t>0
        #                                         = 0, t=0
        #                                         =(1-q)t, t<0


    # Como a perda de pinball é apenas linear nos resíduos, a regressão de quantis é muito mais robusta para outliers do que a estimativa da média baseada no erro quadrático. Um pouco no meio está o HuberRegressor.

    # A regressão quantílica pode ser útil se alguém estiver interessado em prever um intervalo em vez de uma previsão de ponto. Às vezes, os intervalos de predição são calculados com base na suposição de que o erro de predição é distribuído normalmente com média zero e variância constante. A regressão de quantil fornece intervalos de previsão sensíveis, mesmo para erros com variância não constante (mas previsível) ou distribuição não normal. 


    # Com base na minimização da perda de pinball, os quantis condicionais também podem ser estimados por modelos diferentes dos modelos lineares. Por exemplo, GradientBoostingRegressor pode prever quantis condicionais se sua perda de parâmetro for definida como "quantil" e o parâmetro alfa for definido como o quantil que deve ser previsto. Veja o exemplo em Intervalos de previsão para regressão de aumento de gradiente.

    # A maioria das implementações de regressão quantílica é baseada no problema de programação linear. A implementação atual é baseada em scipy.optimize.linprog. 

    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html#sphx-glr-auto-examples-linear-model-plot-quantile-regression-py


    ## Referências:

    ## Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. Econometrica: journal of the Econometric Society, 33-50. (https://gib.people.uic.edu/RQ.pdf)

    ## Portnoy, S., & Koenker, R. (1997). The Gaussian hare and the Laplacian tortoise: computability of squared-error versus absolute-error estimators. Statistical Science, 12, 279-300. (https://doi.org/10.1214/ss/1030037960)

    ## Koenker, R. (2005). Quantile Regression. Cambridge University Press. (https://doi.org/10.1017/CBO9780511754098)