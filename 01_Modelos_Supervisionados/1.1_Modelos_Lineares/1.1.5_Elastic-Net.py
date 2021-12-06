########## 1.1.5 Elastic-Net ##########

    # ElasticNet é um modelo de regressão linear treinado tanto com l1 e l2 regularização normativa dos coeficientes. Esta combinação permite aprender um modelo esparso onde poucos dos pesos são diferentes de zero como Lasso, enquanto ainda mantém as propriedades de regularização de Ridge. Controlamos a combinação convexa de l1 e l2 usando o parâmetro l1_ratio.

    # Elastic-net é útil quando há várias variáveis correlacionados entre si. Lasso provavelmente escolherá um desses aleatoriamente, enquanto elástico-net provavelmente escolherá ambos.

    # Uma vantagem prática da troca entre Lasso e Ridge é que permite que Elastic-Net herde parte da estabilidade de Ridge sob rotação.

    # A função objetivo a minimizar é, neste caso,

        # min 1/2nsamples ||Xw - y||22 + alpha p||1 + alpha(1-p)/2 ||w||22

    # A classe ElasticNetCV pode ser usada para definir os parâmetros alpha() e l1_ratio(p) por validação cruzada.

    ## Exemplos: 
    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py
    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html#sphx-glr-auto-examples-linear-model-plot-lasso-coordinate-descent-path-py

    # As duas referências a seguir explicam as iterações usadas no solucionador de descida por coordenadas do scikit-learn, bem como o cálculo de lacuna de dualidade usado para controle de convergência. 

    # Referências:
    # “Regularization Path For Generalized linear Models by Coordinate Descent”, Friedman, Hastie & Tibshirani, J Stat Softw, 2010 (Paper).
    # “An Interior-Point Method for Large-Scale L1-Regularized Least Squares,” S. J. Kim, K. Koh, M. Lustig, S. Boyd and D. Gorinevsky, in IEEE Journal of Selected Topics in Signal Processing, 2007 (Paper)
