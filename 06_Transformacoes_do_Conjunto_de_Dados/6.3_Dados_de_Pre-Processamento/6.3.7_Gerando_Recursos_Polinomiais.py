########## 6.3.7. Gerando recursos polinomiais ##########



    # Muitas vezes, é útil adicionar complexidade a um modelo considerando recursos não lineares dos dados de entrada. Mostramos duas possibilidades que são ambas baseadas em polinômios: a primeira usa polinômios puros, a segunda usa splines, ou seja, polinômios por partes. 



##### 6.3.7.1. Recursos polinomiais

    # Um método simples e comum de usar são os recursos polinomiais, que podem obter termos de alta ordem e interação dos recursos. Ele é implementado em PolynomialFeatures: 

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
X

poly = PolynomialFeatures(2)
poly.fit_transform(X)


    # As características de X foram transformadas de (X_1, X_2) para (1, X_1, X_2, X_1^2, X_1X_2, X_2^2).

    # Em alguns casos, apenas os termos de interação entre os recursos são necessários, e isso pode ser obtido com a configuração transaction_only=True: 

X = np.arange(9).reshape(3, 3)
X

poly = PolynomialFeatures(degree=3, interaction_only=True)
poly.fit_transform(X)


    # cAs características de X foram transformadas de (X_1, X_2, X_3) para (1, X_1, X_2, X_3, X_1X_2, X_1X_3, X_2X_3, X_1X_2X_3).

    # cObserve que os recursos polinomiais são usados implicitamente em métodos de kernel (por exemplo, SVC, KernelPCA) ao usar funções de kernel polinomiais.

    # consulte Interpolação polinomial e spline para regressão Ridge usando recursos polinomiais criados. 



##### 6.3.7.2. Transformador de spline 

    # Outra maneira de adicionar termos não lineares em vez de polinômios puros de recursos é gerar funções de base spline para cada recurso com o SplineTransformer. Splines são polinômios por partes, parametrizados por seu grau polinomial e as posições dos nós. O SplineTransformer implementa uma base B-spline, cf. as referências abaixo. 

    # Nota: O SplineTransformer trata cada recurso separadamente, ou seja, não fornece termos de interação.

    # Algumas das vantagens de splines sobre polinômios são: 

        # B-splines são muito flexíveis e robustos se você mantiver um grau baixo fixo, geralmente 3, e adaptar parcimoniosamente o número de nós. Polinômios precisariam de um grau mais alto, o que leva ao próximo ponto.

        # B-splines não têm comportamento oscilatório nos limites como têm polinômios (quanto maior o grau, pior). Isso é conhecido como fenômeno de Runge.

        # B-splines fornecem boas opções para extrapolação além dos limites, ou seja, além do intervalo de valores ajustados. Dê uma olhada na extrapolação de opções.

        # B-splines geram uma matriz de recursos com uma estrutura em faixas. Para um único recurso, cada linha contém apenas grau + 1 elementos diferentes de zero, que ocorrem consecutivamente e são até positivos. Isso resulta em uma matriz com boas propriedades numéricas, e. um número de condição baixo, em nítido contraste com uma matriz de polinômios, que recebe o nome de matriz de Vandermonde. Um número de condição baixo é importante para algoritmos estáveis de modelos lineares. 

    # O snippet de código a seguir mostra splines em ação: 

import numpy as np
from sklearn.preprocessing import SplineTransformer
X = np.arange(5).reshape(5, 1)
X

spline = SplineTransformer(degree=2, n_knots=3)
spline.fit_transform(X)

    # À medida que o X é classificado, pode-se ver facilmente a saída da matriz em faixas. Apenas as três diagonais do meio são diferentes de zero para grau=2. Quanto maior o grau, mais sobreposição das splines.

    # Curiosamente, um SplineTransformer de grau=0 é o mesmo que KBinsDiscretizer com encode='onehot-dense' e n_bins = n_knots - 1 se knots = estratégia. 



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html#sphx-glr-auto-examples-linear-model-plot-polynomial-interpolation-py

    ## https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#sphx-glr-auto-examples-applications-plot-cyclical-feature-engineering-py





    ## Referências:

    ## Eilers, P., & Marx, B. (1996). Flexible Smoothing with B-splines and Penalties. Statist. Sci. 11 (1996), no. 2, 89–121. (https://doi.org/10.1214/ss/1038425655)

    ## Perperoglou, A., Sauerbrei, W., Abrahamowicz, M. et al. A review of spline function procedures in R. BMC Med Res Methodol 19, 46 (2019). (https://doi.org/10.1186/s12874-019-0666-3)


    