########## 2.6.4. Estimativa de covariância robusta  ##########

    # Conjuntos de dados reais geralmente estão sujeitos a erros de medição ou registro. Observações regulares, mas incomuns, também podem aparecer por vários motivos. As observações que são muito incomuns são chamadas de outliers. O estimador de covariância empírico e os estimadores de covariância reduzida apresentados acima são muito sensíveis à presença de outliers nos dados. Portanto, deve-se usar estimadores de covariância robustos para estimar a covariância de seus conjuntos de dados reais. Alternativamente, estimadores de covariância robustos podem ser usados para realizar detecção de valores discrepantes e descartar / reduzir algumas observações de acordo com o processamento posterior dos dados.

    # O pacote sklearn.covariance implementa um estimador robusto de covariância, o Determinante de Covariância Mínimo 3. 




##### 2.6.4.1. Minimum Covariance Determinant


    # O estimador do Determinante de Covariância Mínima é um estimador robusto da covariância de um conjunto de dados introduzido por P.J. Rousseeuw em 3. A ideia é encontrar uma determinada proporção (h) de observações "boas" que não são discrepantes e calcular sua matriz de covariância empírica. Essa matriz de covariância empírica é então redimensionada para compensar a seleção de observações realizada (“etapa de consistência”). Tendo calculado o estimador do Determinante de Covariância Mínima, pode-se dar pesos às observações de acordo com sua distância de Mahalanobis, levando a uma estimativa reponderada da matriz de covariância do conjunto de dados (“etapa de reponderação”).

    # Rousseeuw e Van Driessen 4 desenvolveram o algoritmo FastMCD para calcular o Determinante de Covariância Mínima. Este algoritmo é usado no scikit-learn ao ajustar um objeto MCD aos dados. O algoritmo FastMCD também calcula uma estimativa robusta da localização do conjunto de dados ao mesmo tempo.

    # As estimativas brutas podem ser acessadas como atributos raw_location_ e raw_covariance_ de um objeto estimador de covariância robusto MinCovDet. 


    ## Referências:

    ## P. J. Rousseeuw. Least median of squares regression. J. Am Stat Ass, 79:871, 1984.

    ## A Fast Algorithm for the Minimum Covariance Determinant Estimator, 1999, American Statistical Association and the American Society for Quality, TECHNOMETRICS.



    ## Exemplos:

    ## See Robust vs Empirical covariance estimate for an example on how to fit a MinCovDet object to data and see how the estimate remains accurate despite the presence of outliers. (https://scikit-learn.org/stable/auto_examples/covariance/plot_robust_vs_empirical_covariance.html#sphx-glr-auto-examples-covariance-plot-robust-vs-empirical-covariance-py)

    ## See Robust covariance estimation and Mahalanobis distances relevance to visualize the difference between EmpiricalCovariance and MinCovDet covariance estimators in terms of Mahalanobis distance (so we get a better estimate of the precision matrix too). (https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html#sphx-glr-auto-examples-covariance-plot-mahalanobis-distances-py)



        # https://scikit-learn.org/stable/auto_examples/covariance/plot_robust_vs_empirical_covariance.html


        # https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html