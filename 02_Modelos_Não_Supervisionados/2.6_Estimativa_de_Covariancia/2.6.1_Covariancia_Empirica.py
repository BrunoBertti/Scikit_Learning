########## 2.6.1. Covariância empírica ##########

    # A matriz de covariância de um conjunto de dados é conhecida por ser bem aproximada pelo estimador de máxima verossimilhança clássico (ou "covariância empírica"), desde que o número de observações seja grande o suficiente em comparação com o número de recursos (as variáveis que descrevem as observações). Mais precisamente, o Estimador de Máxima Verossimilhança de uma amostra é um estimador não enviesado assintoticamente da matriz de covariância da população correspondente.

    # A matriz de covariância empírica de uma amostra pode ser calculada usando a função empirical_covariance do pacote ou ajustando um objeto de EmpiricalCovariance à amostra de dados com o método EmpiricalCovariance.fit. Tenha cuidado para que os resultados dependam de os dados estarem centralizados, portanto, pode-se querer usar o parâmetro assume_centered com precisão. Mais precisamente, se assume_centered = False, o conjunto de teste deve ter o mesmo vetor médio que o conjunto de treinamento. Caso contrário, ambos devem ser centralizados pelo usuário, e assume_centered = True deve ser usado. 




    ## Exemplos:

    ## See Shrinkage covariance estimation: LedoitWolf vs OAS and max-likelihood for an example on how to fit an EmpiricalCovariance object to data. (https://scikit-learn.org/stable/auto_examples/covariance/plot_covariance_estimation.html#sphx-glr-auto-examples-covariance-plot-covariance-estimation-py)