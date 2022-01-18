########## 1.2.4. Estimador de contração e covariância ##########


    # Shrinkage é uma forma de regularização utilizada para melhorar a estimação de matrizes de covariância em situações onde o número de amostras de treinamento é pequeno em relação ao número de características. Nesse cenário, a covariância da amostra empírica é um estimador ruim e o encolhimento ajuda a melhorar o desempenho de generalização do classificador. Shrinkage LDA pode ser usado definindo o parâmetro de redução da classe LinearDiscriminantAnalysis como 'auto'. Isso determina automaticamente o parâmetro de encolhimento ideal de forma analítica seguindo o lema introduzido por Ledoit e Wolf 2. Observe que atualmente o encolhimento só funciona ao definir o parâmetro do solver como 'lsqr' ou 'eigen'.

    # O parâmetro de contração também pode ser definido manualmente entre 0 e 1. Em particular, um valor de 0 corresponde a nenhuma contração (o que significa que a matriz de covariância empírica será usada) e um valor de 1 corresponde a contração completa (o que significa que a diagonal matriz de variâncias será usada como uma estimativa para a matriz de covariância). Definir esse parâmetro para um valor entre esses dois extremos estimará uma versão reduzida da matriz de covariância.

    # O estimador de covariância reduzido de Ledoit e Wolf pode nem sempre ser a melhor escolha. Por exemplo, se a distribuição dos dados for normalmente distribuída, o estimador Oracle Shrinkage Approximating sklearn.covariance.OAS produzirá um erro quadrático médio menor do que o dado pela fórmula de Ledoit e Wolf usada com shrinkage=”auto”. No LDA, os dados são assumidos como gaussianos condicionalmente à classe. Se essas suposições forem válidas, o uso de LDA com o estimador de covariância OAS produzirá uma melhor precisão de classificação do que se Ledoit e Wolf ou o estimador de covariância empírico for usado.

    # O estimador de covariância pode ser escolhido usando o parâmetro covariance_estimator da classe discriminant_analysis.LinearDiscriminantAnalysis. Um estimador de covariância deve ter um método de ajuste e um atributo covariance_ como todos os estimadores de covariância no módulo sklearn.covariance. 

        # https://scikit-learn.org/stable/auto_examples/classification/plot_lda.html



    ## Exemplos:

    ## Normal, Ledoit-Wolf and OAS Linear Discriminant Analysis for classification: Comparison of LDA classifiers with Empirical, Ledoit Wolf and OAS covariance estimator. ( https://scikit-learn.org/stable/auto_examples/classification/plot_lda.html#sphx-glr-auto-examples-classification-plot-lda-py)