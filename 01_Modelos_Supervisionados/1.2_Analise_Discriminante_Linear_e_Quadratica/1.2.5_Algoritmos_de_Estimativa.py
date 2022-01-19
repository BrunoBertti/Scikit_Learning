########## 1.2.5. Algoritmos de estimativa  ##########

    # O uso de LDA e QDA requer o cálculo do log-posterior que depende das classes prioritárias P(y=k), a classe significa \mu_k e as matrizes de covariância.

    # O solucionador 'svd' é o solucionador padrão usado para LinearDiscriminantAnalysis e é o único solucionador disponível para QuadraticDiscriminantAnalysis. Ele pode realizar tanto classificação quanto transformação (para LDA). Como não depende do cálculo da matriz de covariâncias, o solver ‘svd’ pode ser preferível em situações onde o número de características é grande. O solucionador 'svd' não pode ser usado com encolhimento. Para QDA, o uso do solver SVD depende do fato de que a matriz de covariância \Sigma_k é, por definição, igual a \frac{1}{n - 1}X_k^tX_k = \frac{1}{n - 1} VS^2 V^t onde V vem do SVD da matriz (centralizada): X_k = USV^t. Acontece que podemos calcular o log-posterior acima sem ter que calcular explicitamente \Sigma: calcular S e V através do SVD de X é suficiente. Para LDA, dois SVDs são calculados: o SVD da matriz de entrada centralizada X e o SVD dos vetores médios de classe.

    # O solver ‘lsqr’ é um algoritmo eficiente que só funciona para classificação. Ele precisa calcular explicitamente a matriz de covariância \Sigma e suporta estimadores de encolhimento e covariância personalizados. Este solucionador calcula os coeficientes \omega_k = \Sigma^{-1}\mu_k resolvendo para \Sigma \omega =\mu_k, evitando assim o cálculo explícito do inverso \Sigma^{-1}.

    # O solucionador 'eigen' é baseado na otimização da taxa de dispersão entre classes para dentro da classe de dispersão. Ele pode ser usado tanto para classificação quanto para transformação, e suporta encolhimento. No entanto, o solver 'eigen' precisa calcular a matriz de covariância, portanto, pode não ser adequado para situações com um grande número de recursos. 



    ## Referências:

    ## 1(1,2) “The Elements of Statistical Learning”, Hastie T., Tibshirani R., Friedman J., Section 4.3, p.106-119, 2008.

    ## 2 Ledoit O, Wolf M. Honey, I Shrunk the Sample Covariance Matrix. The Journal of Portfolio Management 30(4), 110-119, 2004.

    ## 3 R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification (Second Edition), section 2.6.2.