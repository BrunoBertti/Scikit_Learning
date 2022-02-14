########## 6.8.7. Kernel Qui-quadrado  ##########


    # O kernel qui-quadrado é uma escolha muito popular para treinar SVMs não lineares em aplicações de visão computacional. Ele pode ser calculado usando chi2_kernel e depois passado para um SVC com kernel="precomputed": 


from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
X = [[0, 1], [1, 0], [.2, .8], [.7, .3]]
y = [0, 1, 0, 1]
K = chi2_kernel(X, gamma=.5)
K
#array([[1.        , 0.36787944, 0.89483932, 0.58364548],
#       [0.36787944, 1.        , 0.51341712, 0.83822343],
#       [0.89483932, 0.51341712, 1.        , 0.7768366 ],
#       [0.58364548, 0.83822343, 0.7768366 , 1.        ]])

svm = SVC(kernel='precomputed').fit(K, y)
svm.predict(K)
#array([0, 1, 0, 1])


    # Ele também pode ser usado diretamente como o argumento do kernel: 

svm = SVC(kernel=chi2_kernel).fit(X, y)
svm.predict(X)
#array([0, 1, 0, 1])

    # O núcleo qui-quadrado é dado por 


        # k(x, y) = \exp \left (-\gamma \sum_i \frac{(x[i] - y[i]) ^ 2}{x[i] + y[i]} \right )


    # Os dados são assumidos como não negativos e geralmente são normalizados para ter uma norma L1 de um. A normalização é racionalizada com a conexão com a distância qui-quadrado, que é uma distância entre distribuições de probabilidade discretas.

    # O núcleo qui quadrado é mais comumente usado em histogramas (sacos) de palavras visuais. 



    ## Referências:

    ## Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C. Local features and kernels for classification of texture and object categories: A comprehensive study International Journal of Computer Vision 2007 (https://research.microsoft.com/en-us/um/people/manik/projects/trade-off/papers/ZhangIJCV06.pdf)