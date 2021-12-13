########## 1.6.5 Nearest Classificador Centrals ##########

    # O classificador NearestCentroid é um algoritmo simples que representa cada classe pelo centróide de seus membros. Na verdade, isso o torna semelhante à fase de atualização de rótulo do algoritmo KMeans. Ele também não tem parâmetros para escolher, o que o torna um bom classificador de linha de base. No entanto, ela sofre em classes não convexas, bem como quando as classes têm variâncias drasticamente diferentes, uma vez que a variância igual em todas as dimensões é assumida. Consulte Análise Discriminante Linear (Análise Discriminante Linear) e Análise Discriminante Quadrática (Análise Discriminante Quadrática) para métodos mais complexos que não fazem essa suposição. O uso do NearestCentroid padrão é simples: 

import numpy
from sklearn.neighbors import NearestCentroid
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = NearestCentroid()
print(clf.fit(X, y))

print(clf.predict([[-0.8, -1]]))



##### 1.6.5.1 Centróide encolhido mais próximo 

    # O classificador NearestCentroid tem um parâmetro shrink_threshold, que implementa o classificador de centróide reduzido mais próximo. Na verdade, o valor de cada recurso para cada centróide é dividido pela variância dentro da classe desse recurso. Os valores do recurso são então reduzidos por shrink_threshold. Mais notavelmente, se um determinado valor de recurso ultrapassar zero, ele será definido como zero. Na verdade, isso impede que o recurso afete a classificação. Isso é útil, por exemplo, para remover recursos ruidosos. 

    # No exemplo abaixo, o uso de um pequeno limite de redução aumenta a precisão do modelo de 0,81 para 0,82. 

        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_nearest_centroid.html

        #https://scikit-learn.org/stable/auto_examples/neighbors/plot_nearest_centroid.html


    # Exemplos:
    
    ## Nearest Centroid Classification: an example of classification using nearest centroid with different shrink thresholds. (https://scikit-learn.org/stable/auto_examples/neighbors/plot_nearest_centroid.html#sphx-glr-auto-examples-neighbors-plot-nearest-centroid-py)