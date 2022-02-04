########## 6.4.5. Atribuição de vizinhos mais próximos ##########





    # A classe KNNImputer fornece imputação para preencher valores ausentes usando a abordagem k-Nearest Neighbors. Por padrão, uma métrica de distância euclidiana que suporta valores ausentes, nan_euclidean_distances, é usada para localizar os vizinhos mais próximos. Cada recurso ausente é imputado usando valores de n_vizinhos vizinhos mais próximos que possuem um valor para o recurso. As características dos vizinhos são calculadas em média uniformemente ou ponderadas pela distância de cada vizinho. Se uma amostra tiver mais de um recurso ausente, os vizinhos dessa amostra podem ser diferentes dependendo do recurso específico que está sendo imputado. Quando o número de vizinhos disponíveis é menor que n_vizinhos e não há distâncias definidas para o conjunto de treinamento, a média do conjunto de treinamento para esse recurso é usada durante a imputação. Se houver pelo menos um vizinho com uma distância definida, a média ponderada ou não ponderada dos vizinhos restantes será utilizada durante a imputação. Se um recurso estiver sempre ausente no treinamento, ele será removido durante a transformação. Para mais informações sobre a metodologia, ver ref. [OL2001].

    # O snippet a seguir demonstra como substituir valores ausentes, codificados como np.nan, usando o valor de recurso médio dos dois vizinhos mais próximos de amostras com valores ausentes: 

import numpy as np
from sklearn.impute import KNNImputer
nan = np.nan
X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2, weights="uniform")
imputer.fit_transform(X)



    ## Referêcias:

    ## Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor Hastie, Robert Tibshirani, David Botstein and Russ B. Altman, Missing value estimation methods for DNA microarrays, BIOINFORMATICS Vol. 17 no. 6, 2001 Pages 520-