########## 6.8. Métricas de pares, afinidades e kernels ##########


    # O submódulo sklearn.metrics.pairwise implementa utilitários para avaliar distâncias de pares ou afinidade de conjuntos de amostras.

    # Este módulo contém métricas de distância e kernels. Um breve resumo é dado sobre os dois aqui.

    # Métricas de distância são funções d(a, b) tais que d(a, b) < d(a, c) se os objetos a e b forem considerados “mais semelhantes” que os objetos a e c. Dois objetos exatamente iguais teriam uma distância de zero. Um dos exemplos mais populares é a distância euclidiana. Para ser uma métrica "verdadeira", ela deve obedecer às quatro condições a seguir: 


        # 1. d(a, b) >= 0, para todos a e b 
        # 2. d(a, b) == 0, se e somente se a = b, definitude positiva
        # 3. d(a, b) == d(b, a), simetria
        # 4. d(a, c) <= d(a, b) + d(b, c), a desigualdade triangular 

    # Kernels são medidas de similaridade, ou seja, s(a, b) > s(a, c) se os objetos a e b são considerados “mais semelhantes” que os objetos a e c. Um kernel também deve ser positivo semidefinido.

    # Há várias maneiras de converter entre uma métrica de distância e uma medida de similaridade, como um kernel. Seja D a distância e S o kernel: 


        # S = np.exp(-D * gamma), onde uma heurística para escolher gama é 1 / num_features 

        # S = 1. / (D / np.max(D))


    # As distâncias entre os vetores de linha de X e os vetores de linha de Y podem ser avaliadas usando pairwise_distances. Se Y for omitido, as distâncias aos pares dos vetores de linha de X são calculadas. Da mesma forma, pairwise.pairwise_kernels pode ser usado para calcular o kernel entre X e Y usando diferentes funções do kernel. Consulte a referência da API para obter mais detalhes. 


import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
X = np.array([[2, 3], [3, 5], [5, 8]])
Y = np.array([[1, 0], [2, 1]])
pairwise_distances(X, Y, metric='manhattan')
#array([[ 4.,  2.],
#       [ 7.,  5.],
#       [12., 10.]])
pairwise_distances(X, metric='manhattan')
#array([[0., 3., 8.],
#       [3., 0., 5.],
#       [8., 5., 0.]])
pairwise_kernels(X, Y, metric='linear')
#array([[ 2.,  7.],
#       [ 3., 11.],
#       [ 5., 18.]])