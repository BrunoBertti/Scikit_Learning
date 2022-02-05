########## 6.6.2. Projeção aleatória gaussiana ##########


    # A GaussianRandomProjection reduz a dimensionalidade projetando o espaço de entrada original em uma matriz gerada aleatoriamente onde os componentes são desenhados da seguinte distribuição N(0, \frac{1}{n_{components}}).

    # Aqui um pequeno trecho que ilustra como usar o transformador de projeção aleatória gaussiana:

import numpy as np
from sklearn import random_projection
X = np.random.rand(100, 10000)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
X_new.shape