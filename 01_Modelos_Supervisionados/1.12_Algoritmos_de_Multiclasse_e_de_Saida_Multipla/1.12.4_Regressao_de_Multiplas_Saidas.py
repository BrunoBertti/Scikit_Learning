########## 1.12.4. Regressão de múltiplas saídas  ##########

    # A regressão de múltiplas saídas prevê várias propriedades numéricas para cada amostra. Cada propriedade é uma variável numérica e o número de propriedades a serem previstas para cada amostra é maior ou igual a 2. Alguns estimadores que oferecem suporte à regressão de múltiplas saídas são mais rápidos do que apenas executar estimadores n_output.

    # Por exemplo, a previsão da velocidade e direção do vento, em graus, usando dados obtidos em um determinado local. Cada amostra seria um dado obtido em um local e a velocidade e a direção do vento seriam geradas para cada amostra. 



##### 1.12.4.1. Formato de destino

    # Uma representação válida de multioutput y é uma matriz densa de forma (n_samples, n_output) de floats. Uma concatenação inteligente de colunas de variáveis contínuas. Um exemplo de y para 3 amostras: 

import numpy as np
y = np.array([[31.4, 94], [40.5, 109], [25.0, 30]])
print(y)


##### 1.12.4.2. MultiOutputRegressor

    # O suporte à regressão multioutput pode ser adicionado a qualquer regressor com MultiOutputRegressor. Esta estratégia consiste em ajustar um regressor por alvo. Uma vez que cada alvo é representado por exatamente um regressor, é possível obter conhecimento sobre o alvo inspecionando seu regressor correspondente. Como MultiOutputRegressor se ajusta a um regressor por alvo, ele não pode tirar vantagem das correlações entre os alvos.

    # Abaixo está um exemplo de regressão de múltiplas saídas: 

from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
X, y = make_regression(n_samples=10, n_targets=3, random_state=1)
MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X, y).predict(X)
 
##### 1.12.4.3. RegressorChain 

    # Cadeias de regressores (consulte RegressorChain) são análogas a ClassifierChain como uma forma de combinar várias regressões em um único modelo de múltiplos alvos que é capaz de explorar correlações entre alvos. 