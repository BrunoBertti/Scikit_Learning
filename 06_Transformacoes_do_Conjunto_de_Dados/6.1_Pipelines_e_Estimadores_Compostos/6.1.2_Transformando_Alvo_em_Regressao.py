########## 6.1.2. Transformando alvo em regressão ##########


    # TransformedTargetRegressor transforma os destinos y antes de ajustar um modelo de regressão. As previsões são mapeadas de volta ao espaço original por meio de uma transformação inversa. Toma como argumento o regressor que será usado para previsão e o transformador que será aplicado à variável alvo: 

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X, y = fetch_california_housing(return_X_y=True)
X, y = X[:2000, :], y[:2000]  # selecione um subconjunto de dados 
transformer = QuantileTransformer(output_distribution='normal')
regressor = LinearRegression()
regr = TransformedTargetRegressor(regressor=regressor,
                                  transformer=transformer)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regr.fit(X_train, y_train)
print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))
raw_target_regr = LinearRegression().fit(X_train, y_train)
print('R2 score: {0:.2f}'.format(raw_target_regr.score(X_test, y_test)))


    # Para transformações simples, ao invés de um objeto Transformer, pode-se passar um par de funções, definindo a transformação e seu mapeamento inverso: 

def func(x):
    return np.log(x)
def inverse_func(x):
    return np.exp(x)

    # Posteriormente, o objeto é criado como: 

regr = TransformedTargetRegressor(regressor=regressor,
                                  func=func,
                                  inverse_func=inverse_func)
regr.fit(X_train, y_train)
TransformedTargetRegressor(...)
print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))


    # Por padrão, as funções fornecidas são verificadas em cada ajuste para serem o inverso uma da outra. No entanto, é possível contornar essa verificação definindo check_inverse como False: 

def inverse_func(x):
    return x
regr = TransformedTargetRegressor(regressor=regressor,
                                  func=func,
                                  inverse_func=inverse_func,
                                  check_inverse=False)
regr.fit(X_train, y_train)
TransformedTargetRegressor(...)
print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))


    # Nota: A transformação pode ser acionada configurando transformador ou o par de funções func e inverse_func. No entanto, definir ambas as opções gerará um erro. 



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py