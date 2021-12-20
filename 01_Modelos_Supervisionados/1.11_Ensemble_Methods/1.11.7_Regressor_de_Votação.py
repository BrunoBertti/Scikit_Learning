########## 1.11. Ensemble Methods -  Métodos de Conjunto ##########

    # A ideia por trás do VotingRegressor é combinar regressores de aprendizado de máquina conceitualmente diferentes e retornar os valores médios previstos. Esse regressor pode ser útil para um conjunto de modelos de desempenho igualmente bom, a fim de equilibrar suas fraquezas individuais.



##### 1.11.7.1. Uso 

    # O exemplo a seguir mostra como ajustar o VotingRegressor: 

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

# Carregando alguns dados de exemplo
X, y = load_diabetes(return_X_y=True)

# Classificadores de treinamento 
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
ereg = ereg.fit(X, y)


        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_regressor.html


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_regressor.html#sphx-glr-auto-examples-ensemble-plot-voting-regressor-py