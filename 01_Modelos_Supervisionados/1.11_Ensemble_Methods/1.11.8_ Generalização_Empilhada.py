########## 1.11.8 Generalização empilhada  ##########

    # A generalização empilhada é um método para combinar estimadores para reduzir seus vieses [W1992] [HTF]. Mais precisamente, as previsões de cada estimador individual são empilhadas e usadas como entrada para um estimador final para calcular a previsão. Este estimador final é treinado por meio de validação cruzada.

    # O StackingClassifier e o StackingRegressor fornecem essas estratégias que podem ser aplicadas a problemas de classificação e regressão.

    # O parâmetro dos estimadores corresponde à lista dos estimadores que são empilhados em paralelo nos dados de entrada. Deve ser fornecido como uma lista de nomes e estimadores: 

from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
estimators = [('ridge', RidgeCV()),
              ('lasso', LassoCV(random_state=42)),
              ('knr', KNeighborsRegressor(n_neighbors=20,
                                          metric='euclidean'))]

    # O final_estimator usará as previsões dos estimadores como entrada. Ele precisa ser um classificador ou regressor ao usar StackingClassifier ou StackingRegressor, respectivamente: 


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
final_estimator = GradientBoostingRegressor(
        n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1,
    random_state=42)
reg = StackingRegressor(
        estimators=estimators, final_estimator=final_estimator)

    #   Para treinar os estimadores e o final_estimator, o método de ajuste precisa ser chamado nos dados de treinamento: 

from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42)
reg.fit(X_train, y_train)


    # Durante o treinamento, os estimadores são ajustados em todos os dados de treinamento X_train. Eles serão usados ao chamar Predict ou Predict_proba. Para generalizar e evitar o sobreajuste, o final_estimator é treinado em out-samples usando sklearn.model_selection.cross_val_predict internamente.

    # Para StackingClassifier, observe que a saída dos estimadores é controlada pelo parâmetro stack_method e é chamada por cada estimador. Este parâmetro é uma string, sendo nomes de métodos do estimador, ou 'auto' que identificará automaticamente um método disponível dependendo da disponibilidade, testado na ordem de preferência: Predict_proba, Decision_function e Predict.

    # Um StackingRegressor e StackingClassifier podem ser usados como qualquer outro regressor ou classificador, expondo métodos de previsão, previsão_proba e função de decisão, por exemplo: 

y_pred = reg.predict(X_test)
from sklearn.metrics import r2_score
print('R2 score: {:.2f}'.format(r2_score(y_test, y_pred)))


    # Observe que também é possível obter a saída dos estimadores empilhados usando o método de transformação: 

reg.transform(X_test[:5])

    # Na prática, um preditor de empilhamento prevê tão bom quanto o melhor preditor da camada de base e, às vezes, até supera-o ao combinar as diferentes intensidades desses preditores. No entanto, treinar um preditor de empilhamento é caro do ponto de vista computacional. 

    # OBS: Para StackingClassifier, ao usar stack_method _ = 'predict_proba', a primeira coluna é descartada quando o problema é um problema de classificação binária. Na verdade, ambas as colunas de probabilidade previstas por cada estimador são perfeitamente colineares. 



    # OBS: Múltiplas camadas de empilhamento podem ser obtidas atribuindo final_estimator a um StackingClassifier ou StackingRegressor: 

from sklearn.ensemble import RandomForestRegressor
final_layer_rfr = RandomForestRegressor(
    n_estimators=10, max_features=1, max_leaf_nodes=5,random_state=42)
final_layer_gbr = GradientBoostingRegressor(
    n_estimators=10, max_features=1, max_leaf_nodes=5,random_state=42)
final_layer = StackingRegressor(
    estimators=[('rf', final_layer_rfr),
                ('gbrt', final_layer_gbr)],
    final_estimator=RidgeCV()
    )
multi_layer_regressor = StackingRegressor(
    estimators=[('ridge', RidgeCV()),
                ('lasso', LassoCV(random_state=42)),
                ('knr', KNeighborsRegressor(n_neighbors=20,
                                            metric='euclidean'))],
    final_estimator=final_layer
)
multi_layer_regressor.fit(X_train, y_train)

print('R2 score: {:.2f}'
      .format(multi_layer_regressor.score(X_test, y_test)))


    ## Referências:

    ## W1992 Wolpert, David H. “Stacked generalization.” Neural networks 5.2 (1992): 241-259. (https://scikit-learn.org/stable/modules/ensemble.html#id33)