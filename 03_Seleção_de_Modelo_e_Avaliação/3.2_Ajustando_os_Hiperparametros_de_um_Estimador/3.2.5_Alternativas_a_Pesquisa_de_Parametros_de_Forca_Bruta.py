########## 3.2.5. Alternativas à pesquisa de parâmetros de força bruta  ##########

##### 3.2.5.1. Validação cruzada específica do modelo 


    # Alguns modelos podem ajustar dados para uma faixa de valores de algum parâmetro quase tão eficientemente quanto ajustar o estimador para um único valor do parâmetro. Esse recurso pode ser aproveitado para realizar uma validação cruzada mais eficiente usada para a seleção do modelo desse parâmetro.

    # O parâmetro mais comum para esta estratégia é o parâmetro que codifica a força do regularizador. Neste caso dizemos que calculamos o caminho de regularização do estimador.

    # Aqui está a lista de tais modelos: 

# linear_model.ElasticNetCV(*[, l1_ratio, ...]) Elastic Net model with iterative fitting along a regularization path.
# linear_model.LarsCV(*[, fit_intercept, ...]) Cross-validated Least Angle Regression model.
# linear_model.LassoCV(*[, eps, n_alphas, ...]) Lasso linear model with iterative fitting along a regularization path.
# linear_model.LassoLarsCV(*[, fit_intercept, ...]) Cross-validated Lasso, using the LARS algorithm.
# linear_model.LogisticRegressionCV(*[, Cs, ...]) Logistic Regression CV (aka logit, MaxEnt) classifier.
# linear_model.MultiTaskElasticNetCV(*[, ...]) Multi-task L1/L2 ElasticNet with built-in cross-validation.
# linear_model.MultiTaskLassoCV(*[, eps, ...]) Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.
# linear_model.OrthogonalMatchingPursuitCV(*) Cross-validated Orthogonal Matching Pursuit model (OMP).
# linear_model.RidgeCV([alphas, ...]) Ridge regression with built-in cross-validation.
# linear_model.RidgeClassifierCV([alphas, ...]) Ridge classifier with built-in cross-validation.



##### 3.2.5.2. Critério de Informação

    # Alguns modelos podem oferecer uma fórmula de forma fechada de teoria da informação da estimativa ótima do parâmetro de regularização calculando um único caminho de regularização (em vez de vários ao usar validação cruzada).

    # Aqui está a lista de modelos que se beneficiam do Akaike Information Criterion (AIC) ou do Bayesian Information Criterion (BIC) para seleção automatizada de modelos: 

# linear_model.LassoLarsIC([criterion, ...]) Modelo de laço ajustado com Lars usando BIC ou AIC para seleção de modelo. 

##### 3.2.5.3. Orçamentos sem limite de bagagem

    # Ao usar métodos ensemble baseados em ensacamento, ou seja, gerando novos conjuntos de treinamento usando amostragem com substituição, parte do conjunto de treinamento permanece sem uso. Para cada classificador no conjunto, uma parte diferente do conjunto de treinamento é omitida.

    # Essa parte omitida pode ser usada para estimar o erro de generalização sem ter que contar com um conjunto de validação separado. Esta estimativa vem “de graça”, pois não são necessários dados adicionais e podem ser usados para a seleção do modelo.

    # Atualmente, isso é implementado nas seguintes classes: 



# ensemble.RandomForestClassifier([...]) A random forest classifier.

# ensemble.RandomForestRegressor([...])A random forest regressor.

# ensemble.ExtraTreesClassifier([...]) An extra-trees classifier.

# ensemble.ExtraTreesRegressor([n_estimators, ...]) An extra-trees regressor.

# ensemble.GradientBoostingClassifier(*[, ...]) Gradient Boosting for classification.

# ensemble.GradientBoostingRegressor(*[, ...]) Gradient Boosting for regression.