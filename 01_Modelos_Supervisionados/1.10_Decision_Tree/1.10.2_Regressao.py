########## 1.10.2 Regressão ##########

        # https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
    

    # Árvores de decisão também podem ser aplicadas a problemas de regressão, usando a classe DecisionTreeRegressor.

    # Como na configuração de classificação, o método de ajuste tomará como argumento as matrizes X e y, apenas que, neste caso, espera-se que y tenha valores de ponto flutuante em vez de valores inteiros: 


from sklearn import tree
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])

    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py