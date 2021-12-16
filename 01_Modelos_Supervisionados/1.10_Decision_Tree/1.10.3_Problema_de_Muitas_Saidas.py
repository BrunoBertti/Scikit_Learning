########## 1.10.3. Problemas de saída múltipla  ##########

    # Um problema de múltiplas saídas é um problema de aprendizado supervisionado com várias saídas para prever, ou seja, quando Y é um array 2d de forma (n_amostras, n_ saídas).

    # Quando não há correlação entre as saídas, uma maneira muito simples de resolver esse tipo de problema é construir n modelos independentes, ou seja, um para cada saída e, em seguida, usar esses modelos para prever independentemente cada uma das n saídas. No entanto, como é provável que os valores de saída relacionados à mesma entrada sejam correlacionados, uma maneira geralmente melhor é construir um único modelo capaz de prever simultaneamente todas as n saídas. Em primeiro lugar, requer menor tempo de treinamento, uma vez que apenas um único estimador é construído. Em segundo lugar, a precisão da generalização do estimador resultante pode frequentemente ser aumentada.

    # Com relação às árvores de decisão, essa estratégia pode ser usada prontamente para dar suporte a problemas de múltiplos produtos. Isso requer as seguintes alterações: 

        # Armazene n valores de saída nas folhas, em vez de 1;
        # Use critérios de divisão que calculam a redução média em todas as n saídas. 

    # Este módulo oferece suporte para problemas de múltiplas saídas implementando esta estratégia em DecisionTreeClassifier e DecisionTreeRegressor. Se uma árvore de decisão for ajustada em uma matriz de saída Y de forma (n_amostras, n_saídas), o estimador resultante irá: 
        
        # Valores de saída n_output na previsão;
        # Produz uma lista de matrizes de n_output de probabilidades de classe em Predict_proba. 

    # O uso de árvores de multi-outputs para regressão é demonstrado em Multi-output Decision Tree Regression. Neste exemplo, a entrada X é um único valor real e as saídas Y são o seno e o cosseno de X. 

        # https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression_multioutput.html

    # O uso de árvores de multi-outputs para classificação é demonstrado em Complementação de face com estimadores de multi-outputs. Neste exemplo, as entradas X são os pixels da metade superior das faces e as saídas Y são os pixels da metade inferior dessas faces. 

        # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_multioutput_face_completion.html



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression_multioutput.html#sphx-glr-auto-examples-tree-plot-tree-regression-multioutput-py

    ##https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_multioutput_face_completion.html#sphx-glr-auto-examples-miscellaneous-plot-multioutput-face-completion-py



    ## Referências:
    ##M. Dumont et al, Fast multi-class image annotation with random subwindows and multiple output randomized trees, International Conference on Computer Vision Theory and Applications 2009 (http://www.montefiore.ulg.ac.be/services/stochastic/pubs/2009/DMWG09/dumont-visapp09-shortpaper.pdf)