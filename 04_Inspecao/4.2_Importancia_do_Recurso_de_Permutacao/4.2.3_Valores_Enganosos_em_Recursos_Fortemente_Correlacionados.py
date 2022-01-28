########## 4.2.3. Valores enganosos em recursos fortemente correlacionados  ##########

    # Quando dois recursos são correlacionados e um dos recursos é permutado, o modelo ainda terá acesso ao recurso por meio de seu recurso correlacionado. Isso resultará em um valor de importância menor para ambos os recursos, onde eles podem realmente ser importantes.

    # Uma maneira de lidar com isso é agrupar recursos correlacionados e manter apenas um recurso de cada cluster. Essa estratégia é explorada no exemplo a seguir: Importância da permutação com recursos multicolineares ou correlacionados. 




    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py

    ## https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py





    ## Referências:

    ## L. Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001. (https://doi.org/10.1023/A:1010933404324)