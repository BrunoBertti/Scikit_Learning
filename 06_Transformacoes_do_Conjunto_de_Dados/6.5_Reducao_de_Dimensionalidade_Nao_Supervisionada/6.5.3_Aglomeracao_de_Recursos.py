########## 6.5.3. Aglomeração de recursos  ##########


    # cluster.FeatureAgglomration aplica agrupamento hierárquico para agrupar recursos que se comportam de forma semelhante. 



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/cluster/plot_feature_agglomeration_vs_univariate_selection.html#sphx-glr-auto-examples-cluster-plot-feature-agglomeration-vs-univariate-selection-py


    ## https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_agglomeration.html#sphx-glr-auto-examples-cluster-plot-digits-agglomeration-py






    # Dimensionamento de recursos

    # Observe que, se os recursos tiverem propriedades estatísticas ou de dimensionamento muito diferentes, o cluster.FeatureAgglomration pode não conseguir capturar os links entre os recursos relacionados. O uso de um preprocessing.StandardScaler pode ser útil nessas configurações. 