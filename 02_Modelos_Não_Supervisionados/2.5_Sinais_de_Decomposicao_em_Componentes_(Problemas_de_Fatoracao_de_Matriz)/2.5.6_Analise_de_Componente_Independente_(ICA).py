########## 2.5.6. Análise de componente independente (ICA) ##########

    # A análise de componentes independentes separa um sinal multivariado em subcomponentes aditivos que são maximamente independentes. Ele é implementado no scikit-learn usando o algoritmo Fast ICA. Normalmente, o ICA não é usado para reduzir a dimensionalidade, mas para separar sinais sobrepostos. Uma vez que o modelo ICA não inclui um termo de ruído, para que o modelo esteja correto, o clareamento deve ser aplicado. Isso pode ser feito internamente usando o argumento whiten ou manualmente usando uma das variantes PCA.

    # É classicamente usado para separar sinais mistos (um problema conhecido como separação cega da fonte), como no exemplo abaixo: 

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html

    
    # O ICA também pode ser usado como outra decomposição não linear que encontra componentes com alguma dispersão: 

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html




    ## Exemplos:


    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_vs_pca.html#sphx-glr-auto-examples-decomposition-plot-ica-vs-pca-py

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py