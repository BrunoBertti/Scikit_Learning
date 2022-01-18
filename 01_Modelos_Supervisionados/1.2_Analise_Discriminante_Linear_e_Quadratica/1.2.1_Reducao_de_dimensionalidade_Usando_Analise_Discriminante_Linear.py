########## 1.2.1. Redução de dimensionalidade usando Análise Discriminante Linear ##########

    # LinearDiscriminantAnalysis pode ser usado para realizar a redução de dimensionalidade supervisionada, projetando os dados de entrada para um subespaço linear que consiste nas direções que maximizam a separação entre classes (em um sentido preciso discutido na seção de matemática abaixo). A dimensão da saída é necessariamente menor que o número de classes, portanto, em geral, isso é uma redução de dimensionalidade bastante forte e só faz sentido em uma configuração multiclasse.

    # Isso é implementado no método de transformação. A dimensionalidade desejada pode ser definida usando o parâmetro n_components. Este parâmetro não tem influência nos métodos de ajuste e previsão. 


    ## Exemplos:

    ## Comparison of LDA and PCA 2D projection of Iris dataset: Comparison of LDA and PCA for dimensionality reduction of the Iris dataset (https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py)