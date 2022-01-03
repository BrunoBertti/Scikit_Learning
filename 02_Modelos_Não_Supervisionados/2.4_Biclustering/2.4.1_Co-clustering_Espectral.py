########## 2.4.1. Co-clustering espectral ##########

    # O algoritmo SpectralCoclustering encontra biclusters com valores mais altos do que aqueles nas outras linhas e colunas correspondentes. Cada linha e cada coluna pertence a exatamente um bicluster, portanto, reorganizar as linhas e colunas para tornar as partições contíguas revela estes altos valores ao longo da diagonal:

    # Nota: O algoritmo trata a matriz de dados de entrada como um gráfico bipartido: as linhas e colunas da matriz correspondem aos dois conjuntos de vértices e cada entrada corresponde a uma aresta entre uma linha e uma coluna. O algoritmo aproxima o corte normalizado deste gráfico para encontrar subgráficos pesados. 


##### 2.4.1.1. Formulação matemática 


    # Uma solução aproximada para o corte normalizado ideal pode ser encontrada através da decomposição generalizada de autovalores do Laplaciano do gráfico. Normalmente, isso significaria trabalhar diretamente com a matriz Laplaciana. Se a matriz de dados original A tem forma m \ vezes n, a matriz Laplaciana para o gráfico bipartido correspondente tem forma (m + n) \ vezes (m + n). Porém, neste caso é possível trabalhar diretamente com A, que é menor e mais eficiente. 

    # A matriz de entrada A é pré-processada da seguinte forma:

        # A_n = R ^ {- 1/2} A C ^ {- 1/2}

    # Onde R é a matriz diagonal com entrada i igual a \ sum_ {j} A_ {ij} e C é a matriz diagonal com entrada j igual a \ sum_ {i} A_ {ij.


    # A decomposição do valor singular, A_n = U \ Sigma V ^ \ top, fornece as partições das linhas e colunas de A. Um subconjunto dos vetores singulares à esquerda fornece as partições das linhas e um subconjunto dos vetores singulares à direita fornece as partições das colunas .


    # Os vetores singulares \ ell = \ lceil \ log_2 k \ rceil, a partir do segundo, fornecem as informações de particionamento desejadas. Eles são usados para formar a matriz Z:


        # \ begin {split} Z = \ begin {bmatrix} R ^ {- 1/2} U \\\\
        #                      C ^ {- 1/2} V
        #        \ end {bmatrix} \ end {split}


    # onde as colunas de U são u_2, \ dots, u _ {\ ell +1}, e da mesma forma para V.

    # Em seguida, as linhas de Z são agrupadas usando k-médias. Os primeiros rótulos n_rows fornecem o particionamento de linha e os rótulos n_columns restantes fornecem o particionamento de coluna. 


    ## Exemplos:

    ## A demo of the Spectral Co-Clustering algorithm: A simple example showing how to generate a data matrix with biclusters and apply this method to it. (https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_coclustering.html#sphx-glr-auto-examples-bicluster-plot-spectral-coclustering-py)

    ## Biclustering documents with the Spectral Co-clustering algorithm: An example of finding biclusters in the twenty newsgroup dataset. (https://scikit-learn.org/stable/auto_examples/bicluster/plot_bicluster_newsgroups.html#sphx-glr-auto-examples-bicluster-plot-bicluster-newsgroups-py)



    ## Referências:

    ## http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.140.3011