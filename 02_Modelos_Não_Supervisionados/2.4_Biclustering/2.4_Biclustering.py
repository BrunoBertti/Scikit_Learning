########## 2.4. Biclustering ##########

    # O biclustering pode ser executado com o módulo sklearn.cluster.bicluster. Algoritmos de biclustering agrupam simultaneamente linhas e colunas de uma matriz de dados. Esses clusters de linhas e colunas são conhecidos como biclusters. Cada um determina uma submatriz da matriz de dados original com algumas propriedades desejadas.

    # Por exemplo, dada uma matriz de forma (10, 10), um possível bicluster com três linhas e duas colunas induz uma submatriz de forma (3, 2):


import numpy as np
data = np.arange(100).reshape(10, 10)
rows = np.array([0, 2, 3])[:, np.newaxis]
columns = np.array([1, 2])
data[rows, columns]



    # Para fins de visualização, dado um bicluster, as linhas e colunas da matriz de dados podem ser reorganizadas para tornar o bicluster contíguo.

    # Os algoritmos diferem na forma como definem os biclusters. Alguns dos tipos comuns incluem: 

        # valores constantes, linhas constantes ou colunas constantes

        # valores excepcionalmente altos ou baixos

        # submatrizes com baixa variância

        # linhas ou colunas correlacionadas 


    # Os algoritmos também diferem em como as linhas e colunas podem ser atribuídas aos biclusters, o que leva a diferentes estruturas de biclusters. As estruturas do bloco diagonal ou quadriculado ocorrem quando as linhas e colunas são divididas em partições.

    # Se cada linha e cada coluna pertencerem a exatamente um bicluster, a reorganização das linhas e colunas da matriz de dados revelará os biclusters na diagonal. Aqui está um exemplo dessa estrutura em que os biclusters têm valores médios mais altos do que as outras linhas e colunas: 

        # https://scikit-learn.org/stable/auto_examples/bicluster/images/sphx_glr_plot_spectral_coclustering_003.png

    
    # No caso do tabuleiro de xadrez, cada linha pertence a todos os clusters de coluna e cada coluna pertence a todos os clusters de linha. Aqui está um exemplo dessa estrutura em que a variação dos valores dentro de cada bicluster é pequena: 

        # https://scikit-learn.org/stable/auto_examples/bicluster/images/sphx_glr_plot_spectral_biclustering_003.png


    # Depois de ajustar um modelo, a associação do cluster de linha e coluna pode ser encontrada nos atributos rows_ e colunas_. row_ [i] é um vetor binário com entradas diferentes de zero correspondendo a linhas que pertencem ao bicluster i. Da mesma forma, as colunas_ [i] indica quais colunas pertencem ao bicluster i.

    # Alguns modelos também possuem atributos row_labels_ e column_labels_. Esses modelos particionam as linhas e colunas, como na diagonal do bloco e nas estruturas quadriculadas do bicluster.

    # Nota: Biclustering tem muitos outros nomes em diferentes campos, incluindo co-clustering, clustering de dois modos, clustering bidirecional, clustering de bloco, clustering bidirecional acoplado, etc. Os nomes de alguns algoritmos, como o algoritmo Spectral Co-Clustering , refletem esses nomes alternativos. 