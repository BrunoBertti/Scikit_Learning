########## 2.5.7. Fatoração de matriz não negativa (NMF ou NNMF) ##########

##### 2.5.7.1. NMF com a norma Frobenius

    # NMF 1 é uma abordagem alternativa para decomposição que assume que os dados e os componentes não são negativos. O NMF pode ser conectado em vez do PCA ou suas variantes, nos casos em que a matriz de dados não contém valores negativos. Ele encontra uma decomposição das amostras X em duas matrizes W e H de elementos não negativos, otimizando a distância d entre X e o produto da matriz WH. A função de distância mais amplamente usada é a norma de Frobenius ao quadrado, que é uma extensão óbvia da norma euclidiana para matrizes:


        #d _ {\ mathrm {Fro}} (X, Y) = \ frac {1} {2} || X - Y || _ {\ mathrm {Fro}} ^ 2 = \ frac {1} {2} \ sum_ {i, j} (X_ {ij} - {Y} _ {ij}) ^ 2


    # Ao contrário do PCA, a representação de um vetor é obtida de forma aditiva, sobrepondo os componentes, sem subtrair. Esses modelos aditivos são eficientes para representar imagens e texto.

    # Foi observado em [Hoyer, 2004] 2 que, quando cuidadosamente restrito, o NMF pode produzir uma representação baseada em partes do conjunto de dados, resultando em modelos interpretáveis. O exemplo a seguir exibe 16 componentes esparsos encontrados por NMF a partir das imagens no conjunto de dados Olivetti faces, em comparação com as eigenfaces PCA. 

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html


    # O atributo init determina o método de inicialização aplicado, o que tem um grande impacto no desempenho do método. O NMF implementa o método Nonnegative Double Singular Value Decomposition. O NNDSVD 4 é baseado em dois processos SVD, um aproximando a matriz de dados, o outro aproximando seções positivas dos fatores SVD parciais resultantes utilizando uma propriedade algébrica de matrizes de classificação unitária. O algoritmo NNDSVD básico é mais adequado para fatoração esparsa. Suas variantes NNDSVDa (em que todos os zeros são definidos iguais à média de todos os elementos dos dados) e NNDSVDar (em que os zeros são definidos para perturbações aleatórias menores do que a média dos dados divididos por 100) são recomendados no denso caso.

    # Observe que o solucionador de Atualização Multiplicativa ('mu') não pode atualizar os zeros presentes na inicialização, por isso leva a resultados piores quando usado em conjunto com o algoritmo NNDSVD básico que introduz muitos zeros; neste caso, NNDSVDa ou NNDSVDar deve ser preferido.

    # O NMF também pode ser inicializado com matrizes não negativas aleatórias corretamente dimensionadas definindo init = "random". Uma semente inteira ou um RandomState também pode ser passado para random_state para controlar a reprodutibilidade. 

    
    # No NMF, as anteriores L1 e L2 podem ser adicionadas à função de perda para regularizar o modelo. O L2 prior usa a norma Frobenius, enquanto o L1 prior usa uma norma elementwise L1. Como no ElasticNet, controlamos a combinação de L1 e L2 com o parâmetro l1_ratio (\ rho), e a intensidade da regularização com os parâmetros alpha_W e alpha_H (\ alpha_W e \ alpha_H). As anteriores são escaladas pelo número de amostras (n \ _samples) para H e o número de recursos (n \ _features) para W para manter seu impacto equilibrado em relação ao outro e ao termo de ajuste de dados o mais independente possível do tamanho do conjunto de treinamento. Então, os termos anteriores são: 


        # (\alpha_W \rho ||W||_1 + \frac{\alpha_W(1-\rho)}{2} ||W||_{\mathrm{Fro}} ^ 2) * n\_features
        # + (\alpha_H \rho ||H||_1 + \frac{\alpha_H(1-\rho)}{2} ||H||_{\mathrm{Fro}} ^ 2) * n\_samples


    # e a função objetivo regularizada é: 

        # d_{\mathrm{Fro}}(X, WH)
        # + (\alpha_W \rho ||W||_1 + \frac{\alpha_W(1-\rho)}{2} ||W||_{\mathrm{Fro}} ^ 2) * n\_features
        # + (\alpha_H \rho ||H||_1 + \frac{\alpha_H(1-\rho)}{2} ||H||_{\mathrm{Fro}} ^ 2) * n\_samples



##### 2.5.7.2. NMF com divergência beta 

    # Conforme descrito anteriormente, a função de distância mais amplamente usada é a norma de Frobenius ao quadrado, que é uma extensão óbvia da norma euclidiana para matrizes: 

        # d_{\mathrm{Fro}}(X, Y) = \frac{1}{2} ||X - Y||_{Fro}^2 = \frac{1}{2} \sum_{i,j} (X_{ij} - {Y}_{ij})^2

    # Outras funções de distância podem ser usadas em NMF como, por exemplo, a divergência (generalizada) de Kullback-Leibler (KL), também referida como divergência I:

        # d_{KL}(X, Y) = \sum_{i,j} (X_{ij} \log(\frac{X_{ij}}{Y_{ij}}) - X_{ij} + Y_{ij})

    # Ou, a divergência Itakura-Saito (IS):

        # d_{IS}(X, Y) = \sum_{i,j} (\frac{X_{ij}}{Y_{ij}} - \log(\frac{X_{ij}}{Y_{ij}}) - 1)

    # Essas três distâncias são casos especiais da família da divergência beta, com \ beta = 2, 1, 0 respectivamente 6. A divergência beta é definida por: 

        # d_{\beta}(X, Y) = \sum_{i,j} \frac{1}{\beta(\beta - 1)}(X_{ij}^\beta + (\beta-1)Y_{ij}^\beta - \beta X_{ij} Y_{ij}^{\beta - 1})


        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_beta_divergence.html


   # NObserve que esta definição não é válida se \ beta \ in (0; 1), mas pode ser continuamente estendida às definições de d_ {KL} e d_ {IS} respectivamente.

   # NMF implementa dois solvers, usando Coordinate Descent ('cd') 5 e Multiplicative Update ('mu') 6. O solver 'mu' pode otimizar cada beta-divergência, incluindo, é claro, a norma Frobenius (\ beta = 2), a divergência de Kullback-Leibler (generalizada) (\ beta = 1) e a divergência de Itakura-Saito (\ beta = 0). Observe que para \ beta \ in (1; 2), o solucionador 'mu' é significativamente mais rápido do que para outros valores de \ beta. Observe também que com um negativo (ou 0, ou seja, ‘itakura-saito’) \ beta, a matriz de entrada não pode conter valores zero.

   # NO solucionador de 'cd' só pode otimizar a norma Frobenius. Devido à não convexidade subjacente do NMF, os diferentes solucionadores podem convergir para diferentes mínimos, mesmo ao otimizar a mesma função de distância.

   # NNMF é melhor usado com o método fit_transform, que retorna a matriz W. A matriz H é armazenada no modelo ajustado no atributo components_; a transformação do método irá decompor uma nova matriz X_new com base nestes componentes armazenados: 


import numpy as np
X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_
X_new = np.array([[1, 0], [1, 6.1], [1, 0], [1, 4], [3.2, 1], [0, 4]])
W_new = model.transform(X_new)



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py

    ## https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_beta_divergence.html#sphx-glr-auto-examples-decomposition-plot-beta-divergence-py



    ## Referências:

    ## 1 “Learning the parts of objects by non-negative matrix factorization” D. Lee, S. Seung, 1999 (http://www.columbia.edu/~jwp2128/Teaching/E4903/papers/nmf_nature.pdf)

    ## 2 “Non-negative Matrix Factorization with Sparseness Constraints” P. Hoyer, 2004 (http://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf)

    ## 4 “SVD based initialization: A head start for nonnegative matrix factorization” C. Boutsidis, E. Gallopoulos, 2008 (http://scgroup.hpclab.ceid.upatras.gr/faculty/stratis/Papers/HPCLAB020107.pdf)

    ## 5 “Fast local algorithms for large scale nonnegative matrix and tensor factorizations.” A. Cichocki, A. Phan, 2009 (http://www.bsp.brain.riken.jp/publications/2009/Cichocki-Phan-IEICE_col.pdf)

    ## 6(1,2) “Algorithms for nonnegative matrix factorization with the beta-divergence” C. Fevotte, J. Idier, 2011 (https://arxiv.org/pdf/1010.1763.pdf)