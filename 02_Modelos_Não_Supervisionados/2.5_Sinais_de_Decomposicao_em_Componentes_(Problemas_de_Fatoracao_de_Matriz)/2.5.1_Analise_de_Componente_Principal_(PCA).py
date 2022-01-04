########## 2.5.1. Análise de componente principal (PCA) ##########


##### 2.5.1.1. PCA exata e interpretação probabilística

    # O PCA é usado para decompor um conjunto de dados multivariado em um conjunto de componentes ortogonais sucessivos que explicam uma quantidade máxima da variação. No scikit-learn, o PCA é implementado como um objeto transformador que aprende n componentes em seu método de ajuste e pode ser usado em novos dados para projetá-los nesses componentes.

    # O PCA centraliza, mas não dimensiona os dados de entrada para cada recurso antes de aplicar o SVD. O parâmetro opcional whiten = True torna possível projetar os dados no espaço singular enquanto dimensiona cada componente para a variação da unidade. Isso geralmente é útil se os modelos a jusante fazem suposições fortes sobre a isotropia do sinal: este é, por exemplo, o caso para Support Vector Machines com o kernel RBF e o algoritmo de agrupamento K-Means.

    # Abaixo está um exemplo do conjunto de dados da íris, que é composto por 4 recursos, projetados nas 2 dimensões que explicam a maior parte da variação: 


        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html

    # O objeto PCA também fornece uma interpretação probabilística do PCA que pode dar uma probabilidade de dados com base na quantidade de variação que ele explica. Como tal, ele implementa um método de pontuação que pode ser usado na validação cruzada: 

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-fa-model-selection-py

##### 2.5.1.2. PCA incremental

    # O objeto PCA também fornece uma interpretação probabilística do PCA que pode dar uma probabilidade de dados com base na quantidade de variação que ele explica. Como tal, ele implementa um método de pontuação que pode ser usado na validação cruzada:

        # Usando seu método partial_fit em blocos de dados buscados sequencialmente no disco rígido local ou em um banco de dados de rede.

        # Chamando seu método de ajuste em uma matriz esparsa ou um arquivo mapeado de memória usando numpy.memmap.

    # IncrementalPCA armazena apenas estimativas de variâncias de componente e ruído, para atualizar explicado_variance_ratio_ incrementalmente. É por isso que o uso da memória depende do número de amostras por lote, em vez do número de amostras a serem processadas no conjunto de dados.

    # Como no PCA, o IncrementalPCA centraliza, mas não dimensiona os dados de entrada para cada recurso antes de aplicar o SVD. 

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html#sphx-glr-auto-examples-decomposition-plot-incremental-pca-py



##### 2.5.1.3. PCA usando SVD randomizado


    # Muitas vezes é interessante projetar dados para um espaço de dimensão inferior que preserva a maior parte da variância, eliminando o vetor singular de componentes associados a valores singulares inferiores.

    # Por exemplo, se trabalharmos com imagens em nível de cinza de 64x64 pixels para reconhecimento de rosto, a dimensionalidade dos dados é 4096 e é lento para treinar uma máquina de vetor de suporte RBF com dados tão amplos. Além disso, sabemos que a dimensionalidade intrínseca dos dados é muito inferior a 4096, uma vez que todas as imagens de rostos humanos se parecem um pouco. As amostras estão em um coletor de dimensão muito inferior (digamos cerca de 200, por exemplo). O algoritmo PCA pode ser usado para transformar linearmente os dados enquanto reduz a dimensionalidade e preserva a maior parte da variância explicada ao mesmo tempo.

    # A classe PCA usada com o parâmetro opcional svd_solver = 'randomized' é muito útil nesse caso: uma vez que vamos eliminar a maioria dos vetores singulares, é muito mais eficiente limitar o cálculo a uma estimativa aproximada dos vetores singulares. mantenha para realmente executar a transformação. 

    # Por exemplo, o seguinte mostra 16 retratos de amostra (centralizados em torno de 0,0) do conjunto de dados da Olivetti. No lado direito estão os primeiros 16 vetores singulares remodelados como retratos. Uma vez que exigimos apenas os 16 principais vetores singulares de um conjunto de dados com tamanho n_ {amostras} = 400 e n_ {recursos} = 64 \ vezes 64 = 4096, o tempo de cálculo é inferior a 1s: 

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html


    # Se observarmos n _ {\ max} = \ max (n _ {\ mathrm {amostras}}, n _ {\ mathrm {recursos}}) e n _ {\ min} = \ min (n _ {\ mathrm {amostras}}, n_ {\ mathrm {features}}), a complexidade de tempo do PCA randomizado é O (n _ {\ max} ^ 2 \ cdot n _ {\ mathrm {componentes}}) em vez de O (n _ {\ max} ^ 2 \ cdot n _ {\ min}) para o método exato implementado no PCA.

    # A pegada de memória do PCA aleatório também é proporcional a 2 \ cdot n _ {\ max} \ cdot n _ {\ mathrm {componentes}} em vez de n _ {\ max} \ cdot n _ {\ min} para o método exato.


    # Nota: a implementação de inverse_transform no PCA com svd_solver = 'randomized' não é a transformação inversa exata da transformação mesmo quando whiten = False (padrão). 


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py



    ## Referências:

    ## Algorithm 4.3 in “Finding structure with randomness: Stochastic algorithms for constructing approximate matrix decompositions” Halko, et al., 2009 (https://arxiv.org/abs/0909.4061)

    ## “An implementation of a randomized algorithm for principal component analysis” A. Szlam et al. 2014 (https://arxiv.org/pdf/1412.3510.pdf)

##### 2.5.1.4. Análise de componentes principais esparsos (SparsePCA e MiniBatchSparsePCA) 


    # SparsePCA é uma variante do PCA, com o objetivo de extrair o conjunto de componentes esparsos que melhor reconstroem os dados.

    # Mini-batch sparse PCA (MiniBatchSparsePCA) é uma variante do SparsePCA que é mais rápido, mas menos preciso. O aumento da velocidade é alcançado pela iteração em pequenos pedaços do conjunto de recursos, para um determinado número de iterações.

    # A análise de componentes principais (PCA) tem a desvantagem de que os componentes extraídos por este método têm expressões exclusivamente densas, ou seja, eles têm coeficientes diferentes de zero quando expressos como combinações lineares das variáveis originais. Isso pode dificultar a interpretação. Em muitos casos, os componentes reais subjacentes podem ser mais naturalmente imaginados como vetores esparsos; por exemplo, no reconhecimento de rosto, os componentes podem ser mapeados naturalmente para partes de rostos.

    # Os componentes principais esparsos produzem uma representação mais parcimoniosa e interpretável, enfatizando claramente quais das características originais contribuem para as diferenças entre as amostras. 

    # O exemplo a seguir ilustra 16 componentes extraídos usando PCA esparso do conjunto de dados de faces da Olivetti. Pode-se ver como o termo de regularização induz muitos zeros. Além disso, a estrutura natural dos dados faz com que os coeficientes diferentes de zero sejam verticalmente adjacentes. O modelo não impõe isso matematicamente: cada componente é um vetor h \ in \ mathbf {R} ^ {4096}, e não há noção de adjacência vertical, exceto durante a visualização amigável como imagens de 64x64 pixels. O fato de os componentes mostrados abaixo parecerem locais é o efeito da estrutura inerente dos dados, o que faz com que tais padrões locais minimizem o erro de reconstrução. Existem normas indutoras de esparsidade que levam em consideração a adjacência e diferentes tipos de estrutura; veja [Jen09] para uma revisão de tais métodos. Para obter mais detalhes sobre como usar o Sparse PCA, consulte a seção Exemplos abaixo. 

        # https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html

    # Observe que existem muitas formulações diferentes para o problema de PCA esparso. O implementado aqui é baseado em [Mrl09]. O problema de otimização resolvido é um problema de PCA (aprendizagem de dicionário) com uma penalidade de \ ell_1 nos componentes:

        # \ begin {split} (U ^ *, V ^ *) = \ underset {U, V} {\ operatorname {arg \, min \,}} & \ frac {1}  {2}
        #              || X-UV || _ {\ text {Fro}} ^ 2+ \ alpha || V || _ {1,1} \\
        #              \ text {assunto para} & || U_k || _2 <= 1 \ text {para todos}
        #              0 \ leq k <n_ {componentes} \ end {divisão}



    # ||. || _ {\ text {Fro}} representa a norma de Frobenius e ||. || _ {1,1} representa a norma da matriz de entrada, que é a soma dos valores absolutos de todas as entradas na matriz. A norma de matriz indutora de esparsidade ||. || _ {1,1} também evita o aprendizado de componentes de ruído quando poucas amostras de treinamento estão disponíveis. O grau de penalização (e, portanto, esparsidade) pode ser ajustado por meio do hiperparâmetro alfa. Valores pequenos levam a uma fatoração suavemente regularizada, enquanto valores maiores reduzem muitos coeficientes a zero. 



    # Nota: Embora dentro do espírito de um algoritmo online, a classe MiniBatchSparsePCA não implementa partial_fit porque o algoritmo está online ao longo da direção dos recursos, não da direção das amostras. 


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html#sphx-glr-auto-examples-decomposition-plot-faces-decomposition-py




    ## Referências:

    ## Mrl09 “Online Dictionary Learning for Sparse Coding” J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009 (https://www.di.ens.fr/sierra/pdfs/icml09.pdf)

    ## Jen09 “Structured Sparse Principal Component Analysis” R. Jenatton, G. Obozinski, F. Bach, 2009 (https://www.di.ens.fr/~fbach/sspca_AISTATS2010.pdf)