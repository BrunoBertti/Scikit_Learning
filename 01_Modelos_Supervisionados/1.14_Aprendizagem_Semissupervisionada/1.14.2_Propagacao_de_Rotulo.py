########## 1.14.2. Propagação de rótulo  ##########

    # A propagação de rótulos denota algumas variações de algoritmos de inferência de grafos semissupervisionados. 

    # Alguns recursos disponíveis neste modelo: 

        # Usado para tarefas de classificação

        # Métodos de kernel para projetar dados em espaços dimensionais alternativos 

    # O scikit-learn fornece dois modelos de propagação de rótulo: LabelPropagation e LabelSpreading. Ambos funcionam construindo um gráfico de similaridade sobre todos os itens no conjunto de dados de entrada. 

        # https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_structure.html


    # LabelPropagation e LabelSpreading diferem nas modificações na matriz de similaridade que representa o gráfico e no efeito de fixação nas distribuições de rótulos. A fixação permite que o algoritmo altere o peso dos dados rotulados do solo verdadeiro em algum grau. O algoritmo LabelPropagation executa a fixação rígida de rótulos de entrada, o que significa \ alpha = 0. Esse fator de fixação pode ser relaxado, para dizer \ alpha = 0,2, o que significa que sempre reteremos 80% de nossa distribuição de rótulo original, mas o algoritmo consegue mudar sua confiança da distribuição em 20%. 

    # LabelPropagation usa a matriz de similaridade bruta construída a partir dos dados sem modificações. Em contraste, LabelSpreading minimiza uma função de perda que tem propriedades de regularização, como tal, é frequentemente mais robusta ao ruído. O algoritmo itera em uma versão modificada do gráfico original e normaliza os pesos das arestas calculando a matriz Laplaciana do gráfico normalizado. Este procedimento também é usado no agrupamento espectral.

    # Os modelos de propagação de rótulo têm dois métodos de kernel integrados. A escolha do kernel afeta a escalabilidade e o desempenho dos algoritmos. Os seguintes estão disponíveis: 

        # rbf (\ exp (- \ gamma | x-y | ^ 2), \ gamma> 0). \ gamma é especificado pela palavra-chave gamma.


        # knn (1 [x '\ in kNN (x)]). k é especificado pela palavra-chave n_neighs. 

    # O kernel RBF produzirá um gráfico totalmente conectado que é representado na memória por uma matriz densa. Esta matriz pode ser muito grande e combinada com o custo de realizar um cálculo de multiplicação de matriz completa para cada iteração do algoritmo pode levar a tempos de execução proibitivamente longos. Por outro lado, o kernel KNN produzirá uma matriz esparsa muito mais amigável à memória, que pode reduzir drasticamente os tempos de execução. 



    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_semi_supervised_versus_svm_iris.html#sphx-glr-auto-examples-semi-supervised-plot-semi-supervised-versus-svm-iris-py

    ## https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_structure.html#sphx-glr-auto-examples-semi-supervised-plot-label-propagation-structure-py

    ## https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_digits.html#sphx-glr-auto-examples-semi-supervised-plot-label-propagation-digits-py

    ## https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_digits_active_learning.html#sphx-glr-auto-examples-semi-supervised-plot-label-propagation-digits-active-learning-py





    ## Referências:

    ## Yoshua Bengio, Olivier Delalleau, Nicolas Le Roux. In Semi-Supervised Learning (2006), pp. 193-216

    ## Olivier Delalleau, Yoshua Bengio, Nicolas Le Roux. Efficient Non-Parametric Function Induction in Semi-Supervised Learning. AISTAT 2005 https://research.microsoft.com/en-us/people/nicolasl/efficient_ssl.pdf (https://research.microsoft.com/en-us/people/nicolasl/efficient_ssl.pdf)