########## 6.7.5. Aproximação de Kernel Polinomial via Tensor Sketch ##########


    # O kernel polinomial é um tipo popular de função de kernel dado por: 


        # k(x, y) = (\gamma x^\top y +c_0)^d


    # Onde:

        # x, y são os vetores de entrada

        # d é o grau do kernel

    # Intuitivamente, o espaço de recursos do kernel polinomial de grau d consiste em todos os produtos de grau d possíveis entre os recursos de entrada, o que permite que algoritmos de aprendizado usando esse kernel considerem as interações entre os recursos.

    # O método TensorSketch [PP2013], conforme implementado em PolynomialCountSketch, é um método independente de dados de entrada escalável para aproximação de kernel polinomial. Ele é baseado no conceito de Count sketch [WIKICS] [CCF2002] , uma técnica de redução de dimensionalidade semelhante ao hashing de recursos, que usa várias funções de hash independentes. O TensorSketch obtém um Count Sketch do produto externo de dois vetores (ou um vetor consigo mesmo), que pode ser usado como uma aproximação do espaço de recursos do kernel polinomial. Em particular, em vez de calcular explicitamente o produto externo, o TensorSketch calcula o Esboço de Contagem dos vetores e, em seguida, usa a multiplicação polinomial por meio da Transformada Rápida de Fourier para calcular o Esboço de Contagem de seu produto externo.

    # Convenientemente, a fase de treinamento do TensorSketch consiste simplesmente em inicializar algumas variáveis ​​aleatórias. Portanto, é independente dos dados de entrada, ou seja, depende apenas do número de recursos de entrada, mas não dos valores dos dados. Além disso, este método pode transformar amostras em \mathcal{O}(n_{\text{samples}}(n_{\text{features}}+_{\text{components}} \log(n_{\text{components }}))) tempo, onde n_{\text{components}} é a dimensão de saída desejada, determinada por n_components. 





    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/kernel_approximation/plot_scalable_poly_kernels.html#sphx-glr-auto-examples-kernel-approximation-plot-scalable-poly-kernels-py
   
