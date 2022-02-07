########## 6.7.3. Kernel Aditivo de Chi Quadrado ##########


    # O kernel aditivo qui quadrado é um kernel em histogramas, frequentemente usado em visão computacional.

    # O kernel qui quadrado aditivo usado aqui é dado por 

        # k(x, y) = \sum_i \frac{2x_iy_i}{x_i+y_i}



    # Isso não é exatamente o mesmo que sklearn.metrics.additive_chi2_kernel. Os autores de [VZ2010] preferem a versão acima, pois é sempre positiva definida. Como o kernel é aditivo, é possível tratar todos os componentes x_i separadamente para incorporação. Isso torna possível amostrar a transformada de Fourier em intervalos regulares, em vez de aproximar usando amostragem de Monte Carlo.

    # A classe AdditiveChi2Sampler implementa essa amostragem determinística inteligente de componentes. Cada componente é amostrado n vezes, resultando em 2n+1 dimensões por dimensão de entrada (o múltiplo de duas derivações da parte real e complexa da transformada de Fourier). Na literatura, n é geralmente escolhido como 1 ou 2, transformando o conjunto de dados para tamanho n_samples * 5 * n_features (no caso de n=2).

    # O mapa de características aproximado fornecido pelo AdditiveChi2Sampler pode ser combinado com o mapa de características aproximado fornecido pelo RBFSampler para produzir um mapa de características aproximado para o kernel qui-quadrado exponenciado. Consulte [VZ2010] para obter detalhes e [VVZ2010] para combinação com o RBFSampler. 