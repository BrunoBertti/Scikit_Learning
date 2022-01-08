########## 2.8. 2.8.2. Estimativa de densidade do kernel  ##########


    # A estimativa da densidade do kernel no scikit-learn é implementada no estimador KernelDensity, que usa Ball Tree ou KD Tree para consultas eficientes (consulte Vizinhos mais próximos para uma discussão sobre isso). Embora o exemplo acima use um conjunto de dados 1D para simplicidade, a estimativa da densidade do kernel pode ser realizada em qualquer número de dimensões, embora na prática a maldição da dimensionalidade faça com que seu desempenho seja degradado em dimensões altas.

    # Na figura a seguir, 100 pontos são extraídos de uma distribuição bimodal e as estimativas de densidade do kernel são mostradas para três opções de kernel: 

        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html

    # É claro como a forma do kernel afeta a suavidade da distribuição resultante. O estimador de densidade do kernel scikit-learn pode ser usado da seguinte forma:

from sklearn.neighbors import KernelDensity
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
kde.score_samples(X)


    # Aqui usamos kernel = 'gaussian', como visto acima. Matematicamente, um kernel é uma função positiva K (x; h) que é controlada pelo parâmetro de largura de banda h. Dada esta forma de kernel, a estimativa da densidade em um ponto y dentro de um grupo de pontos x_i; i = 1 \ cdots N é dado por:


        # \ rho_K (y) = \ sum_ {i = 1} ^ {N} K (y - x_i; h)


    # A largura de banda aqui atua como um parâmetro de suavização, controlando a compensação entre o viés e a variância no resultado. Uma grande largura de banda leva a uma distribuição de densidade muito suave (ou seja, alta polarização). Uma pequena largura de banda leva a uma distribuição de densidade não uniforme (ou seja, alta variação).

    # KernelDensity implementa vários formulários de kernel comuns, que são mostrados na figura a seguir: 

        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html

    

    # A forma desses kernels é a seguinte: 

        # Gaussian kernel (kernel = 'gaussian')

            # K(x; h) \propto \exp(- \frac{x^2}{2h^2} )

        # Tophat kernel (kernel = 'tophat')

            # K(x; h) \propto 1 x < h

        # Epanechnikov kernel (kernel = 'epanechnikov')

            # K(x; h) \propto 1 - \frac{x^2}{h^2} 

        # Exponential kernel (kernel = 'exponential')

            # K(x; h) \propto \exp(-x/h)

        # Linear kernel (kernel = 'linear')

            # K(x; h) \propto 1 - x/h if x < h

        # Cosine kernel (kernel = 'cosine')

            # K(x; h) \propto \cos(\frac{\pi x}{2h}) if x < h



    # O estimador de densidade do kernel pode ser usado com qualquer uma das métricas de distância válidas (consulte DistanceMetric para uma lista de métricas disponíveis), embora os resultados sejam normalizados corretamente apenas para a métrica Euclidiana. Uma métrica particularmente útil é a distância Haversine, que mede a distância angular entre os pontos de uma esfera. Aqui está um exemplo do uso de uma estimativa de densidade de kernel para uma visualização de dados geoespaciais, neste caso, a distribuição de observações de duas espécies diferentes no continente sul-americano: 

        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_species_kde.html

    # Uma outra aplicação útil da estimativa de densidade do kernel é aprender um modelo gerador não paramétrico de um conjunto de dados para desenhar com eficiência novas amostras deste modelo gerador. Aqui está um exemplo de como usar este processo para criar um novo conjunto de dígitos escritos à mão, usando um kernel gaussiano aprendido em uma projeção PCA dos dados: 


        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html

    # Os “novos” dados consistem em combinações lineares dos dados de entrada, com pesos desenhados probabilisticamente de acordo com o modelo KDE. 




    ## Exemplos:

    ## Simple 1D Kernel Density Estimation: computation of simple kernel density estimates in one dimension. (https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html#sphx-glr-auto-examples-neighbors-plot-kde-1d-py)

    ## Kernel Density Estimation: an example of using Kernel Density estimation to learn a generative model of the hand-written digits data, and drawing new samples from this model. (https://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html#sphx-glr-auto-examples-neighbors-plot-digits-kde-sampling-py)

    ## Kernel Density Estimate of Species Distributions: an example of Kernel Density estimation using the Haversine distance metric to visualize geospatial data (https://scikit-learn.org/stable/auto_examples/neighbors/plot_species_kde.html#sphx-glr-auto-examples-neighbors-plot-species-kde-py)
 

