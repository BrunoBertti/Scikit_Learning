########## 1.7.2. Exemplos de GPR ##########


##### 1.7.2.1. GPR com estimativa de nível de ruído

    # Este exemplo ilustra que o GPR com um kernel de soma incluindo um WhiteKernel pode estimar o nível de ruído dos dados. Uma ilustração do cenário de probabilidade log-marginal (LML) mostra que existem dois máximos locais de LML. 

        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html

    # A primeira corresponde a um modelo com alto nível de ruído e grande escala de comprimento, o que explica todas as variações dos dados por ruído. 

        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html
    
    # A segunda tem um nível de ruído menor e uma escala de comprimento menor, o que explica a maior parte da variação pela relação funcional livre de ruído. O segundo modelo tem maior probabilidade; no entanto, dependendo do valor inicial dos hiperparâmetros, a otimização baseada em gradiente também pode convergir para a solução de alto ruído. Portanto, é importante repetir a otimização várias vezes para diferentes inicializações. 

        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html







##### 1.7.2.2. Comparação de GPR e Regressão de Kernel Ridge


    # Tanto a regressão do cume do kernel (KRR) quanto o GPR aprendem uma função de destino empregando internamente o “truque do kernel”. O KRR aprende uma função linear no espaço induzida pelo respectivo kernel que corresponde a uma função não linear no espaço original. A função linear no espaço do kernel é escolhida com base na perda de erro quadrático médio com regularização de crista. O GPR usa o kernel para definir a covariância de uma distribuição anterior sobre as funções de destino e usa os dados de treinamento observados para definir uma função de verossimilhança. Com base no teorema de Bayes, define-se uma distribuição posterior (Gaussiana) sobre funções alvo, cuja média é utilizada para predição.

    # Uma grande diferença é que o GPR pode escolher os hiperparâmetros do kernel com base na subida do gradiente na função de probabilidade marginal, enquanto o KRR precisa realizar uma pesquisa de grade em uma função de perda validada cruzada (perda de erro quadrático médio). Uma outra diferença é que o GPR aprende um modelo generativo e probabilístico da função alvo e pode, assim, fornecer intervalos de confiança significativos e amostras posteriores junto com as previsões, enquanto o KRR fornece apenas previsões.

    # A figura a seguir ilustra ambos os métodos em um conjunto de dados artificial, que consiste em uma função alvo senoidal e ruído forte. A figura compara o modelo aprendido de KRR e GPR baseado em um kernel ExpSineSquared, que é adequado para aprender funções periódicas. Os hiperparâmetros do kernel controlam a suavidade (length_scale) e a periodicidade do kernel (periodicity). Além disso, o nível de ruído dos dados é aprendido explicitamente pelo GPR por um componente WhiteKernel adicional no kernel e pelo parâmetro de regularização alfa do KRR. 


        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html

    # A figura mostra que ambos os métodos aprendem modelos razoáveis da função alvo. GPR identifica corretamente a periodicidade da função como sendo aproximadamente 2*\pi (6,28), enquanto KRR escolhe a periodicidade dobrada 4*\pi. Além disso, o GPR fornece limites de confiança razoáveis na previsão que não estão disponíveis para KRR. Uma grande diferença entre os dois métodos é o tempo necessário para ajustar e prever: enquanto o ajuste de KRR é rápido em princípio, a busca em grade para otimização de hiperparâmetros escala exponencialmente com o número de hiperparâmetros (“maldição da dimensionalidade”). A otimização baseada em gradiente dos parâmetros no GPR não sofre dessa escala exponencial e, portanto, é consideravelmente mais rápida neste exemplo com espaço de hiperparâmetro tridimensional. O tempo de previsão é semelhante; no entanto, gerar a variância da distribuição preditiva de GPR leva muito mais tempo do que apenas prever a média. 








##### 1.7.2.3. GPR nos dados de CO2 de Mauna Loa 


    # Este exemplo é baseado na Seção 5.4.3 de [RW2006]. Ele ilustra um exemplo de engenharia de kernel complexa e otimização de hiperparâmetros usando subida de gradiente na probabilidade log-marginal. Os dados consistem nas concentrações médias mensais de CO2 atmosférico (em partes por milhão por volume (ppmv)) coletadas no Observatório Mauna Loa, no Havaí, entre 1958 e 1997. O objetivo é modelar a concentração de CO2 em função do tempo t .

    # O kernel é composto por vários termos que são responsáveis ​​por explicar diferentes propriedades do sinal:

        # uma tendência de crescimento suave e de longo prazo deve ser explicada por um kernel RBF. O kernel RBF com uma grande escala de comprimento impõe que este componente seja suave; não é imposto que a tendência está aumentando, o que deixa essa escolha para o GP. A escala de comprimento específica e a amplitude são hiperparâmetros livres.

        # um componente sazonal, que deve ser explicado pelo kernel periódico ExpSineSquared com uma periodicidade fixa de 1 ano. A escala de comprimento deste componente periódico, controlando sua suavidade, é um parâmetro livre. A fim de permitir o decaimento da periodicidade exata, o produto com um kernel RBF é obtido. A escala de comprimento deste componente RBF controla o tempo de decaimento e é mais um parâmetro livre.

        # irregularidades menores e de médio prazo devem ser explicadas por um componente do kernel RationalQuadratic, cuja escala de comprimento e parâmetro alfa, que determina a difusão das escalas de comprimento, devem ser determinados. De acordo com [RW2006], essas irregularidades podem ser melhor explicadas por um RationalQuadratic do que um componente do kernel RBF, provavelmente porque ele pode acomodar várias escalas de comprimento.

        # um termo de “ruído”, consistindo em uma contribuição do kernel RBF, que deve explicar os componentes de ruído correlacionados, como fenômenos climáticos locais, e uma contribuição de WhiteKernel para o ruído branco. As amplitudes relativas e a escala de comprimento do RBF são outros parâmetros livres.

    
    # Maximizar a probabilidade log-marginal após subtrair a média do alvo produz o seguinte kernel com um LML de -83,214:

    # Assim, a maior parte do sinal alvo (34,4 ppm) é explicada por uma tendência crescente de longo prazo (escala de comprimento 41,8 anos). O componente periódico tem uma amplitude de 3,27ppm, um tempo de decaimento de 180 anos e uma escala de comprimento de 1,44. O longo tempo de decaimento indica que temos um componente sazonal localmente muito próximo ao periódico. O ruído correlacionado tem uma amplitude de 0,197ppm com uma escala de comprimento de 0,138 anos e uma contribuição de ruído branco de 0,197ppm. Assim, o nível geral de ruído é muito pequeno, indicando que os dados podem ser muito bem explicados pelo modelo. A figura mostra também que o modelo faz previsões muito confiantes até por volta de 2015 

        # https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html