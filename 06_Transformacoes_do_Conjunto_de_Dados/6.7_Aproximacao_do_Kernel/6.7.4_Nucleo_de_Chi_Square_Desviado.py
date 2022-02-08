########## 6.7.4. Núcleo de Chi Square Desviado ##########



    # O kernel assimétrico qui-quadrado é dado por: 


        # k(x,y) = \prod_i \frac{2\sqrt{x_i+c}\sqrt{y_i+c}}{x_i + y_i + 2c}



    # Ele tem propriedades semelhantes ao kernel qui-quadrado exponenciado frequentemente usado em visão computacional, mas permite uma simples aproximação de Monte Carlo do mapa de características.

    # O uso do SkewedChi2Sampler é o mesmo que o uso descrito acima para o RBFSampler. A única diferença está no parâmetro livre, que é chamado de c. Para uma motivação para este mapeamento e os detalhes matemáticos veja [LS2010]. 