########## 2.1.2. Mistura Variacional Bayesiana Gaussiana ##########

    # O objeto BayesianGaussianMixture implementa uma variante do modelo de mistura gaussiana com algoritmos de inferência variacional. A API é semelhante à definida por GaussianMixture. 



##### 2.1.2.1. Algoritmo de estimativa: inferência variacional 

    # A inferência variacional é uma extensão da maximização da expectativa que maximiza um limite inferior na evidência do modelo (incluindo as anteriores) em vez da probabilidade dos dados. O princípio por trás dos métodos variacionais é o mesmo que a maximização da expectativa (ou seja, ambos são algoritmos iterativos que alternam entre encontrar as probabilidades de cada ponto a ser gerado por cada mistura e ajustar a mistura a esses pontos atribuídos), mas os métodos variacionais adicionam regularização por integrando informações de distribuições anteriores. Isso evita as singularidades frequentemente encontradas em soluções de maximização de expectativa, mas introduz alguns vieses sutis ao modelo. A inferência costuma ser notavelmente mais lenta, mas não tanto a ponto de tornar o uso impraticável.

    # Devido à sua natureza bayesiana, o algoritmo variacional precisa de mais hiperparâmetros do que a maximização da expectativa, sendo o mais importante deles o parâmetro de concentração weight_concentration_prior. Especificar um valor baixo para a concentração anterior fará com que o modelo coloque a maior parte do peso em alguns componentes, defina os pesos dos componentes restantes muito próximos de zero. Altos valores da concentração anterior permitirão que um maior número de componentes sejam ativos na mistura.

    # A implementação dos parâmetros da classe BayesianGaussianMixture propõe dois tipos de prior para a distribuição de pesos: um modelo de mistura finita com distribuição de Dirichlet e um modelo de mistura infinita com o Processo de Dirichlet. Na prática, o algoritmo de inferência do Processo de Dirichlet é aproximado e usa uma distribuição truncada com um número máximo fixo de componentes (chamada de representação de quebra de bastão). O número de componentes realmente usados ​​quase sempre depende dos dados.

    # A próxima figura compara os resultados obtidos para os diferentes tipos de concentração de peso anterior (parâmetro weight_concentration_prior_type) para diferentes valores de weight_concentration_prior. Aqui, podemos ver que o valor do parâmetro weight_concentration_prior tem um forte impacto no número efetivo de componentes ativos obtidos. Também podemos notar que grandes valores para o peso da concentração anterior levam a pesos mais uniformes quando o tipo de prior é ‘dirichlet_distribution’, embora não seja necessariamente o caso para o tipo ‘dirichlet_process’ (usado por padrão). 

        # https://scikit-learn.org/stable/auto_examples/mixture/plot_concentration_prior.html

    
    # Os exemplos abaixo comparam os modelos de mistura gaussiana com um número fixo de componentes aos modelos de mistura gaussiana variacional com um processo de Dirichlet anterior. Aqui, uma mistura gaussiana clássica é ajustada com 5 componentes em um conjunto de dados composto por 2 clusters. Podemos ver que a mistura gaussiana variacional com um processo de Dirichlet anterior é capaz de se limitar a apenas 2 componentes, enquanto a mistura gaussiana ajusta os dados com um número fixo de componentes que deve ser definido a priori pelo usuário. Neste caso, o usuário selecionou n_components = 5 que não corresponde à verdadeira distribuição generativa deste conjunto de dados de brinquedo. Observe que, com muito poucas observações, os modelos de mistura gaussiana variacional com um processo de Dirichlet anterior podem assumir uma posição conservadora e ajustar apenas um componente. 

        # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html

    # Na figura a seguir, estamos ajustando um conjunto de dados não bem representado por uma mistura gaussiana. Ajustando o weight_concentration_prior, o parâmetro da BayesianGaussianMixture controla o número de componentes usados para ajustar esses dados. Também apresentamos nos dois últimos gráficos uma amostra aleatória gerada a partir das duas misturas resultantes. 

        # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_sin.html

    

    ## Exemplos:

    ## See Gaussian Mixture Model Ellipsoids for an example on plotting the confidence ellipsoids for both GaussianMixture and BayesianGaussianMixture. (https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py)

    ## Gaussian Mixture Model Sine Curve shows using GaussianMixture and BayesianGaussianMixture to fit a sine wave. (https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_sin.html#sphx-glr-auto-examples-mixture-plot-gmm-sin-py)

    ## See Concentration Prior Type Analysis of Variation Bayesian Gaussian Mixture for an example plotting the confidence ellipsoids for the BayesianGaussianMixture with different weight_concentration_prior_type for different values of the parameter weight_concentration_prior. (https://scikit-learn.org/stable/auto_examples/mixture/plot_concentration_prior.html#sphx-glr-auto-examples-mixture-plot-concentration-prior-py)

##### 2.1.2.2. Prós e contras da inferência variacional com BayesianGaussianMixture

##### 2.1.2.2.1. Prós 

    # Seleção automática: quando weight_concentration_prior é pequeno o suficiente e n_components é maior do que o que é considerado necessário pelo modelo, o modelo de mistura Bayesiana Variacional tem uma tendência natural de definir alguns valores de pesos de mistura próximos de zero. Isso torna possível deixar o modelo escolher um número adequado de componentes eficazes automaticamente. Apenas um limite superior desse número precisa ser fornecido. Observe, entretanto, que o número “ideal” de componentes ativos é muito específico do aplicativo e normalmente é mal definido em uma configuração de exploração de dados.

    # Menos sensibilidade ao número de parâmetros: ao contrário dos modelos finitos, que quase sempre usarão todos os componentes tanto quanto podem e, portanto, produzirão soluções totalmente diferentes para diferentes números de componentes, a inferência variacional com um processo de Dirichlet anterior (weight_concentration_prior_type = 'dirichlet_process') não mudará muito com mudanças nos parâmetros, levando a mais estabilidade e menos ajuste.

    # Regularização: devido à incorporação de informações prévias, as soluções variacionais apresentam menos casos patológicos especiais do que as soluções de maximização da expectativa. 


##### 2.1.2.2.2. Contras 

    # Velocidade: a parametrização extra necessária para a inferência variacional torna a inferência mais lenta, embora não muito.

    # Hiperparâmetros: este algoritmo precisa de um hiperparâmetro extra que pode precisar de ajuste experimental por meio de validação cruzada.

    # Viés: há muitos vieses implícitos nos algoritmos de inferência (e também no processo de Dirichlet, se usado) e, sempre que houver uma incompatibilidade entre esses vieses e os dados, pode ser possível ajustar modelos melhores usando uma mistura finita.



##### 2.1.2.3. O Processo Dirichlet 

    # Aqui, descrevemos algoritmos de inferência variacional na mistura de processos de Dirichlet. O processo de Dirichlet é uma distribuição de probabilidade anterior em agrupamentos com um número infinito e ilimitado de partições. As técnicas variacionais nos permitem incorporar esta estrutura anterior em modelos de mistura gaussiana quase sem penalidade no tempo de inferência, comparando com um modelo de mistura gaussiana finito.

    # Uma questão importante é como o processo de Dirichlet pode usar um número infinito e ilimitado de clusters e ainda ser consistente. Embora uma explicação completa não se encaixe neste manual, pode-se pensar em sua analogia com o processo de quebra do bastão para ajudar a entendê-lo. O processo de quebra de palitos é uma história geradora para o processo de Dirichlet. Começamos com uma vara de comprimento unitário e em cada etapa quebramos uma parte da vara restante. A cada vez, associamos o comprimento do pedaço de pau à proporção de pontos que caem em um grupo da mistura. No final, para representar a mistura infinita, associamos o último pedaço restante do stick à proporção de pontos que não se enquadram em todos os outros grupos. O comprimento de cada peça é uma variável aleatória com probabilidade proporcional ao parâmetro de concentração. O valor menor da concentração dividirá o comprimento da unidade em pedaços maiores do bastão (definindo uma distribuição mais concentrada). Valores de concentração maiores criarão pedaços menores do stick (aumentando o número de componentes com pesos diferentes de zero).

    # As técnicas de inferência variacional para o processo de Dirichlet ainda funcionam com uma aproximação finita para este modelo de mistura infinita, mas em vez de ter que especificar a priori quantos componentes se deseja usar, apenas especifica o parâmetro de concentração e um limite superior no número de mistura componentes (este limite superior, supondo que seja maior do que o número “verdadeiro” de componentes, afeta apenas a complexidade algorítmica, não o número real de componentes usados).