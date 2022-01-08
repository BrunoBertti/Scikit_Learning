########## 2.9.1. Máquinas Boltzmann restritas  ##########

    # As máquinas de Boltzmann restritas (RBM) são aprendizes de recursos não lineares não supervisionados com base em um modelo probabilístico. Os recursos extraídos por um RBM ou uma hierarquia de RBMs geralmente fornecem bons resultados quando alimentados em um classificador linear, como um SVM linear ou um perceptron.

    # O modelo faz suposições sobre a distribuição dos insumos. No momento, o scikit-learn fornece apenas BernoulliRBM, que assume que as entradas são valores binários ou valores entre 0 e 1, cada um codificando a probabilidade de que o recurso específico seja ativado.

    # O RBM tenta maximizar a probabilidade dos dados usando um modelo gráfico específico. O algoritmo de aprendizagem de parâmetro usado (probabilidade máxima estocástica) evita que as representações se afastem dos dados de entrada, o que as faz capturar regularidades interessantes, mas torna o modelo menos útil para pequenos conjuntos de dados e geralmente não é útil para estimativa de densidade.

    # O método ganhou popularidade para inicializar redes neurais profundas com os pesos de RBMs independentes. Este método é conhecido como pré-treinamento não supervisionado. 


        # https://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html



    ## Exemplos:


    ## https://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html#sphx-glr-auto-examples-neural-networks-plot-rbm-logistic-classification-py



##### 2.9.1.1. Modelo gráfico e parametrização 

    # O modelo gráfico de um RBM é um gráfico bipartido totalmente conectado. 

        # https://scikit-learn.org/stable/_images/rbm_graph.png


    # Os nós são variáveis aleatórias cujos estados dependem do estado dos outros nós aos quais estão conectados. O modelo é, portanto, parametrizado pelos pesos das conexões, bem como um termo de interceptação (viés) para cada unidade visível e oculta, omitido da imagem para simplificar.

    # A função de energia mede a qualidade de uma atribuição conjunta: 

        # E(\mathbf{v}, \mathbf{h}) = -\sum_i \sum_j w_{ij}v_ih_j - \sum_i b_iv_i- \sum_j c_jh_j

    # Na fórmula acima, be são os vetores de interceptação para as camadas visível e oculta, respectivamente. A probabilidade conjunta do modelo é definida em termos de energia: 

        # P(\mathbf{v}, \mathbf{h}) = \frac{e^{-E(\mathbf{v}, \mathbf{h})}}{Z}

    
    # A palavra restrito refere-se à estrutura bipartida do modelo, que proíbe a interação direta entre unidades ocultas, ou entre unidades visíveis. Isso significa que as seguintes independências condicionais são assumidas: 

        # \begin{split}h_i \bot h_j | \mathbf{v} \\
        # v_i \bot v_j | \mathbf{h}\end{split}


    # A estrutura bipartida permite o uso de amostragem de bloco de Gibbs eficiente para inferência. 






##### 2.9.1.2. Máquinas Bernoulli Restricted Boltzmann


    # No BernoulliRBM, todas as unidades são unidades estocásticas binárias. Isso significa que os dados de entrada devem ser binários ou com valor real entre 0 e 1, significando a probabilidade de que a unidade visível seja ligada ou desligada. Este é um bom modelo para reconhecimento de caracteres, onde o interesse é em quais pixels estão ativos e quais não estão. Para imagens de cenas naturais, ele não se ajusta mais devido ao fundo, à profundidade e à tendência dos pixels vizinhos de assumirem os mesmos valores.

    # A distribuição de probabilidade condicional de cada unidade é dada pela função de ativação logística sigmóide da entrada que recebe: 


        # \begin{split}P(v_i=1|\mathbf{h}) = \sigma(\sum_j w_{ij}h_j + b_i) \\
        # P(h_i=1|\mathbf{v}) = \sigma(\sum_i w_{ij}v_i + c_j)\end{split}


    # onde \ sigma é a função sigmóide logística: 

        # \sigma(x) = \frac{1}{1 + e^{-x}}


##### 2.9.1.3. Aprendizagem de máxima verossimilhança estocástica 

    # O algoritmo de treinamento implementado no BernoulliRBM é conhecido como Stochastic Maximum Likelihood (SML) ou Persistent Contrastive Divergence (PCD). Otimizar a probabilidade máxima diretamente é inviável devido à forma da probabilidade dos dados:

        # \ log P (v) = \ log \ sum_h e ^ {- E (v, h)} - \ log \ sum_ {x, y} e ^ {- E (x, y)}

    # Para simplificar, a equação acima foi escrita para um único exemplo de treinamento. O gradiente em relação aos pesos é formado por dois termos correspondentes aos anteriores. Geralmente são conhecidos como gradiente positivo e gradiente negativo, por causa de seus respectivos sinais. Nesta implementação, os gradientes são estimados em minilotes de amostras.

    # Ao maximizar a probabilidade de log, o gradiente positivo faz com que o modelo prefira estados ocultos que sejam compatíveis com os dados de treinamento observados. Por causa da estrutura bipartida dos RBMs, ele pode ser calculado com eficiência. O gradiente negativo, entretanto, é intratável. Seu objetivo é diminuir a energia dos estados de junta que o modelo prefere, fazendo com que ele permaneça fiel aos dados. Pode ser aproximado por cadeia de Markov Monte Carlo usando amostragem de bloco de Gibbs por amostragem iterativa de v e h dados um do outro, até que a cadeia se misture. As amostras geradas dessa forma são às vezes chamadas de partículas de fantasia. Isso é ineficiente e é difícil determinar se a cadeia de Markov se mistura.

    # O método de Divergência Contrastiva sugere parar a cadeia após um pequeno número de iterações, k, geralmente até 1. Este método é rápido e tem baixa variância, mas as amostras estão longe da distribuição do modelo.

    # A Divergência Contrastiva Persistente resolve isso. Em vez de iniciar uma nova cadeia cada vez que o gradiente é necessário, e realizar apenas uma etapa de amostragem de Gibbs, no PCD mantemos uma série de cadeias (partículas de fantasia) que são atualizadas k etapas de Gibbs após cada atualização de peso. Isso permite que as partículas explorem o espaço de forma mais completa. 



    ## Referências:

    # “A fast learning algorithm for deep belief nets” G. Hinton, S. Osindero, Y.-W. Teh, 2006 (https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)

    # “Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient” T. Tieleman, 2008 (https://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf)