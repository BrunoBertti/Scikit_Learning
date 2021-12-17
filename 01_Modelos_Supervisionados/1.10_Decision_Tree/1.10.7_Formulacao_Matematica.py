########## 1.10.7. Formulação matemática  ##########

    # Dados os vetores de treinamento x_i \ em R ^ n, i = 1,…, le um vetor de rótulo y \ em R ^ l, uma árvore de decisão particiona recursivamente o espaço de características de modo que as amostras com os mesmos rótulos ou valores de destino semelhantes sejam agrupadas juntos.


    # Deixe que os dados no nó m sejam representados por Q_m com N_m amostras. Para cada divisão candidata \ theta = (j, t_m) que consiste em um recurso j e limite t_m, particionar os dados em subconjuntos Q_m ^ {esquerda} (\ theta) e Q_m ^ {direita} (\ theta) 

        # \begin{align}\begin{aligned}Q_m^{left}(\theta) = \{(x, y) | x_j <= t_m\}\\Q_m^{right}(\theta) = Q_m \setminus Q_m^{left}(\theta)\end{aligned}\end{align}

    # A qualidade de uma divisão candidata do nó m é então calculada usando uma função de impureza ou função de perda H (), a escolha de qual depende da tarefa a ser resolvida (classificação ou regressão) 

        # G(Q_m, \theta) = \frac{N_m^{left}}{N_m} H(Q_m^{left}(\theta))
        #        + \frac{N_m^{right}}{N_m} H(Q_m^{right}(\theta))

    # Selecione os parâmetros que minimizam a impureza 

        # \theta^* = \operatorname{argmin}_\theta  G(Q_m, \theta)

    # Percorra novamente para os subconjuntos Q_m ^ {left} (\ theta ^ *) e Q_m ^ {right} (\ theta ^ *) até que a profundidade máxima permitida seja alcançada, N_m <\ min_ {samples} ou N_m = 1.


##### 1.10.7.1. Critérios de classificação 

    # Se um alvo é um resultado de classificação assumindo valores 0,1, ..., K-1, para o nó m, deixe

        # p_ {mk} = 1 / N_m \ sum_ {y \ in Q_m} I (y = k)

    # ser a proporção de observações de classe k no nó m. m Se for um nó terminal, predict_proba para esta região é definido como p_ {mk}. As medidas comuns de impureza são as seguintes. 

    # Gini:

        # H(Q_m) = \sum_k p_{mk} (1 - p_{mk})

    # Entropy:

        # H(Q_m) = - \sum_k p_{mk} \log(p_{mk})

    # Misclassification:

        # H(Q_m) = 1 - \max(p_{mk})


##### 1.10.7.2. Critérios de regressão 

    # Se o alvo for um valor contínuo, então para o nó m, os critérios comuns para minimizar quanto à determinação de locais para divisões futuras são Erro Quadrático Médio (erro MSE ou L2), desvio de Poisson, bem como Erro Médio Absoluto (erro MAE ou L1). O MSE e o desvio de Poisson definem o valor previsto dos nós terminais para o valor médio aprendido \ bar {y} _m do nó, enquanto o MAE define o valor previsto dos nós terminais para a mediana mediana (y) _m.


    # Erro médio quadrático:


        # \ begin {align} \ begin {alinhados} \ bar {y} _m = \ frac {1} {N_m} \ sum_ {y \ in Q_m} y \\ H (Q_m) = \ frac {1} {N_m} \ sum_ {y \ in Q_m} (y - \ bar {y} _m) ^ 2 \ end {alinhado} \ end {alinhar}


    # Meio desvio de Poisson:


        # H (Q_m) = \ frac {1} {N_m} \ sum_ {y \ in Q_m} (y \ log \ frac {y} {\ bar {y} _m}
        #            - y + \ bar {y} _m)


    # Definir criterio = "poisson" pode ser uma boa escolha se seu alvo for uma contagem ou uma frequencia (contagem por alguma unidade). Em qualquer caso, y> = 0 é uma condição necessária para usar este critério. Observe que ele se ajusta muito mais lentamente do que o critério MSE.


    # Erro médio absoluto:


        # \ begin {align} \ begin {alinhados} mediana (y) _m = \ underset {y \ in Q_m} {\ mathrm {median}} (y) \\ H (Q_m) = \ frac {1} {N_m} \ soma_ {y \ in Q_m} | y - mediana (y) _m | \ end {alinhado} \ end {alinhar}

    # Observe que ele se ajusta muito mais lentamente do que o critério MSE. 