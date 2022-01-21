########## 1.5.8. Formulação matemática ##########

    # Descrevemos aqui os detalhes matemáticos do procedimento SGD. Uma boa visão geral com taxas de convergência pode ser encontrada em 12.

    # Dado um conjunto de exemplos de treinamento (x_1, y_1), \ldots, (x_n, y_n) onde x_i \in \mathbf{R}^m e y_i \in \mathcal{R} (y_i \in{-1, 1} para classificação), nosso objetivo é aprender uma função de pontuação linear f(x) = w^T x + b com parâmetros de modelo w \in \mathbf{R}^m e interceptar b \in \mathbf{R}. Para fazer previsões para classificação binária, simplesmente olhamos para o sinal de f(x). Para encontrar os parâmetros do modelo, minimizamos o erro de treinamento regularizado dado por

        # E(w,b) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, f(x_i)) + \alpha R(w)

    # onde L é uma função de perda que mede o (des)ajuste do modelo e R é um termo de regularização (também conhecido como penalidade) que penaliza a complexidade do modelo; \alpha > 0 é um hiperparâmetro não negativo que controla a força da regularização. 

    # Diferentes escolhas para L implicam em diferentes classificadores ou regressores:

        # Dobradiça (margem suave): equivalente à Classificação do Vetor de Suporte. L(y_i, f(x_i)) = \max(0, 1 - y_i f(x_i)).

        # Perceptron: L(y_i, f(x_i)) = \max(0, - y_i f(x_i)).

        # Huber modificado: L(y_i, f(x_i)) = \max(0, 1 - y_i f(x_i))^2 if y_i f(x_i) >1, e L(y_i, f(x_i)) = -4 y_i f(x_i) caso contrário.

        # Log: equivalente à Regressão Logística. L(y_i, f(x_i)) = \log(1 + \exp (-y_i f(x_i))).

        # Mínimos Quadrados: Regressão linear (Ridge ou Lasso dependendo de R). L(y_i, f(x_i)) = \frac{1}{2}(y_i - f(x_i))^2.

        # Huber: menos sensível aos valores discrepantes do que aos mínimos quadrados. É equivalente a mínimos quadrados quando |y_i - f(x_i)| \leq \varepsilon e L(y_i, f(x_i)) = \varepsilon |y_i - f(x_i)| - \frac{1}{2}\varepsilon^2 caso contrário.

        # Insensível a Epsilon: (margem suave) equivalente à regressão de vetor de suporte. L(y_i, f(x_i)) = \max(0, |y_i - f(x_i)| - \varepsilon).

    # Todas as funções de perda acima podem ser consideradas como um limite superior no erro de classificação incorreta (perda zero-um), conforme mostrado na Figura abaixo. 

        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_loss_functions.html

    # As escolhas populares para o termo de regularização R (o parâmetro de penalidade) incluem:

        # Norma L2: R(w) := \frac{1}{2} \sum_{j=1}^{m} w_j^2 = ||w||_2^2,

        # Norma L1: R(w) := \sum_{j=1}^{m} |w_j|, o que leva a soluções esparsas.

        # Rede elástica: R(w) := \frac{\rho}{2} \sum_{j=1}^{n} w_j^2 +(1-\rho) \sum_{j=1}^{m} |w_j|, uma combinação convexa de L2 e L1, onde \rho é dado por 1 - l1_ratio. 


            # https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_penalties.html

    


##### 1.5.8.1. SGD


    # O gradiente descendente estocástico é um método de otimização para problemas de otimização sem restrições. Em contraste com a descida do gradiente (em lote), o SGD aproxima o gradiente verdadeiro de E(w,b) considerando um único exemplo de treinamento por vez.

    # A classe SGDClassifier implementa uma rotina de aprendizado SGD de primeira ordem. O algoritmo itera sobre os exemplos de treinamento e para cada exemplo atualiza os parâmetros do modelo de acordo com a regra de atualização dada por 


        # w \leftarrow w - \eta \left[\alpha \frac{\partial R(w)}{\partial w}
        # + \frac{\partial L(w^T x_i + b, y_i)}{\partial w}\right]


    # onde \eta é a taxa de aprendizado que controla o tamanho do passo no espaço de parâmetros. O intercepto b é atualizado de forma semelhante, mas sem regularização (e com decaimento adicional para matrizes esparsas, conforme detalhado em Detalhes de implementação).

    # A taxa de aprendizagem \eta pode ser constante ou decair gradualmente. Para classificação, o cronograma de taxa de aprendizado padrão (learning_rate='optimal') é dado por 

        # \eta^{(t)} = \frac {1}{\alpha  (t_0 + t)}

    # onde t é o passo de tempo (há um total de n_amostras * n_iter passos de tempo), t_0 é determinado com base em uma heurística proposta por Léon Bottou tal que as atualizações iniciais esperadas são comparáveis com o tamanho esperado dos pesos (isso assumindo que o norma das amostras de treinamento é de aproximadamente 1). A definição exata pode ser encontrada em _init_t em BaseSGD.

    # Para regressão, o cronograma de taxa de aprendizado padrão é o escalonamento inverso (learning_rate='invscaling'), dado por 

        # \eta^{(t)} = \frac{eta_0}{t^{power\_t}}

    # onde eta_0 e power\_t são hiperparâmetros escolhidos pelo usuário via eta0 e power_t, resp.

    # Para uma taxa de aprendizado constante, use learning_rate='constant' e use eta0 para especificar a taxa de aprendizado.

    # Para uma taxa de aprendizado decrescente adaptativamente, use learning_rate='adaptive' e use eta0 para especificar a taxa de aprendizado inicial. Quando o critério de parada é alcançado, a taxa de aprendizado é dividida por 5 e o algoritmo não para. O algoritmo para quando a taxa de aprendizado fica abaixo de 1e-6.

    # Os parâmetros do modelo podem ser acessados através dos atributos coef_ e intercept_: coef_ contém os pesos w e intercept_ contém b.

    # Ao usar Averaged SGD (com o parâmetro average), coef_ é definido como o peso médio em todas as atualizações: coef_ = \frac{1}{T} \sum_{t=0}^{T-1} w^{(t )}, onde T é o número total de atualizações, encontrado no atributo t_. 