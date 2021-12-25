########## 1.17.1. Perceptron multicamadas ##########

    # Perceptron multicamadas (MLP) é um algoritmo de aprendizado supervisionado que aprende uma função f (\ cdot): R ^ m \ rightarrow R ^ o treinando em um conjunto de dados, onde m é o número de dimensões para entrada e o é o número de dimensões para saída. Dado um conjunto de recursos X = {x_1, x_2, ..., x_m} e um alvo y, ele pode aprender um aproximador de função não linear para classificação ou regressão. É diferente da regressão logística, pois entre a camada de entrada e a de saída pode haver uma ou mais camadas não lineares, chamadas camadas ocultas. A Figura 1 mostra um MLP de uma camada oculta com saída escalar.


        # https://scikit-learn.org/stable/_images/multilayerperceptron_network.png


    # A camada mais à esquerda, conhecida como camada de entrada, consiste em um conjunto de neurônios \ {x_i | x_1, x_2, ..., x_m \} representando os recursos de entrada. Cada neurônio na camada oculta transforma os valores da camada anterior com uma soma linear ponderada w_1x_1 + w_2x_2 + ... + w_mx_m, seguido por uma função de ativação não linear g (\ cdot): R \ rightarrow R - como a hiperbólica função tan. A camada de saída recebe os valores da última camada oculta e os transforma em valores de saída.


    # O módulo contém os atributos públicos coefs_ e intercepts_. coefs_ é uma lista de matrizes de peso, onde a matriz de peso no índice i representa os pesos entre a camada i e a camada i + 1. intercepts_ é uma lista de vetores de polarização, onde o vetor no índice i representa os valores de polarização adicionados à camada i + 1. 

    
    # As vantagens do Perceptron multicamadas são:


        # Capacidade de aprender modelos não lineares.

        # Capacidade de aprender modelos em tempo real (aprendizado on-line) usando o partial_fit.

    # As desvantagens do Multi-layer Perceptron (MLP) incluem:

        # OMLP com camadas ocultas tem uma função de perda não convexa onde existe mais de um mínimo local. Portanto, inicializações de peso aleatório diferentes podem levar a uma precisão de validação diferente.

        # OO MLP requer o ajuste de vários hiperparâmetros, como o número de neurônios ocultos, camadas e iterações.

        # O MLP é sensível ao dimensionamento de recursos.

    
    # Consulte a seção Dicas sobre uso prático que aborda algumas dessas desvantagens. 