########## 1.17.7. Formulação matemática ##########


    # Dado um conjunto de exemplos de treinamento (x_1, y_1), (x_2, y_2), \ ldots, (x_n, y_n) onde x_i \ in \ mathbf {R} ^ n e y_i \ in \ {0, 1 \}, a uma camada oculta um neurônio oculto MLP aprende a função f (x) = W_2 g (W_1 ^ T x + b_1) + b_2 onde W_1 \ in \ mathbf {R} ^ m e W_2, b_1, b_2 \ in \ mathbf {R } são parâmetros do modelo.
    # W_1, W_2 representam os pesos da camada de entrada e da camada oculta, respectivamente; e b_1, b_2 representam a polarização adicionada à camada oculta e à camada de saída, respectivamente. g (\ cdot): R \ rightarrow R é a função de ativação, definida por padrão como o tan hiperbólico. É dado como,


        # g (z) = \ frac {e ^ z-e ^ {- z}} {e ^ z + e ^ {- z}}


    # Para classificação binária, f (x) passa pela função logística g (z) = 1 / (1 + e ^ {- z}) para obter valores de saída entre zero e um. Um limite, definido como 0,5, atribuiria amostras de saídas maiores ou iguais a 0,5 à classe positiva e o restante à classe negativa.


    # Se houver mais de duas classes, f (x) em si seria um vetor de tamanho (n_classes,). Em vez de passar pela função logística, ele passa pela função softmax, que é escrita como,


        # \ text {softmax} (z) _i = \ frac {\ exp (z_i)} {\ sum_ {l = 1} ^ k \ exp (z_l)}

    # onde z_i representa o i ésimo elemento da entrada para softmax, que corresponde à classe i, ek é o número de classes. O resultado é um vetor contendo as probabilidades de que a amostra x pertença a cada classe. A saída é a classe com a maior probabilidade. 

    # Na regressão, a saída permanece como f (x); portanto, a função de ativação de saída é apenas a função de identidade.

    # O MLP usa funções de perda diferentes dependendo do tipo de problema. A função de perda para classificação é Entropia Cruzada, que no caso binário é dada como,

        # Loss (\ hat {y}, y, W) = -y \ ln {\ hat {y}} - (1-y) \ ln {(1- \ hat {y})} + \ alpha || W | | _2 ^ 2


    # onde \ alpha || W || _2 ^ 2 é um termo de regularização L2 (também conhecido como penalidade) que penaliza modelos complexos; e \ alpha> 0 é um hiperparâmetro não negativo que controla a magnitude da penalidade.

    # Para a regressão, o MLP usa a função de perda de erro quadrado; escrito como,

        # Perda (\ hat {y}, y, W) = \ frac {1} {2} || \ hat {y} - y || _2 ^ 2 + \ frac {\ alpha} {2} || W || _2 ^ 2

    # Começando com pesos aleatórios iniciais, o perceptron multicamadas (MLP) minimiza a função de perda atualizando repetidamente esses pesos. Depois de calcular a perda, uma passagem para trás a propaga da camada de saída para as camadas anteriores, fornecendo a cada parâmetro de peso um valor de atualização destinado a diminuir a perda.


    # No gradiente descendente, o gradiente \ nabla Loss_ {W} da perda em relação aos pesos é calculado e deduzido de W. Mais formalmente, isso é expresso como,


        # W ^ {i + 1} = W ^ i - \ epsilon \ nabla {Perda} _ {W} ^ {i}

    # onde i é a etapa de iteração e \ epsilon é a taxa de aprendizado com um valor maior que 0.

    # O algoritmo para quando atinge um número máximo predefinido de iterações; ou quando a melhora na perda está abaixo de um certo número pequeno. 