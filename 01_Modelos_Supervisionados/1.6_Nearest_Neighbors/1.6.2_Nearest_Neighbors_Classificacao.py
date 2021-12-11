########## 1.6.2. Classificação de vizinhos mais próximos ##########

    # A classificação baseada em vizinhos é um tipo de aprendizagem baseada em instância ou aprendizagem não generalizante: ela não tenta construir um modelo interno geral, mas simplesmente armazena instâncias dos dados de treinamento. A classificação é calculada a partir de um voto de maioria simples dos vizinhos mais próximos de cada ponto: um ponto de consulta é atribuído à classe de dados que possui o maior número de representantes nos vizinhos mais próximos do ponto.

    # scikit-learn implementa dois classificadores de vizinhos mais próximos diferentes: KNeighborsClassifier implementa o aprendizado com base nos vizinhos mais próximos de cada ponto de consulta, onde k é um valor inteiro especificado pelo usuário. RadiusNeighborsClassifier implementa o aprendizado com base no número de vizinhos dentro de um raio fixo r de cada ponto de treinamento, onde r é um valor de ponto flutuante especificado pelo usuário.

    # A classificação de k-neighbours em KNeighborsClassifier é a técnica mais comumente usada. A escolha ótima do valor k é altamente dependente dos dados: em geral, um k maior suprime os efeitos do ruído, mas torna os limites de classificação menos distintos.

    # Nos casos em que os dados não são amostrados de maneira uniforme, a classificação de vizinhos baseada em raio em RadiusNeighborsClassifier pode ser uma escolha melhor. O usuário especifica um raio fixo r, de forma que pontos em vizinhanças mais esparsas usem menos vizinhos mais próximos para a classificação. Para espaços de parâmetros de alta dimensão, este método torna-se menos eficaz devido à chamada “maldição da dimensionalidade”.

    # A classificação básica de vizinhos mais próximos usa pesos uniformes: ou seja, o valor atribuído a um ponto de consulta é calculado a partir de uma maioria simples de votos dos vizinhos mais próximos. Em algumas circunstâncias, é melhor pesar os vizinhos de forma que os vizinhos mais próximos contribuam mais para o ajuste. Isso pode ser feito por meio da palavra-chave pesos. O valor padrão, pesos = 'uniforme', atribui pesos uniformes a cada vizinho. pesos = 'distância' atribui pesos proporcionais ao inverso da distância do ponto de consulta. Alternativamente, uma função de distância definida pelo usuário pode ser fornecida para calcular os pesos. 


    ## Exemplos:

    ## Nearest Neighbors Classification: an example of classification using nearest neighbors.(https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py)