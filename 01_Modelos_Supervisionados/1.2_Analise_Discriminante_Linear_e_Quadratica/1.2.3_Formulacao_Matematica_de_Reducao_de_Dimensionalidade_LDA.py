########## 1.2.3. Formulação matemática de redução de dimensionalidade LDA ##########


    # Primeiro note que K significa que \mu_k são vetores em \mathcal{R}^d, e eles estão em um subespaço afim H de dimensão no máximo K - 1 (2 pontos estão em uma linha, 3 pontos estão em um plano, etc.) ).

    # Como mencionado acima, podemos interpretar LDA como a atribuição de x à classe cuja média \mu_k é a mais próxima em termos de distância de Mahalanobis, enquanto também leva em conta as probabilidades anteriores da classe. Alternativamente, LDA é equivalente a primeiro esferificar os dados de modo que a matriz de covariância seja a identidade e, em seguida, atribuir x à média mais próxima em termos de distância euclidiana (ainda contabilizando as classes prioritárias).

    # Calcular distâncias euclidianas neste espaço d-dimensional é equivalente a primeiro projetar os pontos de dados em H e calcular as distâncias lá (já que as outras dimensões contribuirão igualmente para cada classe em termos de distância). Em outras palavras, se x estiver mais próximo de \mu_k no espaço original, também será o caso de H. Isso mostra que, implícito no classificador LDA, há uma redução de dimensionalidade por projeção linear em um espaço dimensional K-1 .

    # Podemos reduzir ainda mais a dimensão, para um L escolhido, projetando no subespaço linear H_L que maximiza a variância do \mu^*_k após a projeção (na verdade, estamos fazendo uma forma de PCA para a classe transformada significa \ mu^*_k). Este L corresponde ao parâmetro n_components usado no método de transformação. Veja 1 para mais detalhes. 