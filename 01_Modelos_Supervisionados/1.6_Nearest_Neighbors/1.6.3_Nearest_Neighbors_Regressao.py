########## 1.6.3. Regressão de vizinhos mais próximos  ##########

    # A regressão baseada em vizinhos pode ser usada nos casos em que os rótulos de dados são contínuos em vez de variáveis ​​discretas. O rótulo atribuído a um ponto de consulta é calculado com base na média dos rótulos de seus vizinhos mais próximos.



    # O scikit-learn implementa dois regressores de vizinhos diferentes: KNeighborsRegressor implementa o aprendizado com base nos k vizinhos mais próximos de cada ponto de consulta, onde k é um valor inteiro especificado pelo usuário. RadiusNeighborsRegressor implementa a aprendizagem com base nos vizinhos dentro de um raio fixo r do ponto de consulta, onde r é um valor de ponto flutuante especificado pelo usuário.


    # A regressão básica de vizinhos mais próximos usa pesos uniformes: ou seja, cada ponto na vizinhança local contribui uniformemente para a classificação de um ponto de consulta. Em algumas circunstâncias, pode ser vantajoso ponderar os pontos de forma que os pontos próximos contribuam mais para a regressão do que os pontos distantes. Isso pode ser feito por meio da palavra-chave pesos. O valor padrão, pesos = 'uniforme', atribui pesos iguais a todos os pontos. pesos = 'distância' atribui pesos proporcionais ao inverso da distância do ponto de consulta. Alternativamente, uma função de distância definida pelo usuário pode ser fornecida, a qual será usada para calcular os pesos.



    # O uso de vizinhos mais próximos de múltiplas saídas para regressão é demonstrado em Completação de face com estimadores de múltiplas saídas. Neste exemplo, as entradas X são os pixels da metade superior das faces e as saídas Y são os pixels da metade inferior dessas faces. 



    ## Exemplos

    ## Nearest Neighbors regression: an example of regression using nearest neighbors. (https://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html#sphx-glr-auto-examples-neighbors-plot-regression-py)

    ## Face completion with a multi-output estimators: an example of multi-output regression using nearest neighbors. (https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_multioutput_face_completion.html#sphx-glr-auto-examples-miscellaneous-plot-multioutput-face-completion-py)