########## 1.4.2 Regressão ##########

    # O método de classificação de vetores de suporte pode ser estendido para resolver problemas de regressão. Este método é denominado Support Vector Regression.

    # O modelo produzido pela classificação do vetor de suporte (conforme descrito acima) depende apenas de um subconjunto dos dados de treinamento, porque a função de custo para construir o modelo não se preocupa com os pontos de treinamento que estão além da margem. Analogamente, o modelo produzido pelo Support Vector Regression depende apenas de um subconjunto dos dados de treinamento, pois a função de custo ignora as amostras cuja previsão está próxima de seu destino.

    # Existem três implementações diferentes de Regressão de vetores de suporte: SVR, NuSVR e LinearSVR. LinearSVR fornece uma implementação mais rápida do que SVR, mas considera apenas o kernel linear, enquanto NuSVR implementa uma formulação ligeiramente diferente de SVR e LinearSVR. Consulte os detalhes de implementação para obter mais detalhes.

    # Tal como acontece com as classes de classificação, o método de ajuste tomará como vetores de argumento X, y, apenas que, neste caso, espera-se que y tenha valores de ponto flutuante em vez de valores inteiros: 

from sklearn import svm
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
regr = svm.SVR()
print(regr.fit(X,y))

print(regr.predict([[1,1]]))

    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py