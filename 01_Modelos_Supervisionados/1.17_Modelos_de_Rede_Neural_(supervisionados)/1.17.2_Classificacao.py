########## 1.17.2. Classificação ##########

    # A classe MLPClassifier implementa um algoritmo perceptron multicamadas (MLP) que treina usando retropropagação.

    # O MLP treina em duas matrizes: matriz X de tamanho (n_samples, n_features), que contém as amostras de treinamento representadas como vetores de recursos de ponto flutuante; e matriz y de tamanho (n_samples,), que contém os valores alvo (rótulos de classe) para as amostras de treinamento: 

from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

    # Após o ajuste (treinamento), o modelo pode prever rótulos para novas amostras: 

clf.predict([[2., 2.], [-1., -2.]])

    # O MLP pode ajustar um modelo não linear aos dados de treinamento. clf.coefs_ contém as matrizes de peso que constituem os parâmetros do modelo: 

[coef.shape for coef in clf.coefs_]

    # Atualmente, o MLPClassifier suporta apenas a função de perda de entropia cruzada, que permite estimativas de probabilidade executando o método predict_proba. 

    # O MLP treina usando Backpropagation. Mais precisamente, ele treina usando alguma forma de gradiente descendente e os gradientes são calculados usando Backpropagation. Para classificação, ele minimiza a função de perda de entropia cruzada, dando um vetor de estimativas de probabilidade P (y | x) por amostra x: 

clf.predict_proba([[2., 2.], [1., 2.]])

    # MLPClassifier oferece suporte à classificação multiclasse, aplicando Softmax como a função de saída.

    # Além disso, o modelo oferece suporte à classificação de vários rótulos, na qual uma amostra pode pertencer a mais de uma classe. Para cada classe, a saída bruta passa pela função logística. Valores maiores ou iguais a 0,5 são arredondados para 1, caso contrário, para 0. Para uma saída prevista de uma amostra, os índices onde o valor é 1 representam as classes atribuídas dessa amostra: 

X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15,), random_state=1)
clf.fit(X, y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(15,), random_state=1,
              solver='lbfgs')
clf.predict([[1., 2.]])
clf.predict([[0., 0.]])


    # Veja os exemplos abaixo e a docstring de MLPClassifier.fit para mais informações. 


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py

    ## https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py