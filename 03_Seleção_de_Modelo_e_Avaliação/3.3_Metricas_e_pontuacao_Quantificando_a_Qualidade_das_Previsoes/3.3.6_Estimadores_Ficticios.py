##########  3.3.6. Estimadores fictícios  ##########



    # Ao fazer o aprendizado supervisionado, uma simples verificação de sanidade consiste em comparar o estimador de alguém com regras simples. DummyClassifier implementa várias dessas estratégias simples para classificação: 

        # stratified gera previsões aleatórias respeitando a distribuição de classes do conjunto de treinamento.

        # most_frequent sempre prevê o rótulo mais frequente no conjunto de treinamento.

        # prior sempre prevê a classe que maximiza a classe anterior (como most_frequent) e predict_proba retorna a classe anterior.

        # uniform gera previsões uniformemente aleatórias.

        # constant sempre prevê um rótulo constante que é fornecido pelo usuário.

        # Uma das principais motivações deste método é a pontuação F1, quando a classe positiva está em minoria.


    # Observe que com todas essas estratégias, o método de previsão ignora completamente os dados de entrada!

    # Para ilustrar o DummyClassifier, primeiro vamos criar um conjunto de dados desbalanceado: 

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
y[y != 1] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Em seguida, vamos comparar a precisão de SVC e most_frequent: 

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)
clf = DummyClassifier(strategy='most_frequent', random_state=0)
clf.fit(X_train, y_train)
DummyClassifier(random_state=0, strategy='most_frequent')
clf.score(X_test, y_test)

    # Vemos que o SVC não se sai muito melhor do que um classificador fictício. Agora, vamos alterar o kernel: 

clf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)


    # Vemos que a precisão foi aumentada para quase 100%. Uma estratégia de validação cruzada é recomendada para uma melhor estimativa da precisão, se não for muito caro para a CPU. Para obter mais informações, consulte a seção Validação cruzada: avaliando o desempenho do estimador. Além disso, se você deseja otimizar o espaço de parâmetros, é altamente recomendável usar uma metodologia apropriada; consulte a seção Ajustando os hiperparâmetros de um estimador para obter detalhes.

    # De maneira mais geral, quando a precisão de um classificador está muito próxima do aleatório, provavelmente significa que algo deu errado: recursos não são úteis, um hiperparâmetro não está ajustado corretamente, o classificador está sofrendo de desequilíbrio de classe, etc.

    # DummyRegressor também implementa quatro regras simples para regressão: 


        # A média sempre prevê a média dos alvos de treinamento.

        # median sempre prevê a mediana das metas de treinamento.

        # quantile sempre prevê um quantil fornecido pelo usuário dos alvos de treinamento.

        # constant sempre prevê um valor constante que é fornecido pelo usuário. 

    # Em todas essas estratégias, o método de previsão ignora completamente os dados de entrada. 