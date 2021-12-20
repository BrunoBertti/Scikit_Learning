########## 1.11.6 Classificador de votação ##########

    # A ideia por trás do VotingClassifier é combinar classificadores de aprendizado de máquina conceitualmente diferentes e usar um voto majoritário ou a probabilidade média prevista (voto suave) para prever os rótulos das classes. Tal classificador pode ser útil para um conjunto de modelos de desempenho igualmente bom, a fim de equilibrar suas fraquezas individuais. 


##### 1.11.6.1. Rótulos da classe majoritária (votação majoritária / dura)

    # Na votação por maioria, o rótulo de classe previsto para uma amostra específica é o rótulo de classe que representa a maioria (modo) dos rótulos de classe previstos por cada classificador individual.

    # Por exemplo, se a previsão para uma determinada amostra for:

        # classificador 1 -> classe 1

        # classificador 2 -> classe 1

        # classificador 3 -> classe 2 


    # o VotingClassifier (com vote = 'hard') classificaria a amostra como “classe 1” com base no rótulo da classe majoritária.

    # Em caso de empate, o VotingClassifier selecionará a classe com base na ordem de classificação crescente. Por exemplo, no seguinte cenário 

        # classificador 1 -> classe 2

        # classificador 2 -> classe 1 

    # o rótulo de classe 1 será atribuído à amostra. 


##### 1.11.6.2. Uso


    # O exemplo a seguir mostra como ajustar o classificador de regra majoritária: 

from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):     
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



##### 1.11.6.3. Probabilidades médias ponderadas (voto suave)


    # Em contraste com a votação por maioria (votação definitiva), a votação suave retorna o rótulo da classe como argmax da soma das probabilidades previstas.

    # Pesos específicos podem ser atribuídos a cada classificador por meio do parâmetro de pesos. Quando os pesos são fornecidos, as probabilidades de classe previstas para cada classificador são coletadas, multiplicadas pelo peso do classificador e calculadas. O rótulo da classe final é então derivado do rótulo da classe com a probabilidade média mais alta.

    # Para ilustrar isso com um exemplo simples, vamos supor que temos 3 classificadores e problemas de classificação de 3 classes, onde atribuímos pesos iguais a todos os classificadores: w1 = 1, w2 = 1, w3 = 1.

    # As probabilidades médias ponderadas para uma amostra seriam então calculadas da seguinte forma: 


    # classifier        class 1         class 2         class 3
    # classifier 1      w1 * 0.2        w1 * 0.5        w1 * 0.3
    # classifier 2      w2 * 0.6        w2 * 0.3        w2 * 0.1
    # classifier 3      w3 * 0.3        w3 * 0.4        w3 * 0.3
    # weighted average  0.37            0.4             0.23


    # Aqui, o rótulo de classe previsto é 2, pois tem a probabilidade média mais alta.

    # O exemplo a seguir ilustra como as regiões de decisão podem mudar quando um soft VotingClassifier é usado com base em uma máquina de vetor de suporte linear, uma árvore de decisão e um classificador K-vizinho mais próximo: 


from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier

# Carregando alguns dados de exemplo 
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

# Classificadores de treinamento 
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])

clf1 = clf1.fit(X, y)
clf2 = clf2.fit(X, y)
clf3 = clf3.fit(X, y)
eclf = eclf.fit(X, y)


        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html



##### 1.11.6.4. Usando o VotingClassifier com GridSearchCV

    # O VotingClassifier também pode ser usado junto com GridSearchCV para ajustar os hiperparâmetros dos estimadores individuais: 

from sklearn.model_selection import GridSearchCV
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='soft')

params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200]}

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
grid = grid.fit(iris.data, iris.target)



##### 1.11.6.5. Uso 

    # Para prever os rótulos de classe com base nas probabilidades de classe previstas (os estimadores scikit-learn no VotingClassifier devem oferecer suporte ao método predict_proba): 

eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='soft')
    
    # Opcionalmente, os pesos podem ser fornecidos para os classificadores individuais: 

eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='soft', weights=[2,5,1]
)