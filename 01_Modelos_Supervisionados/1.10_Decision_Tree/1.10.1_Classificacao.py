########## 1.10.1 Classificação ##########

    # DecisionTreeClassifier é uma classe capaz de realizar classificação multiclasse em um conjunto de dados.

    # Tal como acontece com outros classificadores, DecisionTreeClassifier leva como entrada duas matrizes: uma matriz X, esparsa ou densa, de forma (n_samples, n_features) contendo as amostras de treinamento e uma matriz Y de valores inteiros, forma (n_samples,), contendo os rótulos de classe para os exemplos de treinamento: 

from sklearn import tree
X = [[0,0], [1,1]]
Y = [0,1]
clf = tree.DecisionTreeClassifier()
print(clf.fit(X,Y))

    # Depois de ser ajustado, o modelo pode ser usado para prever a classe de amostras: 

clf.predict([[2.,2.]])

    # Caso existam várias classes com a mesma e maior probabilidade, o classificador irá prever a classe com o menor índice entre essas classes. 

    # Como alternativa à saída de uma classe específica, a probabilidade de cada classe pode ser prevista, que é a fração das amostras de treinamento da classe em uma folha: 

clf.predict_proba([[2.,2.]])

    # DecisionTreeClassifier é capaz de classificação binária (onde os rótulos são [-1, 1]) e multiclasse (onde os rótulos são [0, ..., K-1]).

    # Usando o conjunto de dados Iris, podemos construir uma árvore da seguinte maneira: 

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

    # Uma vez treinado, você pode plotar a árvore com a função plot_tree: 

tree.plot_tree(clf)

    # Também podemos exportar a árvore no formato Graphviz usando o exportador export_graphviz. Se você usar o gerenciador de pacotes conda, os binários graphviz e o pacote python podem ser instalados com conda install python-graphviz.

    # Alternativamente, os binários para o graphviz podem ser baixados da página inicial do projeto graphviz, e o wrapper Python instalado a partir do pypi com pip install graphviz.

    # Abaixo está um exemplo de exportação Graphviz da árvore acima treinada em todo o conjunto de dados da íris; os resultados são salvos em um arquivo de saída iris.pdf: 

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(fot_data)
graph.render('iris')

    # O exportador export_graphviz também oferece suporte a uma variedade de opções estéticas, incluindo nós de cor por sua classe (ou valor para regressão) e usando variáveis explícitas e nomes de classe, se desejado. Os cadernos Jupyter também renderizam esses gráficos embutidos automaticamente:

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names,  class_names=iris.target_names,  filled=True, rounded=True,  special_characters=True)  

graph = graphviz.Source(dot_data)  
graph 

    # Alternativamente, a árvore também pode ser exportada em formato textual com a função export_text. Este método não requer a instalação de bibliotecas externas e é mais compacto: 

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
iris = load_iris()
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(iris.data, iris.target)
r = export_text(decision_tree, feature_names=iris['feature_names'])
print(r)

    ## Exemplos

    ## https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html#sphx-glr-auto-examples-tree-plot-iris-dtc-py

    ## https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py