import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris(return_X_y=True)
X.shape, y.s

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

X_train.shape, y_train.shape
X_test.shape, y_test.shape

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_t)



########## 3.1.1. Calculando métricas de validação cruzada ##########


    # A maneira mais simples de usar a validação cruzada é chamar a função auxiliar cross_val_score no estimador e no conjunto de dados.

    # O exemplo a seguir demonstra como estimar a precisão de uma máquina de vetor de suporte de kernel linear no conjunto de dados da íris dividindo os dados, ajustando um modelo e calculando a pontuação 5 vezes consecutivas (com diferentes divisões a cada vez): 

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
scores  

    # A pontuação média e o desvio padrão são, portanto, dados por: 

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


    # Por padrão, a pontuação calculada em cada iteração do CV é o método de pontuação do estimador. É possível alterar isso usando o parâmetro de pontuação: 

from sklearn import metrics
scores = cross_val_score(
    clf, X, y, cv=5, scoring='f1_macro')
scores


    # Consulte O parâmetro de pontuação: definindo regras de avaliação de modelo para obter detalhes. No caso do conjunto de dados Iris, as amostras são balanceadas entre as classes alvo, portanto, a precisão e a pontuação F1 são quase iguais.

    # Quando o argumento cv é um inteiro, cross_val_score usa as estratégias KFold ou StratifiedKFold por padrão, sendo a última usada se o estimador deriva de ClassifierMixin.

    # Também é possível usar outras estratégias de validação cruzada, passando um iterador de validação cruzada, por exemplo: 

from sklearn.model_selection import ShuffleSplit
n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(clf, X, y, cv=cv)

    # Outra opção é usar divisões de rendimento iteráveis (treinar, testar) como matrizes de índices, por exemplo: 

def custom_cv_2folds(X):
    n = X.shape[0]
    i = 1
    while i <= 2:
        idx = np.arange(n * (i - 1) / 2, n * i / 2, dtype=int)
        yield idx, idx
        i += 1

custom_cv = custom_cv_2folds(X)
cross_val_score(clf, X, y, cv=custom_cv)


    # Transformação de dados com dados retidos

    # Assim como é importante testar um preditor em dados retidos de treinamento, pré-processamento (como padronização, seleção de recursos, etc.) e transformações de dados semelhantes devem ser aprendidos de um conjunto de treinamento e aplicados a dados retidos para predição : 

from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
clf.score(X_test_transformed, y_test)

    # Um Pipeline torna mais fácil compor estimadores, fornecendo este comportamento na validação cruzada: 

from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
cross_val_score(clf, X, y, cv=cv)



##### 3.1.1.1. A função cross_validate e avaliação de múltiplas métricas 

    # A função cross_validate difere de cross_val_score de duas maneiras: 

        # Ele permite especificar várias métricas para avaliação.

        # Ele retorna um dicionário contendo tempos de ajuste, tempos de pontuação (e, opcionalmente, pontuações de treinamento, bem como estimadores ajustados) além da pontuação do teste. 

    # Para avaliação de métrica única, onde o parâmetro de pontuação é uma string, chamável ou Nenhum, as chaves serão - ['test_score', 'fit_time', 'score_time']

    # E para avaliação de múltiplas métricas, o valor de retorno é um dict com as seguintes chaves - ['test_ <scorer1_name>', 'test_ <scorer2_name>', 'test_ <scorer ...>', 'fit_time', 'score_time']

    # return_train_score é definido como False por padrão para economizar tempo de computação. Para avaliar também as pontuações no conjunto de treinamento, você precisa defini-lo como Verdadeiro.

    # Você também pode reter o estimador ajustado em cada conjunto de treinamento, definindo return_estimator = True.

    # As várias métricas podem ser especificadas como uma lista, tupla ou conjunto de nomes de artilheiro predefinidos: 

from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
scoring = ['precision_macro', 'recall_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring)
sorted(scores.keys())
['fit_time', 'score_time', 'test_precision_macro', 'test_recall_macro']
scores['test_recall_macro']

    # Aqui está um exemplo de cross_validate usando uma única métrica: 

scores = cross_validate(clf, X, y,
                        scoring='precision_macro', cv=5,
                        return_estimator=True)
sorted(scores.keys())
['estimator', 'fit_time', 'score_time', 'test_score']

##### 3.1.1.2. Obtenção de previsões por validação cruzada 



    # A função cross_val_predict tem uma interface semelhante a cross_val_score, mas retorna, para cada elemento na entrada, a previsão que foi obtida para aquele elemento quando ele estava no conjunto de teste. Apenas estratégias de validação cruzada que atribuem todos os elementos a um conjunto de teste exatamente uma vez podem ser usadas (caso contrário, uma exceção é levantada).

    # Aviso: Nota sobre o uso impróprio de cross_val_predict
    # O resultado de cross_val_predict pode ser diferente daqueles obtidos usando cross_val_score, pois os elementos são agrupados de maneiras diferentes. A função cross_val_score obtém uma média sobre as dobras de validação cruzada, enquanto cross_val_predict simplesmente retorna os rótulos (ou probabilidades) de vários modelos distintos indistintos. Portanto, cross_val_predict não é uma medida apropriada de erro de generalização. 

    # A função cross_val_predict é apropriada para:
        # Visualização de previsões obtidas a partir de diferentes modelos.

        # Combinação de modelos: quando as previsões de um estimador supervisionado são usadas para treinar outro estimador em métodos de ensemble.

    # Os iteradores de validação cruzada disponíveis são apresentados na seção a seguir. 




    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py

    ## https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py

    ## https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py

    ## https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py

    ## https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html#sphx-glr-auto-examples-model-selection-plot-cv-predict-py

    ## https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py
    