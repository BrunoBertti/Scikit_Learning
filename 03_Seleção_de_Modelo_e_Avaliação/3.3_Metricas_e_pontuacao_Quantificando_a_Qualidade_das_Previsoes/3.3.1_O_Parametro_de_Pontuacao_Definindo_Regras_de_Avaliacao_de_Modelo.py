########## 3.3.1. O parâmetro de pontuação: definindo regras de avaliação de modelo ##########

    # A seleção e avaliação do modelo usando ferramentas, como model_selection.GridSearchCV e model_selection.cross_val_score, usam um parâmetro de pontuação que controla qual métrica eles aplicam aos estimadores avaliados. 


##### 3.3.1.1. Casos comuns: valores predefinidos

    # Para os casos de uso mais comuns, você pode designar um objeto scorer com o parâmetro scoring; a tabela abaixo mostra todos os valores possíveis. Todos os objetos de pontuação seguem a convenção de que valores de retorno mais altos são melhores do que valores de retorno mais baixos. Assim, as métricas que medem a distância entre o modelo e os dados, como metric.mean_squared_error, estão disponíveis como neg_mean_squared_error, que retornam o valor negado da métrica. 

                                #########################
                                #####    TABELA     #####
                                #####    TABELA     #####
                                #########################


    # Exemplos de uso:
     
      
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
X, y = datasets.load_iris(return_X_y=True)
clf = svm.SVC(random_state=0)
cross_val_score(clf, X, y, cv=5, scoring='recall_macro')
model = svm.SVC()
cross_val_score(model, X, y, cv=5, scoring='wrong_choice')


    # Observação: os valores listados pela exceção ValueError correspondem às funções que medem a precisão da previsão descritas nas seções a seguir. Os objetos de pontuação para essas funções são armazenados no dicionário sklearn.metrics.SCORERS. 

##### 3.3.1.2. Definindo sua estratégia de pontuação a partir de funções de métrica

    # O módulo sklearn.metrics também expõe um conjunto de funções simples que medem um erro de previsão dada a verdade e a previsão:

        # funções que terminam com _score retornam um valor para maximizar, quanto maior, melhor.

        # funções que terminam com _error ou _loss retornam um valor para minimizar, quanto menor, melhor. Ao converter em um objeto scorer usando make_scorer, defina o parâmetro large_is_better como False (True por padrão; veja a descrição do parâmetro abaixo).

    # As métricas disponíveis para várias tarefas de aprendizado de máquina são detalhadas nas seções abaixo.

    # Muitas métricas não recebem nomes para serem usadas como valores de pontuação, às vezes porque exigem parâmetros adicionais, como fbeta_score. Nesses casos, você precisa gerar um objeto de pontuação apropriado. A maneira mais simples de gerar um objeto que pode ser chamado para pontuação é usando make_scorer. Essa função converte métricas em callables que podem ser usados ​​para avaliação de modelo.

    # Um caso de uso típico é envolver uma função de métrica existente da biblioteca com valores não padrão para seus parâmetros, como o parâmetro beta para a função fbeta_score:



from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
                    scoring=ftwo_scorer, cv=5)


    # O segundo caso de uso é construir um objeto de pontuação completamente personalizado a partir de uma função python simples usando make_scorer, que pode receber vários parâmetros:

        # a função python que você deseja usar (my_custom_loss_func no exemplo abaixo)

        # se a função python retorna uma pontuação (greater_is_better=True, o padrão) ou uma perda (greater_is_better=False). Se houver perda, a saída da função python é negada pelo objeto scorer, de acordo com a convenção de validação cruzada de que os scorers retornam valores mais altos para modelos melhores.

        # apenas para métricas de classificação: se a função python que você forneceu requer certezas de decisão contínuas (needs_threshold=True). O valor padrão é falso.

        # quaisquer parâmetros adicionais, como beta ou rótulos em f1_score.

    # Aqui está um exemplo de criação de pontuadores personalizados e de uso do parâmetro large_is_better: 

import numpy as np
def my_custom_loss_func(y_true, y_pred):
    diff = np.abs(y_true - y_pred).max()
    return np.log1p(diff)
# score irá negar o valor de retorno de my_custom_loss_func,
# que será np.log(2), 0,693, dados os valores de X
# e y definidos abaixo. 
score = make_scorer(my_custom_loss_func, greater_is_better=False)
X = [[1], [1]]
y = [0, 1]
from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy='most_frequent', random_state=0)
clf = clf.fit(X, y)
my_custom_loss_func(y, clf.predict(X))
score(clf, X, y)




##### 3.3.1.3. Implementando seu próprio objeto de pontuação

    # Você pode gerar pontuadores de modelo ainda mais flexíveis construindo seu próprio objeto de pontuação do zero, sem usar a fábrica make_scorer. Para um callable ser um pontuador, ele precisa atender ao protocolo especificado pelas duas regras a seguir: 

        # Ele pode ser chamado com parâmetros (estimador, X, y), onde estimador é o modelo que deve ser avaliado, X são dados de validação e y é o alvo de verdade para X (no caso supervisionado) ou Nenhum (no caso não supervisionado). caso).

        # Ele retorna um número de ponto flutuante que quantifica a qualidade da previsão do estimador em X, com referência a y. Novamente, por convenção, números mais altos são melhores, portanto, se seu pontuador retornar perda, esse valor deve ser negado.    

    # Nota: Usando pontuadores personalizados em funções onde n_jobs > 1
    # Embora a definição da função de pontuação personalizada juntamente com a função de chamada deva funcionar imediatamente com o backend joblib padrão (loky), importá-la de outro módulo será uma abordagem mais robusta e funcionará independentemente do backend joblib.

    # Por exemplo, para usar n_jobs maior que 1 no exemplo abaixo, a função custom_scoring_function é salva em um módulo criado pelo usuário (custom_scorer_module.py) e importada: 

from custom_scorer_module import custom_scoring_function 
cross_val_score(model, X_train, y_train,
scoring=make_scorer(custom_scoring_function, greater_is_better=False), cv=5, n_jobs=-1) 






##### 3.3.1.4. Como usar a avaliação de várias métricas 

    # O Scikit-learn também permite a avaliação de várias métricas em GridSearchCV, RandomizedSearchCV e cross_validate.

    # Há três maneiras de especificar várias métricas de pontuação para o parâmetro de pontuação: 


        # Como um iterável de métricas de string::

scoring = ['accuracy', 'precision']

        # Como um dict mapeando o nome do pontuador para a função de pontuação::



from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
scoring = {'accuracy': make_scorer(accuracy_score),
           'prec': 'precision'}



        # Observe que os valores dict podem ser funções de pontuação ou uma das sequências de métricas predefinidas.

        # Como um callable que retorna um dicionário de pontuações: 


from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
# Um conjunto de dados de classificação binária de brinquedo de amostra 
X, y = datasets.make_classification(n_classes=2, random_state=0)
svm = LinearSVC(random_state=0)
def confusion_matrix_scorer(clf, X, y):
     y_pred = clf.predict(X)
     cm = confusion_matrix(y, y_pred)
     return {'tn': cm[0, 0], 'fp': cm[0, 1],
             'fn': cm[1, 0], 'tp': cm[1, 1]}
cv_results = cross_validate(svm, X, y, cv=5,
                            scoring=confusion_matrix_scorer)
# Getting the test set true positive scores
print(cv_results['test_tp'])
# Getting the test set false negative scores
print(cv_results['test_fn'])