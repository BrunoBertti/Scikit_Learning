import numpy as np

########## 6.4.6. Marcando valores imputados  ##########



    # O transformador MissingIndicator é útil para transformar um conjunto de dados na matriz binária correspondente, indicando a presença de valores ausentes no conjunto de dados. Essa transformação é útil em conjunto com a imputação. Ao usar a imputação, preservar as informações sobre quais valores estavam faltando pode ser informativo. Observe que tanto o SimpleImputer quanto o IterativeImputer têm o parâmetro booleano add_indicator (False por padrão) que, quando definido como True, fornece uma maneira conveniente de empilhar a saída do transformador MissingIndicator com a saída do imputador.

    # NaN geralmente é usado como espaço reservado para valores ausentes. No entanto, ele impõe que o tipo de dados seja float. O parâmetro missing_values permite especificar outro placeholder, como integer. No exemplo a seguir, usaremos -1 como valores ausentes: 


from sklearn.impute import MissingIndicator
X = np.array([[-1, -1, 1, 3],
              [4, -1, 0, -1],
              [8, -1, 1, 0]])
indicator = MissingIndicator(missing_values=-1)
mask_missing_values_only = indicator.fit_transform(X)
mask_missing_values_only


    #   O parâmetro features é usado para escolher as features para as quais a máscara é construída. Por padrão, é 'missing-only' que retorna a máscara de imputação dos recursos que contêm valores ausentes no tempo de ajuste:


indicator.features_



    # O parâmetro features pode ser definido como 'all' para retornar todos os recursos, independentemente de conterem ou não valores ausentes: 

indicator = MissingIndicator(missing_values=-1, features="all")
mask_all = indicator.fit_transform(X)
mask_all

indicator.features_


    # Ao usar o MissingIndicator em um pipeline, certifique-se de usar o FeatureUnion ou ColumnTransformer para adicionar os recursos do indicador aos recursos regulares. Primeiro, obtemos o conjunto de dados da íris e adicionamos alguns valores ausentes a ele.

from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.tree import DecisionTreeClassifier
X, y = load_iris(return_X_y=True)
mask = np.random.randint(0, 2, size=X.shape).astype(bool)
X[mask] = np.nan
X_train, X_test, y_train, _ = train_test_split(X, y, test_size=100,
                                               random_state=0)


    # Agora criamos um FeatureUnion. Todos os recursos serão imputados usando o SimpleImputer, para permitir que os classificadores trabalhem com esses dados. Além disso, adiciona as variáveis indicadoras de MissingIndicator. 


transformer = FeatureUnion(
    transformer_list=[
        ('features', SimpleImputer(strategy='mean')),
        ('indicators', MissingIndicator())])
transformer = transformer.fit(X_train, y_train)
results = transformer.transform(X_test)
results.shape



    # Claro, não podemos usar o transformador para fazer quaisquer previsões. Devemos envolver isso em um Pipeline com um classificador (por exemplo, um DecisionTreeClassifier) para poder fazer previsões. 

clf = make_pipeline(transformer, DecisionTreeClassifier())
clf = clf.fit(X_train, y_train)
results = clf.predict(X_test)
results.shape