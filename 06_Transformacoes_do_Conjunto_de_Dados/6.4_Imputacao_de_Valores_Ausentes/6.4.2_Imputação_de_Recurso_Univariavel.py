########## 6.4.2. Imputação de recurso univariável ##########


    # A classe SimpleImputer fornece estratégias básicas para imputar valores ausentes. Os valores omissos podem ser imputados com um valor constante fornecido ou usando as estatísticas (média, mediana ou mais frequente) de cada coluna na qual os valores omissos estão localizados. Essa classe também permite diferentes codificações de valores ausentes.

    # O snippet a seguir demonstra como substituir valores ausentes, codificados como np.nan, usando o valor médio das colunas (eixo 0) que contêm os valores ausentes: 


import numpy as np
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit([[1, 2], [np.nan, 3], [7, 6]])

X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))

    # A classe SimpleImputer também suporta matrizes esparsas: 

import scipy.sparse as sp
X = sp.csc_matrix([[1, 2], [0, -1], [8, 4]])
imp = SimpleImputer(missing_values=-1, strategy='mean')
imp.fit(X)
SimpleImputer(missing_values=-1)
X_test = sp.csc_matrix([[-1, 2], [6, -1], [7, 6]])
print(imp.transform(X_test).toarray())



    # Observe que esse formato não deve ser usado para armazenar implicitamente valores ausentes na matriz porque a densificaria no momento da transformação. Os valores ausentes codificados por 0 devem ser usados com entrada densa.

    # A classe SimpleImputer também suporta dados categóricos representados como valores de string ou categóricos pandas ao usar a estratégia 'most_frequent' ou 'constant': 

import pandas as pd
df = pd.DataFrame([["a", "x"],
                   [np.nan, "y"],
                   ["a", np.nan],
                   ["b", "y"]], dtype="category")

imp = SimpleImputer(strategy="most_frequent")
print(imp.fit_transform(df))