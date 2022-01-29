########## 6.1.4. ColumnTransformer para dados heterogêneos ##########




    # Muitos conjuntos de dados contêm recursos de diferentes tipos, como texto, floats e datas, onde cada tipo de recurso requer etapas separadas de pré-processamento ou extração de recursos. Muitas vezes é mais fácil pré-processar os dados antes de aplicar os métodos scikit-learn, por exemplo, usando pandas. Processar seus dados antes de passá-los para o scikit-learn pode ser problemático por um dos seguintes motivos:

        # 1 - A incorporação de estatísticas de dados de teste nos pré-processadores torna as pontuações de validação cruzada não confiáveis ​​(conhecidas como vazamento de dados), por exemplo, no caso de scalers ou imputação de valores ausentes.

        # 2 - Você pode querer incluir os parâmetros dos pré-processadores em uma pesquisa de parâmetros.

    # O ColumnTransformer ajuda a realizar diferentes transformações para diferentes colunas de dados, dentro de um Pipeline que é seguro contra vazamento de dados e que pode ser parametrizado. ColumnTransformer funciona em matrizes, matrizes esparsas e DataFrames de pandas.

    # Para cada coluna, uma transformação diferente pode ser aplicada, como pré-processamento ou um método específico de extração de recursos: 


import pandas as pd
X = pd.DataFrame(
    {'city': ['London', 'London', 'Paris', 'Sallisaw'],
     'title': ["His Last Bow", "How Watson Learned the Trick",
               "A Moveable Feast", "The Grapes of Wrath"],
     'expert_rating': [5, 3, 4, 5],
     'user_rating': [4, 5, 4, 3]})

    
    # Para esses dados, podemos querer codificar a coluna 'city' como uma variável categórica usando OneHotEncoder, mas aplicar um CountVectorizer à coluna 'title'. Como podemos usar vários métodos de extração de recursos na mesma coluna, damos a cada transformador um nome exclusivo, digamos 'city_category' e 'title_bow'. Por padrão, as colunas de classificação restantes são ignoradas (remainder='drop'): 

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
column_trans = ColumnTransformer(
    [('categories', OneHotEncoder(dtype='int'), ['city']),
     ('title_bow', CountVectorizer(), 'title')],
    remainder='drop', verbose_feature_names_out=False)

column_trans.fit(X)
ColumnTransformer(transformers=[('categories', OneHotEncoder(dtype='int'),
                                 ['city']),
                                ('title_bow', CountVectorizer(), 'title')],
                  verbose_feature_names_out=False)

column_trans.get_feature_names_out()

column_trans.transform(X).toarray()


    # No exemplo acima, o CountVectorizer espera uma matriz 1D como entrada e, portanto, as colunas foram especificadas como uma string ('título'). No entanto, o OneHotEncoder, como a maioria dos outros transformadores, espera dados 2D, portanto, nesse caso, você precisa especificar a coluna como uma lista de strings (['city']).

    # Além de uma lista escalar ou de um único item, a seleção de coluna pode ser especificada como uma lista de vários itens, uma matriz de inteiros, uma fatia, uma máscara booleana ou com um make_column_selector. O make_column_selector é usado para selecionar colunas com base no tipo de dados ou no nome da coluna: 


from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector
ct = ColumnTransformer([
      ('scale', StandardScaler(),
      make_column_selector(dtype_include=np.number)),
      ('onehot',
      OneHotEncoder(),
      make_column_selector(pattern='city', dtype_include=object))])
ct.fit_transform(X)


    # Strings podem referenciar colunas se a entrada for um DataFrame, inteiros são sempre interpretados como colunas posicionais.

    # Podemos manter as colunas de classificação restantes definindo resto='passthrough'. Os valores são anexados ao final da transformação: 

column_trans = ColumnTransformer(
    [('city_category', OneHotEncoder(dtype='int'),['city']),
     ('title_bow', CountVectorizer(), 'title')],
    remainder='passthrough')

column_trans.fit_transform(X)

    # O parâmetro restante pode ser definido como um estimador para transformar as colunas de classificação restantes. Os valores transformados são anexados ao final da transformação: 

from sklearn.preprocessing import MinMaxScaler
column_trans = ColumnTransformer(
    [('city_category', OneHotEncoder(), ['city']),
     ('title_bow', CountVectorizer(), 'title')],
    remainder=MinMaxScaler())

column_trans.fit_transform(X)[:, -2:]

    # A função make_column_transformer está disponível para criar mais facilmente um objeto ColumnTransformer. Especificamente, os nomes serão dados automaticamente. O equivalente para o exemplo acima seria: 

from sklearn.compose import make_column_transformer
column_trans = make_column_transformer(
    (OneHotEncoder(), ['city']),
    (CountVectorizer(), 'title'),
    remainder=MinMaxScaler())
column_trans
ColumnTransformer(remainder=MinMaxScaler(),
                  transformers=[('onehotencoder', OneHotEncoder(), ['city']),
                                ('countvectorizer', CountVectorizer(),
                                 'title')])

    # Se ColumnTransformer estiver equipado com um dataframe e o dataframe tiver apenas nomes de coluna de string, a transformação de um dataframe usará os nomes das colunas para selecionar as colunas: 


ct = ColumnTransformer(
         [("scale", StandardScaler(), ["expert_rating"])]).fit(X)
X_new = pd.DataFrame({"expert_rating": [5, 6, 1],
                      "ignored_new_col": [1.2, 0.3, -0.1]})
ct.transform(X_new)