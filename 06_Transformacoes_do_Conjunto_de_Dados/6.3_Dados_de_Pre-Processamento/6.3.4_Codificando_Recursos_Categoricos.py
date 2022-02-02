from sklearn import preprocessing
import numpy as np


########## 6.3.4. Codificando recursos categóricos ##########



    # Muitas vezes, os recursos não são dados como valores contínuos, mas categóricos. Por exemplo, uma pessoa pode ter recursos ["masculino", "feminino"], ["da Europa", "dos EUA", "da Ásia"], ["usa Firefox", "usa Chrome", "usa Safari", "usa o Internet Explorer"]. Esses recursos podem ser codificados de forma eficiente como números inteiros, por exemplo, ["male", "from US", "usa Internet Explorer"] pode ser expresso como [0, 1, 3] enquanto ["female", "from Asia", " usa o Chrome"] seria [1, 2, 1].

    # Para converter recursos categóricos para esses códigos inteiros, podemos usar o OrdinalEncoder. Este estimador transforma cada recurso categórico em um novo recurso de inteiros (0 a n_categories - 1): 

enc = preprocessing.OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)

enc.transform([['female', 'from US', 'uses Safari']])

    # Essa representação inteira pode, no entanto, não ser usada diretamente com todos os estimadores scikit-learn, pois eles esperam entrada contínua e interpretariam as categorias como ordenadas, o que geralmente não é desejado (ou seja, o conjunto de navegadores foi ordenado arbitrariamente).

    # O OrdinalEncoder também passará os valores ausentes indicados por np.nan. 


enc = preprocessing.OrdinalEncoder()
X = [['male'], ['female'], [np.nan], ['female']]
enc.fit_transform(X)

    # Outra possibilidade de converter recursos categóricos em recursos que podem ser usados com estimadores scikit-learn é usar um de K, também conhecido como codificação one-hot ou dummy. Esse tipo de codificação pode ser obtido com o OneHotEncoder, que transforma cada recurso categórico com n_categories valores possíveis em n_categories recursos binários, sendo um deles 1 e todos os outros 0.

    # Continuando o exemplo acima: 


enc = preprocessing.OneHotEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)

enc.transform([['female', 'from US', 'uses Safari'],
               ['male', 'from Europe', 'uses Safari']]).toarray()

    # Por padrão, os valores que cada recurso pode assumir são inferidos automaticamente do conjunto de dados e podem ser encontrados no atributo Categories_: 

enc.categories_

    # É possível especificar isso explicitamente usando as categorias de parâmetros. Existem dois gêneros, quatro continentes possíveis e quatro navegadores da web em nosso conjunto de dados: 

genders = ['female', 'male']
locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']
enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])
# Observe que para existem valores categóricos ausentes para o 2º e 3º
# característica 
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)

enc.transform([['female', 'from Asia', 'uses Chrome']]).toarray()

    # Se houver a possibilidade de que os dados de treinamento possam ter recursos categóricos ausentes, muitas vezes pode ser melhor especificar handle_unknown='ignore' em vez de definir as categorias manualmente como acima. Quando handle_unknown='ignore' for especificado e categorias desconhecidas forem encontradas durante a transformação, nenhum erro será gerado, mas as colunas codificadas one-hot resultantes para este recurso serão todas zeros (handle_unknown='ignore' só é suportado para codificação one-hot ): 

enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)

enc.transform([['female', 'from Asia', 'uses Chrome']]).toarray()

    # Também é possível codificar cada coluna em n_categories - 1 colunas em vez de colunas n_categories usando o parâmetro drop. Este parâmetro permite que o usuário especifique uma categoria para cada recurso a ser descartado. Isso é útil para evitar colinearidade na matriz de entrada em alguns classificadores. Essa funcionalidade é útil, por exemplo, ao usar regressão não regularizada (LinearRegression), pois a colinearidade faria com que a matriz de covariâncias fosse não inversível: 

X = [['male', 'from US', 'uses Safari'],
     ['female', 'from Europe', 'uses Firefox']]
drop_enc = preprocessing.OneHotEncoder(drop='first').fit(X)
drop_enc.categories_

drop_enc.transform(X).toarray()

    # Pode-se querer descartar uma das duas colunas apenas para recursos com 2 categorias. Neste caso, você pode definir o parâmetro drop='if_binary'. 

X = [['male', 'US', 'Safari'],
     ['female', 'Europe', 'Firefox'],
     ['female', 'Asia', 'Chrome']]
drop_enc = preprocessing.OneHotEncoder(drop='if_binary').fit(X)
drop_enc.categories_

drop_enc.transform(X).toarray()

    # No X transformado, a primeira coluna é a codificação do recurso com categorias “masculino”/”feminino”, enquanto as 6 colunas restantes são a codificação dos 2 recursos com respectivamente 3 categorias cada.

    # Quando handle_unknown='ignore' e drop não for None, as categorias desconhecidas serão codificadas como zeros: 

drop_enc = preprocessing.OneHotEncoder(drop='first',
                                       handle_unknown='ignore').fit(X)
X_test = [['unknown', 'America', 'IE']]
drop_enc.transform(X_test).toarray()

    # Todas as categorias em X_test são desconhecidas durante a transformação e serão mapeadas para todos os zeros. Isso significa que as categorias desconhecidas terão o mesmo mapeamento que a categoria descartada. :meth`OneHotEncoder.inverse_transform` mapeará todos os zeros para a categoria descartada se uma categoria for descartada e Nenhum se uma categoria não for descartada: 

drop_enc = preprocessing.OneHotEncoder(drop='if_binary', sparse=False,
                                       handle_unknown='ignore').fit(X)
X_test = [['unknown', 'America', 'IE']]
X_trans = drop_enc.transform(X_test)
X_trans

drop_enc.inverse_transform(X_trans)

    # O OneHotEncoder oferece suporte a recursos categóricos com valores ausentes, considerando os valores ausentes como uma categoria adicional: 

X = [['male', 'Safari'],
     ['female', None],
     [np.nan, 'Firefox']]
enc = preprocessing.OneHotEncoder(handle_unknown='error').fit(X)
enc.categories_

enc.transform(X).toarray()

    # Se um recurso contiver np.nan e None, eles serão considerados categorias separadas: 

X = [['Safari'], [None], [np.nan], ['Firefox']]
enc = preprocessing.OneHotEncoder(handle_unknown='error').fit(X)
enc.categories_

enc.transform(X).toarray()

    # Consulte Carregando recursos de dicts para recursos categóricos que são representados como um dict, não como escalares. 