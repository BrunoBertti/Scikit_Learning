##########  6.2.1. Carregando recursos de dicts ##########


    # A classe DictVectorizer pode ser usada para converter arrays de recursos representados como listas de objetos dict padrão do Python para a representação NumPy/SciPy usada pelos estimadores do scikit-learn.

    # Embora não seja particularmente rápido de processar, o dict do Python tem as vantagens de ser conveniente de usar, ser esparso (recursos ausentes não precisam ser armazenados) e armazenar nomes de recursos além de valores.

    # DictVectorizer implementa o que é chamado de codificação one-of-K ou “one-hot” para recursos categóricos (também conhecidos como nominais, discretos). Características categóricas são pares “atributo-valor” onde o valor é restrito a uma lista de possibilidades discretas sem ordenação (por exemplo, identificadores de tópicos, tipos de objetos, tags, nomes…).

    # A seguir, “cidade” é um atributo categórico enquanto “temperatura” é uma característica numérica tradicional: 


measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Francisco', 'temperature': 18.},
]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()

vec.fit_transform(measurements).toarray()

vec.get_feature_names_out()


    # DictVectorizer aceita vários valores de string para um recurso, como, por exemplo, várias categorias para um filme.

    # Suponha que um banco de dados classifique cada filme usando algumas categorias (não obrigatórias) e seu ano de lançamento. 

movie_entry = [{'category': ['thriller', 'drama'], 'year': 2003},
               {'category': ['animation', 'family'], 'year': 2011},
               {'year': 1974}]
vec.fit_transform(movie_entry).toarray()

vec.get_feature_names_out()

vec.transform({'category': ['thriller'],
               'unseen_feature': '3'}).toarray()



    # DictVectorizer também é uma transformação de representação útil para classificadores de sequência de treinamento em modelos de Processamento de Linguagem Natural que normalmente funcionam extraindo janelas de recursos em torno de uma palavra de interesse específica.

    # Por exemplo, suponha que temos um primeiro algoritmo que extrai tags Part of Speech (PoS) que queremos usar como tags complementares para treinar um classificador de sequência (por exemplo, um chunker). O seguinte dict poderia ser uma janela de características extraídas em torno da palavra 'sat' na frase 'The cat sat on the mat'.':

pos_window = [
    {
        'word-2': 'the',
        'pos-2': 'DT',
        'word-1': 'cat',
        'pos-1': 'NN',
        'word+1': 'on',
        'pos+1': 'PP',
    },
    # em uma aplicação real seria possível extrair muitos desses dicionários 
]               



    # Esta descrição pode ser vetorizada em uma matriz bidimensional esparsa adequada para alimentar um classificador (talvez depois de ser canalizado para um TfidfTransformer para normalização): 

vec = DictVectorizer()
pos_vectorized = vec.fit_transform(pos_window)
pos_vectorized

pos_vectorized.toarray()

vec.get_feature_names_out()


    # Como você pode imaginar, se extrairmos tal contexto em torno de cada palavra individual de um corpus de documentos, a matriz resultante será muito ampla (muitas características únicas) com a maioria delas sendo valorizadas como zero na maioria das vezes. Para tornar a estrutura de dados resultante capaz de caber na memória, a classe DictVectorizer usa uma matriz scipy.sparse por padrão em vez de um numpy.ndarray. 