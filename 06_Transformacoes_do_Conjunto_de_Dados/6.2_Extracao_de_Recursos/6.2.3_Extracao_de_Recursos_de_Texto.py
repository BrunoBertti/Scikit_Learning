##########  6.2.3. Extração de recursos de texto ##########




##### 6.2.3.1. A representação do saco de palavras

    # A Análise de Texto é um importante campo de aplicação para algoritmos de aprendizado de máquina. No entanto, os dados brutos, uma sequência de símbolos não podem ser alimentados diretamente para os próprios algoritmos, pois a maioria deles espera vetores de recursos numéricos com tamanho fixo, em vez de documentos de texto bruto com comprimento variável.

    # Para resolver isso, o scikit-learn fornece utilitários para as formas mais comuns de extrair recursos numéricos do conteúdo de texto, a saber:

        # tokenizando (tokenizing) strings e fornecendo um id inteiro para cada token possível, por exemplo, usando espaços em branco e pontuação como separadores de token.

        # contando (counting) as ocorrências de tokens em cada documento.

        # normalizando (normalizing) e ponderando com tokens de importância decrescente que ocorrem na maioria das amostras/documentos.

    # Nesse esquema, os recursos e as amostras são definidos da seguinte forma:

        # cada frequência de ocorrência de token individual (normalizada ou não) é tratada como um recurso.

        # o vetor de todas as frequências de token para um determinado documento é considerado uma amostra multivariada.

    # Um corpus de documentos pode assim ser representado por uma matriz com uma linha por documento e uma coluna por token (por exemplo, palavra) ocorrendo no corpus.

    # Chamamos de vetorização o processo geral de transformar uma coleção de documentos de texto em vetores de recursos numéricos. Essa estratégia específica (tokenização, contagem e normalização) é chamada de representação Bag of Words ou “Bag of n-grams”. Os documentos são descritos por ocorrências de palavras, ignorando completamente as informações de posição relativa das palavras no documento. 




##### 6.2.3.2. Espasidade

    # Como a maioria dos documentos normalmente usa um subconjunto muito pequeno das palavras usadas no corpus, a matriz resultante terá muitos valores de recursos que são zeros (geralmente mais de 99% deles).

    # Por exemplo, uma coleção de 10.000 documentos de texto curto (como e-mails) usará um vocabulário com um tamanho da ordem de 100.000 palavras únicas no total, enquanto cada documento usará de 100 a 1.000 palavras únicas individualmente.

    # Para poder armazenar tal matriz na memória, mas também para acelerar as operações algébricas de matriz/vetor, as implementações normalmente usarão uma representação esparsa, como as implementações disponíveis no pacote scipy.sparse. 





##### 6.2.3.3. Uso comum do vetorizador

    # CountVectorizer implementa tokenização e contagem de ocorrências em uma única classe: 

from sklearn.feature_extraction.text import CountVectorizer

    # Este modelo possui muitos parâmetros, porém os valores padrão são bastante razoáveis (consulte a documentação de referência para obter detalhes): 

vectorizer = CountVectorizer()
vectorizer

    # Vamos usá-lo para tokenizar e contar as ocorrências de palavras de um corpus minimalista de documentos de texto: 

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)
X

    # A configuração padrão tokeniza a string extraindo palavras de pelo menos 2 letras. A função específica que faz esta etapa pode ser solicitada explicitamente: 

analyze = vectorizer.build_analyzer()
analyze("This is a text document to analyze.") == (
    ['this', 'is', 'text', 'document', 'to', 'analyze'])

    # Cada termo encontrado pelo analisador durante o ajuste recebe um índice inteiro único correspondente a uma coluna na matriz resultante. Essa interpretação das colunas pode ser recuperada da seguinte forma: 

vectorizer.get_feature_names_out()

X.toarray()

    # O mapeamento inverso do nome do recurso para o índice da coluna é armazenado no atributo vocabulário_ do vetorizador: 

vectorizer.vocabulary_.get('document')

    # Portanto, palavras que não foram vistas no corpus de treinamento serão completamente ignoradas em futuras chamadas ao método transform: 

vectorizer.transform(['Something completely new.']).toarray()

    # Observe que no corpus anterior, o primeiro e o último documentos têm exatamente as mesmas palavras, portanto, são codificados em vetores iguais. Em particular, perdemos a informação de que o último documento é uma forma interrogativa. Para preservar algumas das informações de pedidos locais, podemos extrair 2-grams de palavras além dos 1-grams (palavras individuais): 

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
analyze('Bi-grams are cool!') == (
    ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])

    # O vocabulário extraído por este vetorizador é, portanto, muito maior e agora pode resolver ambiguidades codificadas em padrões de posicionamento local: 

X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
X_2

    # Em particular, a forma interrogativa “Is this” está presente apenas no último documento: 

feature_index = bigram_vectorizer.vocabulary_.get('is this')
X_2[:, feature_index]







##### 6.2.3.3.1. Usando palavras de parada (stop words)

    # Palavras de parada são palavras como “e”, “o”, “ele”, que se presume não serem informativas na representação do conteúdo de um texto e que podem ser removidas para evitar que sejam interpretadas como um sinal de previsão. Às vezes, no entanto, palavras semelhantes são úteis para a previsão, como na classificação de estilo de escrita ou personalidade.

    # Existem vários problemas conhecidos em nossa lista de palavras de parada 'inglês' fornecida. Não pretende ser uma solução geral, de "tamanho único", pois algumas tarefas podem exigir uma solução mais personalizada. Consulte [NQY18] para obter mais detalhes.

    # Por favor, tome cuidado ao escolher uma lista de palavras de parada. As listas de palavras de parada populares podem incluir palavras altamente informativas para algumas tarefas, como computador.

    # Você também deve certificar-se de que a lista de palavras de parada tenha o mesmo pré-processamento e tokenização aplicado como aquele usado no vetorizador. A palavra we've é dividida em we e ve pelo tokenizer padrão do CountVectorizer, portanto, se estivermos em stop_words, mas ve não estiver, ve será retido do texto transformado. Nossos vetorizadores tentarão identificar e alertar sobre alguns tipos de inconsistências. 





    ## Referências:

    ## J. Nothman, H. Qin and R. Yurchak (2018). “Stop Word Lists in Free Open-source Software Packages”. In Proc. Workshop for NLP Open Source Software. (https://aclweb.org/anthology/W18-2502)


##### 6.2.3.4. Ponderação do termo Tf–idf


    # Em um corpus de texto grande, algumas palavras estarão muito presentes (por exemplo, “the”, “a”, “is” em inglês), portanto, carregam muito pouca informação significativa sobre o conteúdo real do documento. Se fôssemos alimentar os dados de contagem direta diretamente para um classificador, esses termos muito frequentes sombreariam as frequências de termos mais raros, porém mais interessantes.

    # Para reponderar os recursos de contagem em valores de ponto flutuante adequados para uso por um classificador, é muito comum usar a transformada tf–idf.

    # Tf significa termo-frequência enquanto tf–idf significa termo-frequência vezes inversa do documento-frequência: \text{tf-idf(t,d)}=\text{tf(t,d)} \times \text{idf(t)}.

    # Usando as configurações padrão do TfidfTransformer, TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False) a frequência do termo, o número de vezes que um termo ocorre em um determinado documento, é multiplicado pelo componente idf, que é calculado como 


        # \text{idf}(t) = \log{\frac{1 + n}{1+\text{df}(t)}} + 1


    # onde n é o número total de documentos no conjunto de documentos e \text{df}(t) é o número de documentos no conjunto de documentos que contém o termo t. Os vetores tf-idf resultantes são então normalizados pela norma euclidiana: 

        # v_{norm} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v{_1}^2 +
        # v{_2}^2 + \dots + v{_n}^2}}

    # Este foi originalmente um esquema de ponderação de termos desenvolvido para recuperação de informações (como uma função de classificação para resultados de mecanismos de busca) que também encontrou bom uso na classificação e agrupamento de documentos.

    # As seções a seguir contêm mais explicações e exemplos que ilustram como os tf-idfs são calculados exatamente e como os tf-idfs calculados no TfidfTransformer e TfidfVectorizer do scikit-learn diferem ligeiramente da notação padrão do livro didático que define o idf como 

        # \text{idf}(t) = \log{\frac{n}{1+\text{df}(t)}}.

    # No TfidfTransformer e TfidfVectorizer com smooth_idf=False, a contagem “1” é adicionada ao idf em vez do denominador do idf: 

        # \text{idf}(t) = \log{\frac{n}{\text{df}(t)}} + 1

    # Essa normalização é implementada pela classe TfidfTransformer: 

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
transformer

    # Novamente, consulte a documentação de referência para obter detalhes sobre todos os parâmetros.

    # Vamos dar um exemplo com as seguintes contagens. O primeiro termo está presente 100% do tempo, portanto, não é muito interessante. Os outros dois recursos apenas em menos de 50% do tempo, portanto, provavelmente mais representativos do conteúdo dos documentos: 

counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]]

tfidf = transformer.fit_transform(counts)
tfidf


tfidf.toarray()

    # Cada linha é normalizada para ter a norma euclidiana unitária: 

        # v_{norm} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v{_1}^2 +
        # v{_2}^2 + \dots + v{_n}^2}}

    # Por exemplo, podemos calcular o tf-idf do primeiro termo no primeiro documento no array counts da seguinte forma: 

        # n = 6

        # \text{df}(t)_{\text{term1}} = 6

        # \text{idf}(t)_{\text{term1}} = \log \frac{n}{\text{df}(t)} + 1 = \log(1)+1 = 1

        # \text{tf-idf}_{\text{term1}} = \text{tf} \times \text{idf} = 3 \times 1 = 3

        # Agora, se repetirmos esse cálculo para os 2 termos restantes no documento, obtemos 

        # \text{tf-idf}_{\text{term2}} = 0 \times (\log(6/1)+1) = 0

        # \text{tf-idf}_{\text{term3}} = 1 \times (\log(6/2)+1) \approx 2.0986

        # e o vetor de tf-idfs brutos: 

        # \text{tf-idf}_{\text{raw}} = [3, 0, 2.0986].

        # Então, aplicando a norma euclidiana (L2), obtemos os seguintes tf-idfs para o documento 1: 

        # \frac{[3, 0, 2.0986]}{\sqrt{\big(3^2 + 0^2 + 2.0986^2\big)}}= [ 0.819,  0,  0.573].

        # Além disso, o parâmetro padrão smooth_idf=True adiciona “1” ao numerador e denominador como se um documento extra fosse visto contendo todos os termos da coleção exatamente uma vez, o que evita divisões zero: 

        # \text{idf}(t) = \log{\frac{1 + n}{1+\text{df}(t)}} + 1

        # Usando esta modificação, o tf-idf do terceiro termo no documento 1 muda para 1,8473: 

        # \text{tf-idf}_{\text{term3}} = 1 \times \log(7/3)+1 \approx 1.8473

        # E o tf-idf normalizado em L2 muda para 

        # \frac{[3, 0, 1.8473]}{\sqrt{\big(3^2 + 0^2 + 1.8473^2\big)}}= [0.8515, 0, 0.5243] 
         
transformer = TfidfTransformer()
transformer.fit_transform(counts).toarray()


    # Os pesos de cada recurso calculados pela chamada do método fit são armazenados em um atributo de modelo: 

transformer.idf_

    # Como tf–idf é muito usado para recursos de texto, existe também outra classe chamada TfidfVectorizer que combina todas as opções de CountVectorizer e TfidfTransformer em um único modelo: 

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus)

    # Embora a normalização tf–idf seja frequentemente muito útil, pode haver casos em que os marcadores de ocorrência binários possam oferecer melhores recursos. Isso pode ser feito usando o parâmetro binário de CountVectorizer. Em particular, alguns estimadores como Bernoulli Naive Bayes modelam explicitamente variáveis aleatórias booleanas discretas. Além disso, é provável que textos muito curtos tenham valores tf–idf ruidosos, enquanto as informações de ocorrência binária são mais estáveis.

    # Como de costume, a melhor maneira de ajustar os parâmetros de extração de recursos é usar uma pesquisa de grade com validação cruzada, por exemplo, canalizando o extrator de recursos com um classificador: 






    ## https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py



##### 6.2.3.5. Decodificando arquivos de texto

    # O texto é feito de caracteres, mas os arquivos são feitos de bytes. Esses bytes representam caracteres de acordo com alguma codificação. Para trabalhar com arquivos de texto em Python, seus bytes devem ser decodificados em um conjunto de caracteres chamado Unicode. As codificações comuns são ASCII, Latin-1 (Europa Ocidental), KOI8-R (russo) e as codificações universais UTF-8 e UTF-16. Muitos outros existem.

    # Nota: Uma codificação também pode ser chamada de 'conjunto de caracteres', mas esse termo é menos preciso: várias codificações podem existir para um único conjunto de caracteres.
    # Os extratores de recursos de texto no scikit-learn sabem como decodificar arquivos de texto, mas somente se você informar em qual codificação os arquivos estão. O CountVectorizer usa um parâmetro de codificação para essa finalidade. Para arquivos de texto modernos, a codificação correta provavelmente é UTF-8, que é, portanto, o padrão (encoding="utf-8").

    # Se o texto que você está carregando não estiver realmente codificado com UTF-8, no entanto, você receberá um UnicodeDecodeError. Os vetorizadores podem ser instruídos a ficarem em silêncio sobre erros de decodificação definindo o parâmetro decode_error para "ignorar" ou "substituir". Consulte a documentação da função Python bytes.decode para obter mais detalhes (digite help(bytes.decode) no prompt do Python).

    # Se você está tendo problemas para decodificar o texto, aqui estão algumas coisas para tentar: 


        # Descubra qual é a codificação real do texto. O arquivo pode vir com um cabeçalho ou README que informa a codificação, ou pode haver alguma codificação padrão que você pode assumir com base na origem do texto.

        # Você pode descobrir que tipo de codificação é em geral usando o arquivo de comando do UNIX. O módulo chardet do Python vem com um script chamado chardetect.py que adivinhará a codificação específica, embora você não possa confiar que sua suposição esteja correta.

        # Você pode tentar UTF-8 e desconsiderar os erros. Você pode decodificar strings de bytes com bytes.decode(errors='replace') para substituir todos os erros de decodificação por um caractere sem sentido, ou definir decode_error='replace' no vetorizador. Isso pode prejudicar a utilidade de seus recursos.

        # O texto real pode vir de uma variedade de fontes que podem ter usado codificações diferentes, ou até mesmo ser decodificado de forma descuidada em uma codificação diferente daquela com a qual foi codificado. Isso é comum em texto recuperado da Web. O pacote Python ftfy pode classificar automaticamente algumas classes de erros de decodificação, então você pode tentar decodificar o texto desconhecido como latin-1 e então usar ftfy para corrigir erros.

        # Se o texto estiver em uma mistura de codificações que é simplesmente muito difícil de classificar (que é o caso do conjunto de dados 20 Newsgroups), você pode recorrer a uma codificação simples de byte único, como latin-1. Alguns textos podem ser exibidos incorretamente, mas pelo menos a mesma sequência de bytes sempre representará o mesmo recurso. 



    # Por exemplo, o trecho a seguir usa chardet (não fornecido com o scikit-learn, deve ser instalado separadamente) para descobrir a codificação de três textos. Em seguida, vetoriza os textos e imprime o vocabulário aprendido. A saída não é mostrada aqui. 


import chardet    
text1 = b"Sei mir gegr\xc3\xbc\xc3\x9ft mein Sauerkraut"
text2 = b"holdselig sind deine Ger\xfcche"
text3 = b"\xff\xfeA\x00u\x00f\x00 \x00F\x00l\x00\xfc\x00g\x00e\x00l\x00n\x00 \x00d\x00e\x00s\x00 \x00G\x00e\x00s\x00a\x00n\x00g\x00e\x00s\x00,\x00 \x00H\x00e\x00r\x00z\x00l\x00i\x00e\x00b\x00c\x00h\x00e\x00n\x00,\x00 \x00t\x00r\x00a\x00g\x00 \x00i\x00c\x00h\x00 \x00d\x00i\x00c\x00h\x00 \x00f\x00o\x00r\x00t\x00"
decoded = [x.decode(chardet.detect(x)['encoding'])
            for x in (text1, text2, text3)]        
v = CountVectorizer().fit(decoded).vocabulary_    
for term in v: print(v) 

    # (Dependendo da versão do chardet, ele pode errar o primeiro.)

    # Para uma introdução ao Unicode e codificações de caracteres em geral, consulte Joel Spolsky’s Absolute Minimum Every Software Developer Must Know About Unicode. 







##### 6.2.3.6. Aplicações e exemplos

    # A representação do saco de palavras é bastante simplista, mas surpreendentemente útil na prática.

    # Em particular em uma configuração supervisionada, ele pode ser combinado com sucesso com modelos lineares rápidos e escaláveis para treinar classificadores de documentos, por exemplo: 

        # https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py

    # Em uma configuração não supervisionada, ele pode ser usado para agrupar documentos semelhantes aplicando algoritmos de agrupamento, como K-means: 

        # https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py

    # Finally it is possible to discover the main topics of a corpus by relaxing the hard assignment constraint of clustering, for instance by using Non-negative matrix factorization (NMF or NNMF):

        # https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py


##### 6.2.3.7. Limitações da representação do Saco de Palavras

    # Uma coleção de unigramas (o que é um pacote de palavras) não pode capturar frases e expressões com várias palavras, desconsiderando efetivamente qualquer dependência da ordem das palavras. Além disso, o modelo de saco de palavras não leva em conta possíveis erros ortográficos ou derivações de palavras.

    # N-grams para o resgate! Ao invés de construir uma simples coleção de unigramas (n=1), pode-se preferir uma coleção de bigramas (n=2), onde são contadas as ocorrências de pares de palavras consecutivas.

    # Pode-se, alternativamente, considerar uma coleção de n-grams de caracteres, uma representação resistente a erros de ortografia e derivações.

    # Por exemplo, digamos que estamos lidando com um corpus de dois documentos: ['words', 'wprds']. O segundo documento contém um erro de ortografia da palavra 'words'. Uma simples representação de um saco de palavras consideraria esses dois documentos muito distintos, diferindo em ambas as duas características possíveis. Uma representação de 2 gramas de caracteres, no entanto, encontraria os documentos correspondentes em 4 de 8 recursos, o que pode ajudar o classificador preferido a decidir melhor: 

ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
counts = ngram_vectorizer.fit_transform(['words', 'wprds'])
ngram_vectorizer.get_feature_names_out()

counts.toarray().astype(int)

    # No exemplo acima, é usado o analisador char_wb, que cria n-grams apenas a partir de caracteres dentro dos limites da palavra (preenchidos com espaço em cada lado). O analisador de caracteres, alternativamente, cria n-gramas que abrangem palavras: 

ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5, 5))
ngram_vectorizer.fit_transform(['jumpy fox'])

ngram_vectorizer.get_feature_names_out()


ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5))
ngram_vectorizer.fit_transform(['jumpy fox'])

ngram_vectorizer.get_feature_names_out()

    # A variante com reconhecimento de limites de palavras char_wb é especialmente interessante para idiomas que usam espaços em branco para separação de palavras, pois gera recursos significativamente menos barulhentos do que a variante char bruta nesse caso. Para essas linguagens, ele pode aumentar a precisão preditiva e a velocidade de convergência de classificadores treinados usando esses recursos, mantendo a robustez em relação a erros de ortografia e derivações de palavras.

    # Embora algumas informações de posicionamento local possam ser preservadas pela extração de n-gramas em vez de palavras individuais, o pacote de palavras e o pacote de n-grams destroem a maior parte da estrutura interna do documento e, portanto, a maior parte do significado carregado por essa estrutura interna.

    # A fim de abordar a tarefa mais ampla de compreensão da linguagem natural, a estrutura local de frases e parágrafos deve ser levada em consideração. Muitos desses modelos serão, portanto, lançados como problemas de “saída estruturada” que estão atualmente fora do escopo do scikit-learn. 




##### 6.2.3.8. Vetorizando um grande corpus de texto com o truque de hash

    # O esquema de vetorização acima é simples, mas o fato de ele manter um mapeamento na memória dos tokens de string para os índices de recursos inteiros (o atributo vocabulário_) causa vários problemas ao lidar com grandes conjuntos de dados: 

        # quanto maior o corpus, maior será o vocabulário e, portanto, o uso da memória também,

        # ajuste requer a alocação de estruturas de dados intermediárias de tamanho proporcional ao do conjunto de dados original.

        # construir o mapeamento de palavras requer uma passagem completa sobre o conjunto de dados, portanto, não é possível ajustar classificadores de texto de maneira estritamente online.

        # vetorizadores de decapagem e remoção de decapagem com um grande vocabulário_ podem ser muito lentos (normalmente muito mais lentos do que decapagem / remoção de decapagem de estruturas de dados planas, como um array NumPy do mesmo tamanho),

        # não é facilmente possível dividir o trabalho de vetorização em subtarefas simultâneas, pois o atributo vocabulário_ teria que ser um estado compartilhado com uma barreira de sincronização refinada: o mapeamento da string de token para o índice de recursos depende da ordenação da primeira ocorrência de cada token, portanto, teria que ser compartilhado, potencialmente prejudicando o desempenho dos trabalhadores simultâneos a ponto de torná-los mais lentos que a variante sequencial. 

    # É possível superar essas limitações combinando o “truque de hash” (Feature hashing) implementado pela classe FeatureHasher e os recursos de pré-processamento de texto e tokenização do CountVectorizer.

    # Essa combinação está sendo implementada em HashingVectorizer, uma classe de transformador que é principalmente compatível com API com CountVectorizer. O HashingVectorizer é sem estado, o que significa que você não precisa chamá-lo de ajuste: 

from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer(n_features=10)
hv.transform(corpus)

    # Você pode ver que 16 tokens de recursos diferentes de zero foram extraídos na saída do vetor: isso é menos do que os 19 tokens diferentes de zero extraídos anteriormente pelo CountVectorizer no mesmo corpus de brinquedo. A discrepância vem de colisões de funções de hash devido ao baixo valor do parâmetro n_features.

    # Em uma configuração do mundo real, o parâmetro n_features pode ser deixado em seu valor padrão de 2 ** 20 (aproximadamente um milhão de recursos possíveis). Se a memória ou o tamanho dos modelos downstream for um problema, selecionar um valor mais baixo, como 2 ** 18, pode ajudar sem introduzir muitas colisões adicionais em tarefas típicas de classificação de texto.

    # Observe que a dimensionalidade não afeta o tempo de treinamento da CPU de algoritmos que operam em matrizes CSR (LinearSVC(dual=True), Perceptron, SGDClassifier, PassiveAggressive), mas afeta algoritmos que trabalham com matrizes CSC (LinearSVC(dual=False), Lasso(), etc).

    # Vamos tentar novamente com a configuração padrão: 

hv = HashingVectorizer()
hv.transform(corpus)

    # Não temos mais as colisões, mas isso ocorre às custas de uma dimensionalidade muito maior do espaço de saída. É claro que outros termos além dos 19 usados aqui ainda podem colidir uns com os outros.

    # O HashingVectorizer também vem com as seguintes limitações: 


        # não é possível inverter o modelo (sem método inverse_transform), nem acessar a representação de string original dos recursos, devido à natureza unidirecional da função hash que realiza o mapeamento.

        # ele não fornece ponderação IDF, pois isso introduziria statefulness no modelo. Um TfidfTransformer pode ser anexado a ele em um pipeline, se necessário. 




##### 6.2.3.9. Executando dimensionamento fora do núcleo com HashingVectorizer

    # Um desenvolvimento interessante do uso de um HashingVectorizer é a capacidade de realizar dimensionamento fora do núcleo. Isso significa que podemos aprender com dados que não cabem na memória principal do computador.

    # Uma estratégia para implementar o dimensionamento fora do núcleo é transmitir dados para o estimador em minilotes. Cada mini-lote é vetorizado usando HashingVectorizer para garantir que o espaço de entrada do estimador tenha sempre a mesma dimensionalidade. A quantidade de memória usada a qualquer momento é, portanto, limitada pelo tamanho de um minilote. Embora não haja limite para a quantidade de dados que podem ser ingeridos usando essa abordagem, do ponto de vista prático, o tempo de aprendizado geralmente é limitado pelo tempo de CPU que se deseja gastar na tarefa.

    # Para obter um exemplo completo de dimensionamento fora do núcleo em uma tarefa de classificação de texto, consulte Classificação fora do núcleo de documentos de texto. 






##### 6.2.3.10. Personalizando as classes do vetorizador 

    # É possível customizar o comportamento passando um callable para o construtor do vetorizador: 

def my_tokenizer(s):
    return s.split()

vectorizer = CountVectorizer(tokenizer=my_tokenizer)
vectorizer.build_analyzer()(u"Some... punctuation!") == (
    ['some...', 'punctuation!'])

    # Em particular, nomeamos: 

        # pré-processador: um callable que recebe um documento inteiro como entrada (como uma única string) e retorna uma versão possivelmente transformada do documento, ainda como uma string inteira. Isso pode ser usado para remover tags HTML, colocar todo o documento em minúsculas, etc.

        # tokenizer: um callable que pega a saída do pré-processador e a divide em tokens, então retorna uma lista deles.

        # analisador: um callable que substitui o pré-processador e o tokenizer. Todos os analisadores padrão chamam o pré-processador e o tokenizador, mas os analisadores personalizados ignorarão isso. A extração de N-gram e a filtragem de palavras de parada ocorrem no nível do analisador, portanto, um analisador personalizado pode ter que reproduzir essas etapas. 


    # (Os usuários do Lucene podem reconhecer esses nomes, mas esteja ciente de que os conceitos do scikit-learn podem não mapear um a um para os conceitos do Lucene.)

    # Para tornar o pré-processador, tokenizer e analisadores cientes dos parâmetros do modelo, é possível derivar da classe e substituir os métodos de fábrica build_preprocessor, build_tokenizer e build_analyzer em vez de passar funções personalizadas.

    # Algumas dicas e truques: 

        # Se os documentos forem pré-tokenizados por um pacote externo, armazene-os em arquivos (ou strings) com os tokens separados por espaço em branco e passe analyzer=str.split

        # Análises sofisticadas em nível de token, como lematização, divisão composta, filtragem baseada em parte da fala, etc., não estão incluídas na base de código do scikit-learn, mas podem ser adicionadas personalizando o tokenizer ou o analisador. Aqui está um CountVectorizer com um tokenizer e lematizer usando NLTK: 

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

vect = CountVectorizer(tokenizer=LemmaTokenizer())  

    # (Observe que isso não filtrará a pontuação.)

    # O exemplo a seguir irá, por exemplo, transformar alguma ortografia britânica em ortografia americana: 


import re
def to_british(tokens):
    for t in tokens:
        t = re.sub(r"(...)our$", r"\1or", t)
        t = re.sub(r"([bt])re$", r"\1er", t)
        t = re.sub(r"([iy])s(e$|ing|ation)", r"\1z\2", t)
        t = re.sub(r"ogue$", "og", t)
        yield t

class CustomVectorizer(CountVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(to_british(tokenize(doc)))

print(CustomVectorizer().build_analyzer()(u"color colour"))


    # para outros estilos de pré-processamento; exemplos incluem derivação, lematização ou normalização de tokens numéricos, com o último ilustrado em: 

        # https://scikit-learn.org/stable/auto_examples/bicluster/plot_bicluster_newsgroups.html#sphx-glr-auto-examples-bicluster-plot-bicluster-newsgroups-py

    # A personalização do vetorizador também pode ser útil ao lidar com idiomas asiáticos que não usam um separador de palavras explícito, como espaços em branco. 