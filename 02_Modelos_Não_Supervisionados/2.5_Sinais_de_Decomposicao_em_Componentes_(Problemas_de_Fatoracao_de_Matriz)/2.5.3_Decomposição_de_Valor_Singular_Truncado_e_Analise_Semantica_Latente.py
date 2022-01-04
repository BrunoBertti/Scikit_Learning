########## 2.5.3. Decomposição de valor singular truncado e análise semântica latente ##########

    # TruncatedSVD implementa uma variante da decomposição de valor singular (SVD) que apenas calcula os k maiores valores singulares, onde k é um parâmetro especificado pelo usuário.

    # Quando SVD truncado é aplicado a matrizes de documento de termo (conforme retornado por CountVectorizer ou TfidfVectorizer), essa transformação é conhecida como análise semântica latente (LSA), porque transforma tais matrizes em um espaço “semântico” de baixa dimensionalidade. Em particular, LSA é conhecido por combater os efeitos de sinonímia e polissemia (ambos os quais significam aproximadamente que há vários significados por palavra), o que faz com que as matrizes de documento de termo sejam excessivamente esparsas e exibam similaridade pobre em medidas como similaridade de cosseno.

    # Nota: LSA também é conhecido como indexação semântica latente, LSI, embora se refira estritamente ao seu uso em índices persistentes para fins de recuperação de informações. 


    # Matematicamente, o SVD truncado aplicado às amostras de treinamento X produz uma aproximação de classificação baixa X:


        # X \ approx X_k = U_k \ Sigma_k V_k ^ \ top


    # Após esta operação, U_k \ Sigma_k é o conjunto de treinamento transformado com k recursos (chamados de n_components na API).

    # Para transformar também um conjunto de teste X, nós o multiplicamos por V_k:

        # X '= X V_k


    # Nota: A maioria dos tratamentos de LSA na literatura de processamento de linguagem natural (PNL) e recuperação de informação (IR) troca os eixos da matriz X para que ela tenha a forma n_features × n_samples. Apresentamos o LSA de uma maneira diferente que corresponde melhor à API do scikit-learn, mas os valores singulares encontrados são os mesmos.


    # O TruncatedSVD é muito semelhante ao PCA, mas difere porque a matriz X não precisa ser centralizada. Quando as médias de X em colunas (por recurso) são subtraídas dos valores de recursos, o SVD truncado na matriz resultante é equivalente ao PCA. Em termos práticos, isso significa que o transformador TruncatedSVD aceita matrizes scipy.sparse sem a necessidade de densificá-las, pois a densificação pode preencher a memória mesmo para coleções de documentos de tamanho médio.


    # Embora o transformador TruncatedSVD funcione com qualquer matriz de recurso, usá-lo em matrizes tf – idf é recomendado em vez de contagens de frequência bruta em uma configuração de processamento de documentos / LSA. Em particular, a escala sublinear e a frequência inversa do documento devem ser ativadas (sublinear_tf = True, use_idf = True) para trazer os valores dos recursos mais próximos de uma distribuição gaussiana, compensando as suposições errôneas do LSA sobre os dados textuais. 


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py



    ## Referências:

    ## Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze (2008), Introduction to Information Retrieval, Cambridge University Press, chapter 18: Matrix decompositions & latent semantic indexing (https://nlp.stanford.edu/IR-book/pdf/18lsi.pdf)