from sklearn import preprocessing


########## 6.3.3. Normalização ##########

    # A normalização é o processo de dimensionar amostras individuais para ter uma norma unitária. Esse processo pode ser útil se você planeja usar uma forma quadrática, como o produto escalar ou qualquer outro kernel, para quantificar a semelhança de qualquer par de amostras.

    # Essa suposição é a base do Modelo de Espaço Vetorial frequentemente usado em classificação de texto e contextos de agrupamento.

    # A função normalize fornece uma maneira rápida e fácil de executar essa operação em um único conjunto de dados do tipo array, usando as normas l1, l2 ou max: 


X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')

X_normalized


    # O módulo de pré-processamento fornece ainda um Normalizer de classe de utilitário que implementa a mesma operação usando a API Transformer (mesmo que o método de ajuste seja inútil neste caso: a classe é sem estado, pois esta operação trata amostras de forma independente).

    # Esta classe é, portanto, adequada para uso nas etapas iniciais de um pipeline: 

normalizer = preprocessing.Normalizer().fit(X)  # fit does nothing
normalizer

    # A instância do normalizador pode então ser usada em vetores de amostra como qualquer transformador: 

normalizer.transform(X)

normalizer.transform([[-1.,  1., 0.]])


    # Nota: A normalização L2 também é conhecida como pré-processamento de sinal espacial. 


    ## Entrada esparsa

    ## normalize e Normalizer aceitam matrizes densas do tipo array e esparsas de scipy.sparse como entrada.

    ## Para entrada esparsa, os dados são convertidos para a representação Compressed Sparse Rows (consulte scipy.sparse.csr_matrix) antes de serem alimentados em rotinas Cython eficientes. Para evitar cópias de memória desnecessárias, é recomendável escolher a representação CSR upstream. 