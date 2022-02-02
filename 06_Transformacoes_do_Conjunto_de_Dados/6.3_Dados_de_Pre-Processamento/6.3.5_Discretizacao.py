import numpy as np
from sklearn import preprocessing

########## 6.3.5. Discretização ##########

    # A discretização (também conhecida como quantização ou binning) fornece uma maneira de particionar recursos contínuos em valores discretos. Certos conjuntos de dados com características contínuas podem se beneficiar da discretização, pois a discretização pode transformar o conjunto de dados de atributos contínuos em um com apenas atributos nominais.

    # Recursos discretizados codificados one-hot podem tornar um modelo mais expressivo, mantendo a interpretabilidade. Por exemplo, o pré-processamento com um discretizador pode introduzir não linearidade em modelos lineares. Para possibilidades mais avançadas, em particular suaves, consulte Gerando recursos polinomiais mais abaixo. 


##### 6.3.5.1. Discretização de K-bins

    # KBinsDiscretizer discretiza recursos em k bins: 

X = np.array([[ -3., 5., 15 ],
              [  0., 6., 14 ],
              [  6., 3., 11 ]])
est = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit(X)


    # Por padrão, a saída é codificada em uma matriz esparsa (consulte Codificação de recursos categóricos) e isso pode ser configurado com o parâmetro de codificação. Para cada recurso, as arestas dos compartimentos são calculadas durante o ajuste e, juntamente com o número de compartimentos, definirão os intervalos. Portanto, para o exemplo atual, esses intervalos são definidos como: 

        # feature 1:  {[-\infty, -1), [-1, 2), [2, \infty)}

        # feature 2: {[-\infty, 5), [5, \infty)}

        # feature 3: {[-\infty, 14), [14, \infty)}

    # Com base nesses intervalos bin, X é transformado da seguinte forma: 

est.transform(X)    


    # O conjunto de dados resultante contém atributos ordinais que podem ser usados em um pipeline.

    # A discretização é semelhante à construção de histogramas para dados contínuos. No entanto, os histogramas se concentram na contagem de recursos que se enquadram em determinados compartimentos, enquanto a discretização se concentra na atribuição de valores de recursos a esses compartimentos.

    # KBinsDiscretizer implementa diferentes estratégias de binning, que podem ser selecionadas com o parâmetro strategy. A estratégia “uniforme” usa caixas de largura constante. A estratégia 'quantil' usa os valores dos quantis para ter compartimentos igualmente preenchidos em cada recurso. A estratégia 'kmeans' define bins com base em um procedimento de agrupamento k-means realizado em cada recurso de forma independente.

    # Esteja ciente de que é possível especificar bins personalizados passando um callable definindo a estratégia de discretização para FunctionTransformer. Por exemplo, podemos usar a função Pandas pandas.cut: 


import pandas as pd
import numpy as np
bins = [0, 1, 13, 20, 60, np.inf]
labels = ['infant', 'kid', 'teen', 'adult', 'senior citizen']
transformer = preprocessing.FunctionTransformer(
    pd.cut, kw_args={'bins': bins, 'labels': labels, 'retbins': False}
)
X = np.array([0.2, 2, 15, 25, 97])
transformer.fit_transform(X)


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization.html#sphx-glr-auto-examples-preprocessing-plot-discretization-py

    ## https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_classification.html#sphx-glr-auto-examples-preprocessing-plot-discretization-classification-py

    ## https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_strategies.html#sphx-glr-auto-examples-preprocessing-plot-discretization-strategies-py








##### 6.3.5.2. Binarização de recursos 

    # A binarização de recursos é o processo de delimitar recursos numéricos para obter valores booleanos. Isso pode ser útil para estimadores probabilísticos downstream que supõem que os dados de entrada são distribuídos de acordo com uma distribuição de Bernoulli multivariada. Por exemplo, este é o caso do BernoulliRBM.

    # Também é comum entre a comunidade de processamento de texto usar valores de recursos binários (provavelmente para simplificar o raciocínio probabilístico), mesmo que contagens normalizadas (também conhecidas como frequências de termos) ou recursos de valor TF-IDF geralmente tenham um desempenho um pouco melhor na prática.

    # Quanto ao Normalizer, a classe utilitária Binarizer deve ser usada nos estágios iniciais do Pipeline. O método de ajuste não faz nada, pois cada amostra é tratada independentemente das outras: 


X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]

binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
binarizer


binarizer.transform(X)

    # É possível ajustar o limiar do binarizador: 

binarizer = preprocessing.Binarizer(threshold=1.1)
binarizer.transform(X)

    # Quanto à classe Normalizer, o módulo de pré-processamento fornece uma função complementar binarize para ser usada quando a API do transformador não for necessária.

    # Observe que o Binarizer é semelhante ao KBinsDiscretizer quando k = 2 e quando a borda do compartimento está no limite de valor. 


    ## Entrada esparsa

    ## binarize e Binarizer aceitam matrizes densas do tipo array e esparsas de scipy.sparse como entrada.

    ## Para entrada esparsa, os dados são convertidos para a representação Compressed Sparse Rows (consulte scipy.sparse.csr_matrix). Para evitar cópias de memória desnecessárias, é recomendável escolher a representação CSR upstream. 