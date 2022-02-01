########## 6.3.1. Padronização, ou remoção média e escala de variação ##########


    # A padronização de conjuntos de dados é um requisito comum para muitos estimadores de aprendizado de máquina implementados no scikit-learn; eles podem se comportar mal se os recursos individuais não se parecerem mais ou menos com dados padrão normalmente distribuídos: Gaussiano com média zero e variância unitária.

    # Na prática, muitas vezes ignoramos a forma da distribuição e apenas transformamos os dados para centralizá-los, removendo o valor médio de cada recurso e, em seguida, dimensionamos dividindo recursos não constantes por seu desvio padrão.

    # Por exemplo, muitos elementos usados ​​na função objetivo de um algoritmo de aprendizado (como o kernel RBF de Support Vector Machines ou os regularizadores l1 e l2 de modelos lineares) assumem que todos os recursos estão centrados em torno de zero e têm variância na mesma ordem. Se um recurso tem uma variância que é ordens de magnitude maior do que outros, ele pode dominar a função objetivo e tornar o estimador incapaz de aprender com outros recursos corretamente como esperado.

    # O módulo de pré-processamento fornece a classe de utilitário StandardScaler, que é uma maneira rápida e fácil de realizar a seguinte operação em um conjunto de dados tipo array: 



from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
scaler = preprocessing.StandardScaler().fit(X_train)
scaler

scaler.mean_

scaler.scale_

X_scaled = scaler.transform(X_train)
X_scaled

    # Os dados em escala têm média zero e variância de unidade: 

X_scaled.mean(axis=0)


X_scaled.std(axis=0)


    # Essa classe implementa a API do Transformer para calcular a média e o desvio padrão em um conjunto de treinamento para poder reaplicar posteriormente a mesma transformação no conjunto de teste. Esta classe é, portanto, adequada para uso nas etapas iniciais de um pipeline: 

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)  # aplicar dimensionamento em dados de treinamento


pipe.score(X_test, y_test)  # aplique dimensionamento em dados de teste, sem vazar dados de treinamento. 


    # É possível desabilitar a centralização ou o dimensionamento passando with_mean=False ou with_std=False para o construtor de StandardScaler. 








##### 6.3.1.1. Dimensionando recursos para um intervalo

    # Uma padronização alternativa é dimensionar os recursos para ficarem entre um determinado valor mínimo e máximo, geralmente entre zero e um, ou de modo que o valor absoluto máximo de cada recurso seja dimensionado para o tamanho da unidade. Isso pode ser feito usando MinMaxScaler ou MaxAbsScaler, respectivamente.

    # A motivação para usar esse dimensionamento inclui robustez para desvios padrão muito pequenos de recursos e preservação de entradas zero em dados esparsos.

    # Aqui está um exemplo para dimensionar uma matriz de dados de brinquedo para o intervalo [0, 1]: 

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_train_minmax


    # A mesma instância do transformador pode então ser aplicada a alguns novos dados de teste não vistos durante a chamada de ajuste: as mesmas operações de dimensionamento e deslocamento serão aplicadas para serem consistentes com a transformação realizada nos dados do trem: 

X_test = np.array([[-3., -1.,  4.]])
X_test_minmax = min_max_scaler.transform(X_test)
X_test_minmax

    # É possível fazer uma introspecção dos atributos do scaler para descobrir a natureza exata da transformação aprendida nos dados de treinamento: 

min_max_scaler.scale_


min_max_scaler.min_


    # Se MinMaxScaler receber um feature_range=(min, max) explícito, a fórmula completa será: 

X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

X_scaled = X_std * (max - min) + min

    # O MaxAbsScaler funciona de maneira muito semelhante, mas é dimensionado de forma que os dados de treinamento fiquem dentro do intervalo [-1, 1] dividindo-se pelo maior valor máximo em cada recurso. Destina-se a dados que já estão centrados em zero ou dados esparsos.

    # Aqui está como usar os dados do brinquedo do exemplo anterior com este scaler: 

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_train_maxabs

X_test = np.array([[ -3., -1.,  4.]])
X_test_maxabs = max_abs_scaler.transform(X_test)
X_test_maxabs

max_abs_scaler.scale_



##### 6.3.1.2. Como escalonar dados esparsos

    # A centralização de dados esparsos destruiria a estrutura esparsa nos dados e, portanto, raramente é uma coisa sensata a fazer. No entanto, pode fazer sentido dimensionar entradas esparsas, especialmente se os recursos estiverem em escalas diferentes.

    # O MaxAbsScaler foi projetado especificamente para dimensionar dados esparsos e é a maneira recomendada de fazer isso. No entanto, StandardScaler pode aceitar matrizes scipy.sparse como entrada, desde que with_mean=False seja explicitamente passado para o construtor. Caso contrário, um ValueError será gerado, pois a centralização silenciosa quebraria a esparsidade e muitas vezes travaria a execução alocando quantidades excessivas de memória involuntariamente. O RobustScaler não pode ser ajustado a entradas esparsas, mas você pode usar o método transform em entradas esparsas.

    # Observe que os scalers aceitam o formato de linhas esparsas compactadas e colunas esparsas compactadas (consulte scipy.sparse.csr_matrix e scipy.sparse.csc_matrix). Qualquer outra entrada esparsa será convertida na representação Compressed Sparse Rows. Para evitar cópias de memória desnecessárias, é recomendável escolher a representação CSR ou CSC upstream.

    # Finalmente, se espera-se que os dados centralizados sejam pequenos o suficiente, converter explicitamente a entrada em um array usando o método toarray de matrizes esparsas é outra opção. 



##### 6.3.1.3. Dimensionamento de dados com outliers


    # Se seus dados contiverem muitos valores discrepantes, o dimensionamento usando a média e a variância dos dados provavelmente não funcionará muito bem. Nesses casos, você pode usar o RobustScaler como um substituto imediato. Ele usa estimativas mais robustas para o centro e o alcance de seus dados. 



    ## referências:

    ## Further discussion on the importance of centering and scaling data is available on this FAQ: Should I normalize/standardize/rescale the data? (http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html)


    ## Escalonamento vs Clareamento

    ## Às vezes, não é suficiente centralizar e dimensionar os recursos de forma independente, pois um modelo downstream pode ainda fazer algumas suposições sobre a independência linear dos recursos.

    ## Para resolver esse problema, você pode usar o PCA com whiten=True para remover ainda mais a correlação linear entre os recursos. 





##### 6.3.1.4. Centralizando matrizes de kernel 

    # Se você tiver uma matriz de kernel de um kernel K que calcula um produto escalar em um espaço de recursos (possivelmente implicitamente) definido por uma função \phi(\cdot), um KernelCenterer pode transformar a matriz de kernel para que ela contenha produtos internos no recurso espaço definido por \phi seguido pela remoção da média nesse espaço. Em outras palavras, o KernelCenterer calcula a matriz Gram centrada associada a um kernel semidefinido positivo K. 


    # Formulação matemática 


    # Podemos dar uma olhada na formulação matemática agora que temos a intuição. Seja K uma matriz kernel de forma (n_samples, n_samples) calculada a partir de X, uma matriz de dados de forma (n_samples, n_features), durante a etapa de ajuste. L é definido por

        # K(X, X) = \phi(X) . \phi(X)^{T}

    # \phi(X) é um mapeamento de função de X para um espaço de Hilbert. Um kernel centralizado \tilde{K} é definido como:

        # \til{K}(X, X) = \til{\phi}(X) . \til{\phi}(X)^{T}

    # onde \tilde{\phi}(X) resulta da centralização de \phi(X) no espaço de Hilbert.


    # Assim, pode-se calcular \tilde{K} mapeando X usando a função \phi(\cdot) e centralizando os dados neste novo espaço. No entanto, kernels são frequentemente usados ​​porque permitem alguns cálculos de álgebra que evitam computar explicitamente esse mapeamento usando \phi(\cdot). De fato, pode-se implicitamente centrar como mostrado no Apêndice B em [Scholkopf1998]:

        # \tilde{K} = K - 1_{\text{n}_{amostras}} K - K 1_{\text{n}_{amostras}} + 1_{\text{n}_{amostras}} K 1_ {\text{n}_{amostras}}



    #  1_{\text{n}_{samples}} é uma matriz de (n_samples, n_samples) onde todas as entradas são iguais a \frac{1}{\text{n}_{samples}}. Na etapa de transformação, o kernel se torna K_{test}(X, Y) definido como:


        # K_{teste}(X, Y) = \phi(Y) . \phi(X)^{T}


    # Y é o conjunto de dados de teste da forma (n_samples_test, n_features) e, portanto, K_{test} é da forma (n_samples_test, n_samples). Neste caso, a centralização de K_{test} é feita como:


        # \tilde{K}_{teste}(X, Y) = K_{teste} - 1'_{\text{n}_{amostras}} K - K_{teste} 1_{\text{n}_{amostras }} + 1'_{\text{n}_{amostras}} K 1_{\text{n}_{amostras}}



    # 1'_{\text{n}_{samples}} é uma matriz de forma (n_samples_test, n_samples) onde todas as entradas são iguais a \frac{1}{\text{n}_{samples}}