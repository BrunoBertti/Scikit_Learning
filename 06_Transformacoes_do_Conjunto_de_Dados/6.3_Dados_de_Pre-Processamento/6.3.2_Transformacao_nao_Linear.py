from sklearn import preprocessing


########## 6.3.2. Transformação não linear ##########

    # Dois tipos de transformações estão disponíveis: transformadas quantílicas e transformadas de potência. Ambas as transformações de quantil e potência são baseadas em transformações monotônicas dos recursos e, assim, preservam a classificação dos valores ao longo de cada recurso.

    # As transformações de quantil colocam todos os recursos na mesma distribuição desejada com base na fórmula G^{-1}(F(X)) onde F é a função de distribuição cumulativa do recurso e G^{-1} a função de quantil da saída desejada distribuição G. Esta fórmula está usando os dois seguintes fatos: (i) se X é uma variável aleatória com uma função de distribuição cumulativa contínua F então F(X) é uniformemente distribuído em [0,1]; (ii) se U é uma variável aleatória com distribuição uniforme em [0,1], então G^{-1}(U) tem distribuição G. Ao realizar uma transformação de classificação, uma transformação quantílica suaviza distribuições incomuns e é menos influenciada por outliers do que métodos de escalonamento. No entanto, distorce as correlações e distâncias dentro e entre as feições.

    # Transformações de potência são uma família de transformações paramétricas que visam mapear dados de qualquer distribuição para o mais próximo de uma distribuição gaussiana. 








##### 6.3.2.1. Mapeamento para uma distribuição uniforme

    # O QuantileTransformer fornece uma transformação não paramétrica para mapear os dados para uma distribuição uniforme com valores entre 0 e 1: 

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]) 


    # Esta característica corresponde ao comprimento da sépala em cm. Uma vez aplicada a transformação quantílica, esses marcos aproximam-se dos percentis previamente definidos: 

np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100])

    # Isso pode ser confirmado em um conjunto de testes independente com observações semelhantes: 

np.percentile(X_test[:, 0], [0, 25, 50, 75, 100])

np.percentile(X_test_trans[:, 0], [0, 25, 50, 75, 100])







##### 6.3.2.2. Mapeamento para uma distribuição gaussiana

    # Em muitos cenários de modelagem, a normalidade dos recursos em um conjunto de dados é desejável. Transformações de potência são uma família de transformações paramétricas e monotônicas que visam mapear dados de qualquer distribuição para o mais próximo possível de uma distribuição gaussiana, a fim de estabilizar a variância e minimizar a assimetria.

    # PowerTransformer atualmente fornece duas dessas transformações de energia, a transformação de Yeo-Johnson e a transformação de Box-Cox.

    # A transformada de Yeo-Johnson é dada por: 


        # \begin{split}x_i^{(\lambda)} =
        # \begin{cases}
        #  [(x_i + 1)^\lambda - 1] / \lambda & \text{if } \lambda \neq 0, x_i \geq 0, \\[8pt]
        # \ln{(x_i + 1)} & \text{if } \lambda = 0, x_i \geq 0 \\[8pt]
        # -[(-x_i + 1)^{2 - \lambda} - 1] / (2 - \lambda) & \text{if } \lambda \neq 2, x_i < 0, \\[8pt]
        #  - \ln (- x_i + 1) & \text{if } \lambda = 2, x_i < 0
        # \end{cases}\end{split}


    # enquanto a transformada Box-Cox é dada por: 


        # \begin{split}x_i^{(\lambda)} =
        # \begin{cases}
        # \dfrac{x_i^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, \\[8pt]
        # \ln{(x_i)} & \text{if } \lambda = 0,
        # \end{cases}\end{split}


    # Box-Cox só pode ser aplicado a dados estritamente positivos. Em ambos os métodos, a transformação é parametrizada por , que é determinada através da estimação de máxima verossimilhança. Aqui está um exemplo de uso de Box-Cox para mapear amostras extraídas de uma distribuição lognormal para uma distribuição normal: 

pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))
X_lognormal

pt.fit_transform(X_lognormal)

    # Enquanto o exemplo acima define a opção de padronização como False, o PowerTransformer aplicará a normalização de média zero e variância de unidade à saída transformada por padrão.

    # Abaixo estão exemplos de Box-Cox e Yeo-Johnson aplicados a várias distribuições de probabilidade. Observe que, quando aplicadas a certas distribuições, as transformadas de potência atingem resultados muito gaussianos, mas com outras, são ineficazes. Isso destaca a importância de visualizar os dados antes e depois da transformação. 


        # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html


    # Também é possível mapear dados para uma distribuição normal usando QuantileTransformer definindo output_distribution='normal'. Usando o exemplo anterior com o conjunto de dados da íris: 

quantile_transformer = preprocessing.QuantileTransformer(
    output_distribution='normal', random_state=0)
X_trans = quantile_transformer.fit_transform(X)
quantile_transformer.quantiles_

    # Assim, a mediana da entrada torna-se a média da saída, centrada em 0. A saída normal é cortada para que o mínimo e o máximo da entrada - correspondentes aos quantis 1e-7 e 1 - 1e-7 respectivamente - não se tornem infinitos sob a transformação. 