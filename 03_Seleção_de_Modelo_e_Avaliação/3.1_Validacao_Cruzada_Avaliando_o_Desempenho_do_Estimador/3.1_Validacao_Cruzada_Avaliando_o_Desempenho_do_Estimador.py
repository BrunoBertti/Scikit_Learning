########## 3.1. Validação cruzada: avaliando o desempenho do estimador ##########


    # Aprender os parâmetros de uma função de predição e testá-la com os mesmos dados é um erro metodológico: um modelo que apenas repetisse os rótulos das amostras que acabou de ver teria uma pontuação perfeita, mas não conseguiria prever nada útil no ainda- dados invisíveis. Essa situação é chamada de overfitting. Para evitá-lo, é prática comum, ao realizar um experimento de aprendizado de máquina (supervisionado), armazenar parte dos dados disponíveis como um conjunto de testes X_test, y_test. Observe que a palavra “experimento” não tem a intenção de denotar apenas uso acadêmico, porque mesmo em ambientes comerciais, o aprendizado de máquina geralmente começa experimentalmente. Aqui está um fluxograma do fluxo de trabalho de validação cruzada típico no treinamento do modelo. Os melhores parâmetros podem ser determinados por técnicas de pesquisa em grade. 

        # https://scikit-learn.org/stable/_images/grid_search_workflow.png

    # No scikit-learn, uma divisão aleatória em conjuntos de treinamento e teste pode ser calculada rapidamente com a função auxiliar train_test_split. Vamos carregar o conjunto de dados da íris para ajustar uma máquina de vetor de suporte linear: 


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris(return_X_y=True)
X.shape, y.shape

    # Agora podemos amostrar rapidamente um conjunto de treinamento, mantendo 40% dos dados para testar (avaliar) nosso classificador: 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

X_train.shape, y_train.shape

X_test.shape, y_test.shape


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)
    
    # Ao avaliar diferentes configurações ("hiperparâmetros") para estimadores, como a configuração C que deve ser definida manualmente para um SVM, ainda há o risco de overfitting no conjunto de teste porque os parâmetros podem ser ajustados até que o estimador tenha um desempenho ideal. Dessa forma, o conhecimento sobre o conjunto de teste pode “vazar” para o modelo e as métricas de avaliação não relatam mais o desempenho de generalização. Para resolver este problema, outra parte do conjunto de dados pode ser realizada como um chamado "conjunto de validação": o treinamento continua no conjunto de treinamento, após o que a avaliação é feita no conjunto de validação, e quando o experimento parece ser bem sucedido , a avaliação final pode ser feita no conjunto de teste.

    # No entanto, ao particionar os dados disponíveis em três conjuntos, reduzimos drasticamente o número de amostras que podem ser usadas para aprender o modelo, e os resultados podem depender de uma escolha aleatória particular para o par de (treinar, validação) conjuntos.

    # Uma solução para este problema é um procedimento denominado validação cruzada (abreviatura de CV). Um conjunto de teste ainda deve ser apresentado para avaliação final, mas o conjunto de validação não é mais necessário ao fazer o CV. Na abordagem básica, chamada k-fold CV, o conjunto de treinamento é dividido em k conjuntos menores (outras abordagens são descritas abaixo, mas geralmente seguem os mesmos princípios). O seguinte procedimento é seguido para cada uma das k "dobras": 

        # Um modelo é treinado usando k-1 das dobras como dados de treinamento;

        # o modelo resultante é validado na parte restante dos dados (ou seja, é usado como um conjunto de teste para calcular uma medida de desempenho, como precisão). 

    # A medida de desempenho relatada pela validação cruzada k-fold é, então, a média dos valores calculados no loop. Essa abordagem pode ser computacionalmente cara, mas não desperdiça muitos dados (como é o caso ao corrigir um conjunto de validação arbitrário), o que é uma grande vantagem em problemas como inferência inversa, onde o número de amostras é muito pequeno. 

        # https://scikit-learn.org/stable/_images/grid_search_cross_validation.png