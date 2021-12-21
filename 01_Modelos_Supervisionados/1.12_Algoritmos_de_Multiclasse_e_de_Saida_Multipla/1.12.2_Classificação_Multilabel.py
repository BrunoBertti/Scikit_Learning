########## 1.12.2. Classificação Multilabel ##########

    # A classificação de múltiplas etiquetas (intimamente relacionada à classificação de múltiplas saídas) é uma tarefa de classificação que rotula cada amostra com m etiquetas de n_classes de classes possíveis, onde m pode ser de 0 a n_classes inclusive. Isso pode ser considerado como propriedades de previsão de uma amostra que não são mutuamente exclusivas. Formalmente, uma saída binária é atribuída a cada classe, para cada amostra. Classes positivas são indicadas com 1 e classes negativas com 0 ou -1. Portanto, é comparável à execução de tarefas de classificação binária n_classes, por exemplo, com MultiOutputClassifier. Essa abordagem trata cada rótulo de forma independente, enquanto os classificadores de vários rótulos podem tratar as várias classes simultaneamente, levando em consideração o comportamento correlacionado entre elas.

    # Por exemplo, previsão dos tópicos relevantes para um documento de texto ou vídeo. O documento ou vídeo pode ser sobre 'religião', 'política', 'finanças' ou 'educação', várias das aulas temáticas ou todas as aulas temáticas. 



##### 1.12.2.1. Formato de destino

    # Uma representação válida de multilabel y é uma matriz binária densa ou esparsa de forma (n_samples, n_classes). Cada coluna representa uma classe. Os 1 em cada linha denotam as classes positivas com as quais uma amostra foi rotulada. Um exemplo de uma matriz densa y para 3 amostras: 

import numpy as np
y = np.array([[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
print(y)

    # Matrizes binárias densas também podem ser criadas usando MultiLabelBinarizer. Para obter mais informações, consulte Transformando o alvo de previsão (y).

    # Um exemplo do mesmo y na forma de matriz esparsa: 

y_sparse = sparse.csr_matrix(y)
print(y_sparse)

##### 1.12.2.2. MultiOutputClassifier

    # O suporte à classificação de múltiplas etiquetas pode ser adicionado a qualquer classificador com MultiOutputClassifier. Essa estratégia consiste em ajustar um classificador por alvo. Isso permite várias classificações de variáveis de destino. O objetivo desta classe é estender estimadores para poder estimar uma série de funções alvo (f1, f2, f3 ..., fn) que são treinadas em uma única matriz preditora X para prever uma série de respostas (y1, y2, y3 …, Yn).

    # Abaixo está um exemplo de classificação multilabel: 

from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import numpy as np
X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
y2 = shuffle(y1, random_state=1)
y3 = shuffle(y1, random_state=2)
Y = np.vstack((y1, y2, y3)).T
n_samples, n_features = X.shape # 10,100
n_outputs = Y.shape[1] # 3
n_classes = 3
forest = RandomForestClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(X, Y).predict(X)

##### 1.12.2.3. ClassifierChain 

    # As cadeias de classificadores (consulte ClassifierChain) são uma forma de combinar vários classificadores binários em um único modelo de vários rótulos que é capaz de explorar correlações entre os alvos.

    # Para um problema de classificação de vários rótulos com N classes, N classificadores binários são atribuídos a um número inteiro entre 0 e N-1. Esses inteiros definem a ordem dos modelos na cadeia. Cada classificador é então ajustado aos dados de treinamento disponíveis mais os verdadeiros rótulos das classes cujos modelos foram atribuídos a um número inferior.

    # Ao fazer a previsão, os rótulos verdadeiros não estarão disponíveis. Em vez disso, as previsões de cada modelo são passadas para os modelos subsequentes na cadeia para serem usadas como recursos.

    # Obviamente, a ordem da cadeia é importante. O primeiro modelo da cadeia não tem informações sobre os outros rótulos, enquanto o último modelo da cadeia possui recursos que indicam a presença de todos os outros rótulos. Em geral, não se sabe a ordenação ideal dos modelos na cadeia, então, normalmente, muitas cadeias ordenadas aleatoriamente são ajustadas e suas previsões são calculadas em conjunto. 




    ## Referências:

    ## Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank, “Classifier Chains for Multi-label Classification”, 2009.