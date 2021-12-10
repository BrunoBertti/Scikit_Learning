########## 1.4.6 Funções do Kernel ##########

    # A função do kernel pode ser qualquer uma das seguintes: 

        # linear: \langle x, x'\rangle
        # polynomial: (\gamma \langle x, x'\rangle + r)^d, onde d é especificado pelo paramatro degree, r por coef0
        # rbf: \exp(-\gamma \|x-x'\|^2), onde é especificado pelo parametro gamma, deve ser maior que 0. 
        # sigmoid \tanh(\gamma \langle x,x'\rangle + r) onde r é especificado por coef0

    # Diferentes kernels são especificados pelo parâmetro kernel: 

from sklearn import svm


linear_svc = svm.SVC(kernel='linear')
print(linear_svc.kernel)
rbf_svc = svm.SVC(kernel='rbf')
print(rbf_svc.kernel)




##### 1.4.6.1 Parametros do Kernel RBF

    # Ao treinar um SVM com o kernel Radial Basis Function (RBF), dois parâmetros devem ser considerados: C e gama. O parâmetro C, comum a todos os kernels SVM, compensa a classificação incorreta dos exemplos de treinamento pela simplicidade da superfície de decisão. Um C baixo torna a superfície de decisão suave, enquanto um C alto visa classificar todos os exemplos de treinamento corretamente. gamma define quanta influência um único exemplo de treinamento tem. Quanto maior for o gama, mais próximos os outros exemplos devem estar para serem afetados.

    # A escolha adequada de C e gama é crítica para o desempenho do SVM. É aconselhável usar GridSearchCV com C e gama espaçados exponencialmente para escolher bons valores. 

    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py

    ## https://scikit-learn.org/stable/auto_examples/svm/plot_svm_nonlinear.html#sphx-glr-auto-examples-svm-plot-svm-nonlinear-py

##### 1.4.6.2 Kernels personalizados 

    # Você pode definir seus próprios kernels fornecendo o kernel como uma função python ou pré-computando a matriz de Gram.

    # Classificadores com kernels personalizados se comportam da mesma maneira que qualquer outro classificador, exceto que: 

        # O campo support_vectors_ agora está vazio, apenas índices de vetores de suporte são armazenados em support_
        #Uma referência (e não uma cópia) do primeiro argumento no método fit () é armazenada para referência futura. Se essa matriz mudar entre o uso de fit () e predizer (), você terá resultados inesperados. 


##### 1.4.6.2.1 Usando funções Python como kernels 

    # Você pode usar seus próprios kernels definidos passando uma função para o parâmetro do kernel.

    # Seu kernel deve tomar como argumentos duas matrizes de forma (n_samples_1, n_features), (n_samples_2, n_features) e retornar uma matriz de kernel de shape (n_samples_1, n_samples_2).

    # O código a seguir define um kernel linear e cria uma instância do classificador que usará esse kernel: 

import numpy as np
from sklearn import svm
def my_kernel(X, Y):
    return np.dot(X, Y.T)

clf = svm.Svc(kernel=my_kernel)


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html#sphx-glr-auto-examples-svm-plot-custom-kernel-py

##### 1.4.6.2.2 Usando a matriz de Gram 
    
    # Você pode passar kernels pré-calculados usando a opção kernel = 'pré-computado'. Você deve então passar a matriz de Gram em vez de X para os métodos de ajuste e previsão. Os valores do kernel entre todos os vetores de treinamento e os vetores de teste devem ser fornecidos: 

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
X, y = make_classification(n_samples=10, random_state=0)
X_train , X_test , y_train, y_test = train_test_split(X, y, random_state=0)
clf = svm.SVC(kernel='precomputed')
# computação de kernel linear 
gram_train = np.dot(X_train, X_train.T)
print(clf.fit(gram_train, y_train))
# prever exemplos de treinamento 
gram_test = np.dot(X_test, X_train.T)
print(clf.predict(gram_test))