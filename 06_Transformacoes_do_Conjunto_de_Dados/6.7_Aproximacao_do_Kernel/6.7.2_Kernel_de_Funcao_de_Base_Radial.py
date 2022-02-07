########## 6.7.2. Kernel de função de base radial ##########

    # O RBFSampler constrói um mapeamento aproximado para o kernel da função de base radial, também conhecido como Random Kitchen Sinks [RR2007]. Essa transformação pode ser usada para modelar explicitamente um mapa do kernel, antes de aplicar um algoritmo linear, por exemplo, um SVM linear: 


from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 0, 1, 1]
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
clf = SGDClassifier(max_iter=5)
clf.fit(X_features, y)

clf.score(X_features, y)


    # O mapeamento depende de uma aproximação de Monte Carlo para os valores do kernel. A função fit realiza a amostragem de Monte Carlo, enquanto o método transform realiza o mapeamento dos dados. Devido à aleatoriedade inerente ao processo, os resultados podem variar entre diferentes chamadas para a função de ajuste.

    # A função fit recebe dois argumentos: n_components, que é a dimensionalidade de destino da transformação de recurso, e gamma, o parâmetro do kernel RBF. Um n_components maior resultará em uma melhor aproximação do kernel e produzirá resultados mais semelhantes aos produzidos por um kernel SVM. Observe que “ajustar” a função de recurso não depende realmente dos dados fornecidos à função de ajuste. Apenas a dimensionalidade dos dados é usada. Detalhes sobre o método podem ser encontrados em [RR2007].

    # Para um determinado valor de n_components, o RBFSampler geralmente é menos preciso que o Nystroem. O RBFSampler é mais barato de calcular, porém, fazendo uso de espaços de recursos maiores de forma mais eficiente. 



        # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_approximation.html




    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_approximation.html#sphx-glr-auto-examples-miscellaneous-plot-kernel-approximation-py