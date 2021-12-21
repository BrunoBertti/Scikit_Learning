########## 1.12.1. Classificação multiclasse ##########

    # Aviso: Todos os classificadores no scikit-learn fazem a classificação multiclasse fora da caixa. Você não precisa usar o módulo sklearn.multiclass, a menos que queira experimentar diferentes estratégias multiclasse.
    # A classificação multiclasse é uma tarefa de classificação com mais de duas classes. Cada amostra só pode ser rotulada como uma classe.

    # Por exemplo, a classificação usando recursos extraídos de um conjunto de imagens de frutas, onde cada imagem pode ser de uma laranja, uma maçã ou uma pêra. Cada imagem é uma amostra e é rotulada como uma das 3 classes possíveis. A classificação multiclasse pressupõe que cada amostra é atribuída a um e apenas um rótulo - uma amostra não pode, por exemplo, ser uma pêra e uma maçã.

    # Embora todos os classificadores scikit-learn sejam capazes de classificação multiclasse, os metaestimadores oferecidos por sklearn.multiclass permitem alterar a maneira como lidam com mais de duas classes porque isso pode ter um efeito no desempenho do classificador (em termos de erro de generalização ou computacional exigido Recursos). 


##### 1.12.1.1. Formato de destino

    # As representações multiclasse válidas para type_of_target (y) são: 

    # 1d ou vetor de coluna contendo mais de dois valores discretos. Um exemplo de vetor y para 4 amostras: 

import numpy as np
y = np.array(['apple', 'pear', 'apple', 'orange'])
print(y)

    # Matriz binária densa ou esparsa de forma (n_samples, n_classes) com uma única amostra por linha, onde cada coluna representa uma classe. Um exemplo de uma matriz binária densa e esparsa y para 4 amostras, onde as colunas, em ordem, são maçã, laranja e pêra: 


import numpy as np
from sklearn.preprocessing import LabelBinarizer
y = np.array(['apple', 'pear', 'apple', 'orange'])
y_dense = LabelBinarizer().fit_transform(y)
print(y_dense)
from scipy import sparse
y_sparse = sparse.csr_matrix(y_dense)
print(y_sparse)

    # Para obter mais informações sobre o LabelBinarizer, consulte Transformando o destino de predição (y). 

##### 1.12.1.2. OneVsRestClassifier

    # A estratégia um contra o resto, também conhecida como um contra todos, é implementada em OneVsRestClassifier. A estratégia consiste em ajustar um classificador por classe. Para cada classificador, a classe é ajustada em relação a todas as outras classes. Além de sua eficiência computacional (apenas n_classes classificadores são necessários), uma vantagem desta abordagem é sua interpretabilidade. Como cada classe é representada por um e apenas um classificador, é possível obter conhecimento sobre a classe inspecionando seu classificador correspondente. Esta é a estratégia mais comumente usada e é uma escolha padrão justa.

    # Abaixo está um exemplo de aprendizagem multiclasse usando OvR: 

from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
X, y = datasets.load_iris(return_X_y=True)
OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)

    # OneVsRestClassifier também oferece suporte à classificação multilabel. Para usar este recurso, alimente o classificador com uma matriz indicadora, na qual a célula [i, j] indica a presença do rótulo j na amostra i. 

        # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_multilabel.html

    
    ## Exemplos:

    ##  https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_multilabel.html#sphx-glr-auto-examples-miscellaneous-plot-multilabel-py


##### 1.12.1.3. OneVsOneClassifier

    # OneVsOneClassifier constrói um classificador por par de classes. No momento da previsão, a classe que recebeu mais votos é selecionada. Em caso de empate (entre duas classes com igual número de votos), ele seleciona a classe com a maior confiança de classificação agregada pela soma dos níveis de confiança de classificação por pares calculados pelos classificadores binários subjacentes.

    # Visto que requer o ajuste de n_classes * (n_classes - 1) / 2 classificadores, este método é geralmente mais lento do que um-contra-o-resto, devido à sua complexidade O (n_classes ^ 2). No entanto, este método pode ser vantajoso para algoritmos como algoritmos de kernel que não escalam bem com n_samples. Isso ocorre porque cada problema de aprendizado individual envolve apenas um pequeno subconjunto de dados, enquanto, com um contra o resto, o conjunto de dados completo é usado n_classes vezes. A função de decisão é o resultado de uma transformação monotônica da classificação um contra um.

    # Abaixo está um exemplo de aprendizagem multiclasse usando OvO: 

from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
X, y = datasets.load_iris(return_X_y=True)
OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)



    ## Referências:

    ## “Pattern Recognition and Machine Learning. Springer”, Christopher M. Bishop, page 183, (First Edition)


##### 1.12.1.4. OutputCodeClassifier 

    # Saída para correção de erros As estratégias baseadas em código são bastante diferentes de um contra o resto e um contra um. Com essas estratégias, cada classe é representada em um espaço euclidiano, onde cada dimensão só pode ser 0 ou 1. Outra forma de colocar isso é que cada classe é representada por um código binário (uma matriz de 0 e 1). A matriz que mantém o controle da localização / código de cada classe é chamada de livro de códigos. O tamanho do código é a dimensionalidade do espaço mencionado. Intuitivamente, cada classe deve ser representada por um código o mais exclusivo possível e um bom livro de código deve ser projetado para otimizar a precisão da classificação. Nesta implementação, simplesmente usamos um livro de código gerado aleatoriamente, conforme defendido em 3, embora métodos mais elaborados possam ser adicionados no futuro.

    # No momento do ajuste, um classificador binário por bit no livro de código é ajustado. No momento da previsão, os classificadores são usados ​​para projetar novos pontos no espaço da classe e a classe mais próxima dos pontos é escolhida.

    # Em OutputCodeClassifier, o atributo code_size permite ao usuário controlar o número de classificadores que serão usados. É uma porcentagem do número total de classes.

    # Um número entre 0 e 1 exigirá menos classificadores do que um contra o resto. Em teoria, log2 (n_classes) / n_classes é suficiente para representar cada classe sem ambigüidade. No entanto, na prática, pode não levar a uma boa precisão, pois log2 (n_classes) é muito menor do que n_classes.

    # Um número maior que 1 exigirá mais classificadores do que um contra o resto. Neste caso, alguns classificadores irão, em teoria, corrigir os erros cometidos por outros classificadores, daí o nome “corretor de erros”. Na prática, entretanto, isso pode não acontecer, pois os erros do classificador normalmente serão correlacionados. Os códigos de saída de correção de erros têm um efeito semelhante ao ensacamento.

    # Abaixo está um exemplo de aprendizagem multiclasse usando códigos de saída:


from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
X, y = datasets.load_iris(return_X_y=True)
clf = OutputCodeClassifier(LinearSVC(random_state=0),
                           code_size=2, random_state=0)
clf.fit(X, y).predict(X)


    ## Referências:

    ## “Solving multiclass learning problems via error-correcting output codes”, Dietterich T., Bakiri G., Journal of Artificial Intelligence Research 2, 1995.

    ## 3 “The error coding method and PICTs”, James G., Hastie T., Journal of Computational and Graphical statistics 7, 1998.

    ## “The Elements of Statistical Learning”, Hastie T., Tibshirani R., Friedman J., page 606 (second-edition) 2008.
