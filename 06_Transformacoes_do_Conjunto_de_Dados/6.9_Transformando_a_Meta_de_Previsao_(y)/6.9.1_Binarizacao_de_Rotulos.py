########## 6.9.1. Binarização de rótulos ##########




##### 6.9.1.1. Label Binarizer

    # LabelBinarizer é uma classe de utilitário para ajudar a criar uma matriz indicadora de rótulos a partir de uma lista de rótulos multiclasse: 

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])
#LabelBinarizer()
lb.classes_
#array([1, 2, 4, 6])
lb.transform([1, 6])
#array([[1, 0, 0, 0],
#       [0, 0, 0, 1]])


    # O uso desse formato pode permitir a classificação multiclasse em estimadores que suportam o formato de matriz de indicador de rótulo.

    # Aviso: LabelBinarizer não é necessário se você estiver usando um estimador que já suporta dados multiclasse.

    # Para obter mais informações sobre classificação multiclasse, consulte Classificação multiclasse.


##### 6.9.1.2. MultiLabel Binarizer 


    # No aprendizado multilabel, o conjunto conjunto de tarefas de classificação binária é expresso com um array de indicadores binários de rótulo: cada amostra é uma linha de um array 2d de forma (n_samples, n_classes) com valores binários onde o um, ou seja, os elementos diferentes de zero, corresponde para o subconjunto de rótulos dessa amostra. Uma matriz como np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]]) representa o rótulo 0 na primeira amostra, os rótulos 1 e 2 na segunda amostra , e nenhum rótulo na terceira amostra.

    # Produzir dados multilabel como uma lista de conjuntos de rótulos pode ser mais intuitivo. O transformador MultiLabelBinarizer pode ser usado para converter entre uma coleção de coleções de rótulos e o formato do indicador: 


from sklearn.preprocessing import MultiLabelBinarizer
y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
MultiLabelBinarizer().fit_transform(y)
#array([[0, 0, 1, 1, 1],
#       [0, 0, 1, 0, 0],
#       [1, 1, 0, 1, 0],
#       [1, 1, 1, 1, 1],
#       [1, 1, 1, 0, 0]])


    # Para obter mais informações sobre classificação multirrótulo, consulte Classificação multirrótulo. 