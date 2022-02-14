########## 6.9.2. Codificação de rótulo  ##########




    # LabelEncoder é uma classe de utilitário para ajudar a normalizar rótulos de forma que contenham apenas valores entre 0 e n_classes-1. Isso às vezes é útil para escrever rotinas Cython eficientes. LabelEncoder pode ser usado da seguinte forma:

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
#LabelEncoder()
le.classes_
#array([1, 2, 6])
le.transform([1, 1, 2, 6])
#array([0, 0, 1, 2])
le.inverse_transform([0, 0, 1, 2])
#array([1, 1, 2, 6])








    # Também pode ser usado para transformar rótulos não numéricos (desde que sejam hashable e comparáveis) em rótulos numéricos: 

le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
#LabelEncoder()
list(le.classes_)
['amsterdam', 'paris', 'tokyo']
le.transform(["tokyo", "tokyo", "paris"])
#array([2, 2, 1])
list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']



                  