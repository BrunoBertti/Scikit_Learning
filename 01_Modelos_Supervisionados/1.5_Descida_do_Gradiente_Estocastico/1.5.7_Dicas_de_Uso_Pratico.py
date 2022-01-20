########## 1.5.7. Dicas de uso prático ##########

    # A descida do gradiente estocástico é sensível ao dimensionamento de recursos, portanto, é altamente recomendável dimensionar seus dados. Por exemplo, dimensione cada atributo no vetor de entrada X para [0,1] ou [-1,+1], ou padronize-o para ter média 0 e variância 1. Observe que o mesmo dimensionamento deve ser aplicado ao vetor de teste para obter resultados significativos. Isso pode ser feito facilmente usando StandardScaler: 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)  # Não trapaceie - ajuste apenas nos dados de treinamento
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # aplicar a mesma transformação aos dados de teste

# Ou melhor ainda: use um pipeline! 
from sklearn.pipeline import make_pipeline
est = make_pipeline(StandardScaler(), SGDClassifier())
est.fit(X_train)
est.predict(X_test)

    # Se seus atributos tiverem uma escala intrínseca (por exemplo, frequências de palavras ou recursos de indicadores), a escala não será necessária.

    # Encontrar um termo de regularização razoável é melhor feito usando a pesquisa automática de hiperparâmetros, por exemplo. GridSearchCV ou RandomizedSearchCV, geralmente no intervalo 10.0**-np.arange(1,7).

    # Empiricamente, descobrimos que o SGD converge após observar aproximadamente 10^6 amostras de treinamento. Assim, uma primeira estimativa razoável para o número de iterações é max_iter = np.ceil(10**6 / n), onde n é o tamanho do conjunto de treinamento.

    # Se você aplicar SGD a recursos extraídos usando PCA, descobrimos que geralmente é aconselhável dimensionar os valores de recursos por alguma constante c, de modo que a norma L2 média dos dados de treinamento seja igual a um.

    # Descobrimos que o SGD médio funciona melhor com um número maior de recursos e um eta0 maior    




    ## Referências:

    
    ## “Efficient BackProp” Y. LeCun, L. Bottou, G. Orr, K. Müller - In Neural Networks: Tricks of the Trade 1998. (http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)