########## 1.17.8. Dicas de uso prático ##########

    # O Perceptron multicamadas é sensível ao dimensionamento de recursos, portanto, é altamente recomendável dimensionar seus dados. Por exemplo, dimensione cada atributo no vetor de entrada X para [0, 1] ou [-1, +1], ou padronize-o para ter média 0 e variância 1. Observe que você deve aplicar a mesma escala ao conjunto de teste para resultados significativos. Você pode usar StandardScaler para padronização. 

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
# Não trapaceie - ajuste apenas os dados de treinamento
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
# aplique a mesma transformação para testar os dados 
X_test = scaler.transform(X_test)  


    # Uma abordagem alternativa e recomendada é usar StandardScaler em um pipeline 



    # Encontrar um parâmetro de regularização razoável é melhor feito usando GridSearchCV, geralmente no intervalo 10.0 ** -np.arange (1, 7).

    # Empiricamente, observamos que o L-BFGS converge mais rápido e com melhores soluções em pequenos conjuntos de dados. Para conjuntos de dados relativamente grandes, no entanto, Adam é muito robusto. Geralmente converge rapidamente e oferece um desempenho muito bom. SGD com momentum ou momentum de nesterov, por outro lado, pode ter um desempenho melhor do que esses dois algoritmos se a taxa de aprendizagem for ajustada corretamente.