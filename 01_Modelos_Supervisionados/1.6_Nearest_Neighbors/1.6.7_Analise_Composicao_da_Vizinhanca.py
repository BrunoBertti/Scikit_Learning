########## 1.6.7 Análise de Componentes de Vizinhança  ##########

    # A análise de componentes de vizinhança (NCA, NeighbourhoodComponentsAnalysis) é um algoritmo de aprendizado de métrica de distância que visa melhorar a precisão da classificação de vizinhos mais próximos em comparação com a distância euclidiana padrão. O algoritmo maximiza diretamente uma variante estocástica da pontuação dos vizinhos mais próximos de um de fora (KNN) no conjunto de treinamento. Ele também pode aprender uma projeção linear de baixa dimensão de dados que pode ser usada para visualização de dados e classificação rápida. 

        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_illustration.html
    
    # Na figura ilustrativa acima, consideramos alguns pontos de um conjunto de dados gerado aleatoriamente. Nós nos concentramos na classificação estocástica KNN do ponto no. 3. A espessura de um link entre a amostra 3 e outro ponto é proporcional à sua distância e pode ser vista como o peso relativo (ou probabilidade) que uma regra estocástica de previsão do vizinho mais próximo atribuiria a este ponto. No espaço original, a amostra 3 tem muitos vizinhos estocásticos de várias classes, portanto, a classe certa não é muito provável. Porém, no espaço projetado apreendido pelo NCA, os únicos vizinhos estocásticos com peso não desprezível são da mesma classe da amostra 3, garantindo que esta última seja bem classificada. Veja a formulação matemática para mais detalhes. 


##### 1.6.7.1 Classificação

    # Combinado com um classificador de vizinhos mais próximos (KNeighborsClassifier), o NCA é atraente para classificação porque pode lidar naturalmente com problemas de várias classes sem qualquer aumento no tamanho do modelo e não introduz parâmetros adicionais que requerem ajuste fino pelo usuário.

    # A classificação NCA mostrou funcionar bem na prática para conjuntos de dados de tamanho e dificuldade variados. Em contraste com os métodos relacionados, como a Análise Discriminante Linear, a NCA não faz suposições sobre as distribuições de classes. A classificação do vizinho mais próximo pode naturalmente produzir limites de decisão altamente irregulares.

    # Para usar este modelo para classificação, é necessário combinar uma instância de NeighborhoodComponentsAnalysis que aprende a transformação ótima com uma instância de KNeighborsClassifier que realiza a classificação no espaço projetado. Aqui está um exemplo usando as duas classes: 

from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.7, random_state=42)
nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(X_train, y_train)
print(nca_pipe.score(X_test, y_test))


        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_classification.html

    # O gráfico mostra os limites de decisão para classificação de vizinho mais próximo e classificação de análise de componentes de vizinhança no conjunto de dados íris, ao treinar e pontuar em apenas dois recursos, para fins de visualização. 


##### 1.6.7.2. Redução de dimensionalidade 

    # O NCA pode ser usado para realizar a redução de dimensionalidade supervisionada. Os dados de entrada são projetados em um subespaço linear que consiste nas direções que minimizam o objetivo do NCA. A dimensionalidade desejada pode ser definida usando o parâmetro n_components. Por exemplo, a figura a seguir mostra uma comparação da redução de dimensionalidade com Análise de Componente Principal (PCA), Análise Discriminante Linear (LinearDiscriminantAnalysis) e Análise de Componente de Vizinhança (NeighbourhoodComponentsAnalysis) no conjunto de dados Digits, um conjunto de dados com tamanho n_ {samples} = 1797 e n_ {features} = 64. O conjunto de dados é dividido em um treinamento e um conjunto de teste de tamanho igual e, em seguida, padronizado. Para avaliação, a precisão da classificação dos 3 vizinhos mais próximos é calculada nos pontos projetados bidimensionais encontrados por cada método. Cada amostra de dados pertence a uma das 10 classes. 

        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_classification.html#sphx-glr-auto-examples-neighbors-plot-nca-classification-py

    ## https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html#sphx-glr-auto-examples-neighbors-plot-nca-dim-reduction-py
    
    ## https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py


##### 1.6.7.3. Formulação matemática 

    # O objetivo do NCA é aprender uma matriz de transformação linear ideal de tamanho (n_components, n_features), o que maximiza a soma de todas as amostras i da probabilidade p_i de que i seja classificado corretamente, ou seja: 

        # \underset{L}{\arg\max} \sum\limits_{i=0}^{N - 1} p_{i}

    # com N = n_samples e p_i a probabilidade de a amostra i ser classificada corretamente de acordo com uma regra estocástica de vizinhos mais próximos no espaço incorporado aprendido:

        # p_{i}=\sum\limits_{j \in C_i}{p_{i j}}

    # onde C_i é o conjunto de pontos na mesma classe que a amostra i, e p_ {i j} é o softmax sobre distâncias euclidianas no espaço embutido: 

        # p_{i j} = \frac{\exp(-||L x_i - L x_j||^2)}{\sum\limits_{k \nei} {\exp{-(||L x_i - L x_k||^2)}}} , \quad p_{i i} = 0


##### 1.6.7.3.1. Distância de Mahalanobis 

    # A NCA pode ser vista como aprendendo uma métrica de distância Mahalanobis (ao quadrado):

        # || L (x_i - x_j) || ^ 2 = (x_i - x_j) ^ TM (x_i - x_j),

    # onde M = L ^ T L é uma matriz simétrica positiva semi-definida de tamanho (n_features, n_features). 


##### 1.6.7.4. Implementação 

    # Esta implementação segue o que é explicado no artigo original 1. Para o método de otimização, ele atualmente usa L-BFGS-B de scipy com um cálculo de gradiente completo em cada iteração, para evitar ajustar a taxa de aprendizagem e fornecer aprendizagem estável.

    # Veja os exemplos abaixo e a docstring de NeighbourhoodComponentsAnalysis.fit para mais informações. 



##### 1.6.7.5. Complexidade


##### 1.6.7.5.1 Treinando

    # NCA armazena uma matriz de distâncias em pares, levando n_samples ** 2 de memória. A complexidade do tempo depende do número de iterações feitas pelo algoritmo de otimização. No entanto, pode-se definir o número máximo de iterações com o argumento max_iter. Para cada iteração, a complexidade do tempo é O (n_components x n_amples x min (n_amples, n_features)). 

##### 1.6.7.5.2 Transformação

    # Aqui, a operação de transformação retorna LX ^ T, portanto, sua complexidade de tempo é igual a n_components * n_features * n_samples_test. Não há complexidade de espaço adicional na operação. 



    ## Referências:

    ## “Neighbourhood Components Analysis”, J. Goldberger, S. Roweis, G. Hinton, R. Salakhutdinov, Advances in Neural Information Processing Systems, Vol. 17, May 2005, pp. 513-520. (http://www.cs.nyu.edu/~roweis/papers/ncanips.pdf)

    ## https://en.wikipedia.org/wiki/Neighbourhood_components_analysis