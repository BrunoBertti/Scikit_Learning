########## 1.4.1 Classificação ##########

    # SVC, NuSVC and LinearSVC are classes capable of performing binary and multi-class classification on a dataset.

        ## https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html

    
    # SVC e NuSVC são métodos semelhantes, mas aceitam conjuntos de parâmetros ligeiramente diferentes e têm formulações matemáticas diferentes (consulte a seção Formulação matemática). Por outro lado, LinearSVC é outra implementação (mais rápida) da Classificação de Vetores de Suporte para o caso de um kernel linear. Observe que LinearSVC não aceita kernel de parâmetro, pois ele é considerado linear. Ele também não possui alguns dos atributos de SVC e NuSVC, como support_.

    # Como outros classificadores, SVC, NuSVC e LinearSVC tomam como entrada duas matrizes: uma matriz X de forma (n_samples, n_features) contendo as amostras de treinamento e uma matriz y de rótulos de classe (strings ou inteiros), de forma (n_samples):


from sklearn import svm
X = [[0,0], [1,1]]
y = [0,1]
clf = svm.SVC()
print(clf.fit(X,y))

    # Depois de ser treinado, o modelo pode ser usado para prever novos valores: 

print(clf.predict([[2.,2.]]))

    # A função de decisão do SVM (detalhada na formulação matemática) depende de algum subconjunto dos dados de treinamento, chamados de vetores de suporte. Algumas propriedades desses vetores de suporte podem ser encontradas nos atributos support_vectors_, support_ e n_support_: 


# Obtando os vetores de suporte
print(clf.support_vectors_)
# obter índices de vetores de suporte 
print(clf.support_)
# Obter o número de suporte de vetores de cada classe
print(clf.n_support_)


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py

    ## https://scikit-learn.org/stable/auto_examples/svm/plot_svm_nonlinear.html#sphx-glr-auto-examples-svm-plot-svm-nonlinear-py

    ## https://scikit-learn.org/stable/auto_examples/svm/plot_svm_anova.html#sphx-glr-auto-examples-svm-plot-svm-anova-py



##### 1.4.1.1 Classificação de Multi-classes

    # O SVC e o NuSVC implementam a abordagem “um contra um” para classificação de várias classes. No total, n_classes * (n_classes - 1) / 2 classificadores são construídos e cada um treina dados de duas classes. Para fornecer uma interface consistente com outros classificadores, a opção decision_function_shape permite transformar monotonicamente os resultados dos classificadores "um contra um" em uma função de decisão de forma "um contra o resto" (n_samples, n_classes). 

X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
print(clf.fit(X,Y))
dec = clf.decision_function([[1]])
print(dec.shape[1]) # 4 Classes: 4*3/2 = 6
clf.decision_function = 'ovr'
dec = clf.decision_function([[1]])
print(dec.shape[1]) # 4 clasees

    # Por outro lado, o LinearSVC implementa a estratégia multiclasse “um contra o resto”, treinando, assim, modelos de n_classes. 
lin_clf = svm.LinearSVC()
print(lin_clf.fit(X,Y))
dec = lin_clf.decision_function([[1]])
print(dec.shape[1])

    # Veja Formulação matemática para uma descrição completa da função de decisão.

    # Observe que o LinearSVC também implementa uma estratégia multiclasse alternativa, o chamado SVM multiclasse formulado por Crammer e Singer 16, usando a opção multi_class = 'crammer_singer'. Na prática, a classificação um vs resto é geralmente preferida, uma vez que os resultados são em sua maioria semelhantes, mas o tempo de execução é significativamente menor.

    # Para LinearSVC “um vs resto”, os atributos coef_ e intercept_ têm a forma (n_classes, n_features) e (n_classes,) respectivamente. Cada linha dos coeficientes corresponde a um dos classificadores n_classes “one-vs-rest” e similares para os interceptos, na ordem da classe “one”.

    # No caso de SVC e NuSVC “um contra um”, o layout dos atributos é um pouco mais complicado. No caso de um kernel linear, os atributos coef_ e intercept_ têm a forma (n_classes * (n_classes - 1) / 2, n_features) e (n_classes * (n_classes - 1) / 2) respectivamente. Isso é semelhante ao layout do LinearSVC descrito acima, com cada linha agora correspondendo a um classificador binário. A ordem para as classes de 0 a n é “0 vs 1”, “0 vs 2”,… “0 vs n”, “1 vs 2”, “1 vs 3”, “1 vs n”,. . . “N-1 vs n”.

    # A forma de dual_coef_ é (n_classes-1, n_SV) com um layout um tanto difícil de entender. As colunas correspondem aos vetores de suporte envolvidos em qualquer um dos n_classes * (n_classes - 1) / 2 classificadores “um contra um”. Cada um dos vetores de suporte é usado em n_classes - 1 classificadores. As n_classes - 1 entradas em cada linha correspondem aos coeficientes duais para esses classificadores.

    # Isso pode ficar mais claro com um exemplo: considere um problema de três classes com a classe 0 tendo três vetores de suporte v ^ {0} _0, v ^ {1} _0, v ^ {2} _0 e as classes 1 e 2 tendo dois vetores de suporte v ^ {0} _1, v ^ {1} _1 ev ^ {0} _2, v ^ {1} _2 respectivamente. Para cada vetor de suporte v ^ {j} _i há dois coeficientes duais. Vamos chamar o vetor de coeficiente de suporte v ^ {j} _i no classificador entre as classes i e k \ alpha ^ {j} _ {i, k}. Então dual_coef_ se parece com isto: 


        # https://scikit-learn.org/stable/modules/svm.html#:~:text=%CE%B10%2C10,SVs%20of%20class%202


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html#sphx-glr-auto-examples-svm-plot-iris-svc-py


##### 1.4.1.2 Pontuações e probabilidades  

    # O método de função de decisão de SVC e NuSVC fornece pontuações por classe para cada amostra (ou uma pontuação única por amostra no caso binário). Quando a probabilidade da opção do construtor é definida como True, as estimativas de probabilidade de associação de classe (dos métodos predict_proba e predict_log_proba) são habilitadas. No caso binário, as probabilidades são calibradas usando o escalonamento Platt 9: regressão logística nas pontuações do SVM, ajustadas por uma validação cruzada adicional nos dados de treinamento. No caso multiclasse, isso é estendido de acordo com 10. 

    # OBS: O mesmo procedimento de calibração de probabilidade está disponível para todos os estimadores por meio do CalibratedClassifierCV (consulte Calibração de probabilidade). No caso de SVC e NuSVC, este procedimento é embutido no libsvm, que é usado internamente, portanto, não depende do CalibratedClassifierCV do scikit-learn. 

    # A validação cruzada envolvida no dimensionamento Platt é uma operação cara para grandes conjuntos de dados. Além disso, as estimativas de probabilidade podem ser inconsistentes com as pontuações:


        # o “argmax” das pontuações pode não ser o argmax das probabilidades
        # na classificação binária, uma amostra pode ser rotulada por predição como pertencente à classe positiva, mesmo se a saída de prediz_proba for menor que 0,5; e da mesma forma, pode ser rotulado como negativo mesmo se a saída de Predict_proba for maior que 0,5.

    # O método de Platt também é conhecido por ter problemas teóricos. Se as pontuações de confiança forem necessárias, mas não precisam ser probabilidades, é aconselhável definir probabilidade = False e usar função_de_decisão em vez de previsão_proba.

    # Observe que quando decision_function_shape = 'ovr' e n_classes> 2, ao contrário de Decision_function, o método de previsão não tenta quebrar empates por padrão. Você pode definir break_ties = True para que a saída de previsão seja igual a np.argmax (clf.decision_function (...), axis = 1), caso contrário, a primeira classe entre as classes empatadas sempre será retornada; mas tenha em mente que isso vem com um custo computacional. Consulte o Exemplo de desempate SVM para um exemplo de desempate. 
        



##### 1.4.1.3 Problemas desequilibrados 

    # Em problemas onde se deseja dar mais importância a certas classes ou certas amostras individuais, os parâmetros class_weight e sample_weight podem ser usados.

    # SVC (mas não NuSVC) implementa o parâmetro class_weight no método de ajuste. É um dicionário da forma {class_label: value}, onde value é um número de ponto flutuante> 0 que define o parâmetro C da classe class_label como valor C *. A figura abaixo ilustra o limite de decisão de um problema desequilibrado, com e sem correção de peso. 

        # https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html

    # SVC, NuSVC, SVR, NuSVR, LinearSVC, LinearSVR e OneClassSVM implementam também pesos para amostras individuais no método de ajuste por meio do parâmetro sample_weight. Semelhante a class_weight, isso define o parâmetro C para o i-ésimo exemplo como C * sample_weight [i], o que encorajará o classificador a obter essas amostras certas. A figura abaixo ilustra o efeito da ponderação da amostra no limite de decisão. O tamanho dos círculos é proporcional aos pesos da amostra: 

        # https://scikit-learn.org/stable/auto_examples/svm/plot_weighted_samples.html
    


    ## Exemplos

    ## https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-unbalanced-py

    ## https://scikit-learn.org/stable/auto_examples/svm/plot_weighted_samples.html#sphx-glr-auto-examples-svm-plot-weighted-samples-py