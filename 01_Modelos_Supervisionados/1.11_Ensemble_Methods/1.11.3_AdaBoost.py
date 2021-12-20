########## 1.11.3 AdaBoost ##########


    # O módulo sklearn.ensemble inclui o algoritmo de boosting popular AdaBoost, introduzido em 1995 por Freund e Schapire [FS1995].

    # O princípio central do AdaBoost é ajustar uma sequência de alunos fracos (ou seja, modelos que são apenas ligeiramente melhores do que suposições aleatórias, como pequenas árvores de decisão) em versões repetidamente modificadas dos dados. As previsões de todos eles são então combinadas por meio de uma maioria ponderada de votos (ou soma) para produzir a previsão final. As modificações de dados em cada assim chamada iteração de reforço consistem na aplicação de pesos w1, w1,…, w_N a cada uma das amostras de treinamento. Inicialmente, esses pesos são todos definidos para w_i = 1 / N, de modo que a primeira etapa simplesmente treina um aluno fraco nos dados originais. Para cada iteração sucessiva, os pesos da amostra são modificados individualmente e o algoritmo de aprendizagem é reaplicado aos dados reponderados. Em uma determinada etapa, aqueles exemplos de treinamento que foram preditos incorretamente pelo modelo impulsionado induzido na etapa anterior têm seus pesos aumentados, enquanto os pesos são diminuídos para aqueles que foram preditos corretamente. À medida que as iterações prosseguem, os exemplos difíceis de prever recebem uma influência cada vez maior. Cada aluno fraco subsequente é, portanto, forçado a se concentrar nos exemplos que foram perdidos pelos anteriores na sequência [HTF].

        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html

    # AdaBoost pode ser usado para problemas de classificação e regressão:

        # Para classificação multiclasse, AdaBoostClassifier implementa AdaBoost-SAMME e AdaBoost-SAMME.R [ZZRH2009].

        # Para regressão, AdaBoostRegressor implementa AdaBoost.R2 [D1997]. 



##### 1.11.3.1. Uso 

    # O exemplo a seguir mostra como ajustar um classificador AdaBoost com 100 alunos fracos: 

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

X, y = load_iris(return_X_y=True)
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean())


    # O número de alunos fracos é controlado pelo parâmetro n_estimators. O parâmetro learning_rate controla a contribuição dos alunos fracos na combinação final. Por padrão, os alunos fracos são tocos de decisão. Diferentes alunos fracos podem ser especificados por meio do parâmetro base_estimator. Os principais parâmetros a serem ajustados para obter bons resultados são n_estimators e a complexidade dos estimadores de base (por exemplo, sua profundidade max_depth ou o número mínimo necessário de amostras para considerar uma divisão min_samples_split). 



    ## Exemplos:

        ## Discrete versus Real AdaBoost compares the classification error of a decision stump, decision tree, and a boosted decision stump using AdaBoost-SAMME and AdaBoost-SAMME.R. (https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_hastie_10_2.html#sphx-glr-auto-examples-ensemble-plot-adaboost-hastie-10-2-py)

        ## Multi-class AdaBoosted Decision Trees shows the performance of AdaBoost-SAMME and AdaBoost-SAMME.R on a multi-class problem. (https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-multiclass-py)

        ## Two-class AdaBoost shows the decision boundary and decision function values for a non-linearly separable two-class problem using AdaBoost-SAMME. (https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py)

        ## Decision Tree Regression with AdaBoost demonstrates regression with the AdaBoost.R2 algorithm. (https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_regression.html#sphx-glr-auto-examples-ensemble-plot-adaboost-regression-py)




    ## Referências:

    ## FS1995 Y. Freund, and R. Schapire, “A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting”, 1997. (https://scikit-learn.org/stable/modules/ensemble.html#id9)

    ## ZZRH2009 J. Zhu, H. Zou, S. Rosset, T. Hastie. “Multi-class AdaBoost”, 2009. (https://scikit-learn.org/stable/modules/ensemble.html#id11)

    ## D1997 Drucker. “Improving Regressors using Boosting Techniques”, 1997. (https://scikit-learn.org/stable/modules/ensemble.html#id12)

    ## HTF(1,2,3) T. Hastie, R. Tibshirani and J. Friedman, “Elements of Statistical Learning Ed. 2”, Springer, 2009. 1(https://scikit-learn.org/stable/modules/ensemble.html#id10), 2 (https://scikit-learn.org/stable/modules/ensemble.html#id20), 3(https://scikit-learn.org/stable/modules/ensemble.html#id34)
