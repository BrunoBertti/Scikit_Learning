########## 1.12. Algoritmos de multiclasse e de saída múltipla ##########

    # Esta seção do guia do usuário cobre a funcionalidade relacionada a problemas de multi-aprendizagem, incluindo multiclasse, multilabel e classificação e regressão de múltiplas saídas.

    # Os módulos nesta seção implementam metaestimadores, que requerem um estimador de base a ser fornecido em seu construtor. Meta-estimadores estendem a funcionalidade do estimador de base para suportar problemas de multi-aprendizado, o que é realizado transformando o problema de multi-aprendizado em um conjunto de problemas mais simples, então ajustando um estimador por problema.

    # Esta seção cobre dois módulos: sklearn.multiclass e sklearn.multioutput. O gráfico abaixo demonstra os tipos de problemas pelos quais cada módulo é responsável e os metaestimadores correspondentes que cada módulo fornece. 


        # https://scikit-learn.org/stable/_images/multi_org_chart.png

    # A tabela a seguir fornece uma referência rápida sobre as diferenças entre os tipos de problemas. Explicações mais detalhadas podem ser encontradas nas seções subsequentes deste guia. 




#                                               Number of targets       Target cardinality     Valid type_of_target                           
# Multiclassclassification                      1                       >2                     ‘multiclass’ 
# Multilabel classification                     >1                      2 (0 or 1)             ‘multilabel-indicator’
# Multiclass-multioutput classification         >1                      >2                     ‘multiclass-multioutput’
# Multioutput regression                        >1                      Continuous             ‘continuous-multioutput


    # Abaixo está um resumo dos estimadores scikit-learn que possuem suporte multi-aprendizado integrado, agrupado por estratégia. Você não precisa dos metaestimadores fornecidos por esta seção se estiver usando um desses estimadores. No entanto, metaestimadores podem fornecer estratégias adicionais além do que está embutido: 

    # Inerentemente multiclasse: 

        # naive_bayes.BernoulliNB

        # tree.DecisionTreeClassifier

        # tree.ExtraTreeClassifier

        # ensemble.ExtraTreesClassifier

        # naive_bayes.GaussianNB

        # neighbors.KNeighborsClassifier

        # semi_supervised.LabelPropagation

        # semi_supervised.LabelSpreading

        # discriminant_analysis.LinearDiscriminantAnalysis

        # svm.LinearSVC (setting multi_class=”crammer_singer”)

        # linear_model.LogisticRegression (setting multi_class=”multinomial”)

        # linear_model.LogisticRegressionCV (setting multi_class=”multinomial”)

        # neural_network.MLPClassifier

        # neighbors.NearestCentroid

        # discriminant_analysis.QuadraticDiscriminantAnalysis

        # neighbors.RadiusNeighborsClassifier

        # ensemble.RandomForestClassifier

        # linear_model.RidgeClassifier

        # linear_model.RidgeClassifierCV

    
    # Multiclasse como um-contra-um: 

        # svm.NuSVC

        # svm.SVC.

        # gaussian_process.GaussianProcessClassifier (setting multi_class = “one_vs_one”)


    # Multiclasse como um contra o resto: 

        # ensemble.GradientBoostingClassifier

        # gaussian_process.GaussianProcessClassifier (setting multi_class = “one_vs_rest”)

        # svm.LinearSVC (setting multi_class=”ovr”)

        # linear_model.LogisticRegression (setting multi_class=”ovr”)

        # linear_model.LogisticRegressionCV (setting multi_class=”ovr”)

        # linear_model.SGDClassifier

        # linear_model.Perceptron

        # linear_model.PassiveAggressiveClassifier

    
    # Suporta multilabel: 

        # tree.DecisionTreeClassifier

        # tree.ExtraTreeClassifier

        # ensemble.ExtraTreesClassifier

        # neighbors.KNeighborsClassifier

        # neural_network.MLPClassifier

        # neighbors.RadiusNeighborsClassifier

        # ensemble.RandomForestClassifier

        # linear_model.RidgeClassifierCV


    # Suporta multiclasse-multioutput: 

        # tree.DecisionTreeClassifier

        # tree.ExtraTreeClassifier

        # ensemble.ExtraTreesClassifier

        # neighbors.KNeighborsClassifier

        # neighbors.RadiusNeighborsClassifier

        # ensemble.RandomForestClassifier