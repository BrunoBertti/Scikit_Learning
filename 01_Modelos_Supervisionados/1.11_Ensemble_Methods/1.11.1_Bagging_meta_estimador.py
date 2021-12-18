########## 1.11.1 Bagging metaestimador ##########

    # Em algoritmos de ensemble, métodos de ensacamento formam uma classe de algoritmos que constroem várias instâncias de um estimador de caixa preta em subconjuntos aleatórios do conjunto de treinamento original e então agregam suas predições individuais para formar uma predição final. Esses métodos são usados como uma forma de reduzir a variância de um estimador de base (por exemplo, uma árvore de decisão), introduzindo a randomização em seu procedimento de construção e, em seguida, fazendo um conjunto a partir dela. Em muitos casos, os métodos de bagging constituem uma maneira muito simples de melhorar em relação a um único modelo, sem tornar necessário adaptar o algoritmo de base subjacente. Como eles fornecem uma maneira de reduzir o sobreajuste, os métodos de ensacamento funcionam melhor com modelos fortes e complexos (por exemplo, árvores de decisão totalmente desenvolvidas), em contraste com métodos de reforço que geralmente funcionam melhor com modelos fracos (por exemplo, árvores de decisão rasas). 

    # Os métodos de ensacamento vêm em vários sabores, mas diferem principalmente uns dos outros pela maneira como desenham subconjuntos aleatórios do conjunto de treinamento: 


        # Quando subconjuntos aleatórios do conjunto de dados são desenhados como subconjuntos aleatórios das amostras, esse algoritmo é conhecido como Colar [B1999]. (https://scikit-learn.org/stable/modules/ensemble.html#b1999)

        # Quando as amostras são coletadas com reposição, o método é conhecido como Bagging [B1996]. (https://scikit-learn.org/stable/modules/ensemble.html#b1996)

        # Quando subconjuntos aleatórios do conjunto de dados são desenhados como subconjuntos aleatórios dos recursos, o método é conhecido como Subespaços aleatórios [H1998]. (https://scikit-learn.org/stable/modules/ensemble.html#h1998)

        # Finalmente, quando os estimadores de base são construídos em subconjuntos de amostras e recursos, o método é conhecido como Random Patches [LG2012].  (https://scikit-learn.org/stable/modules/ensemble.html#lg2012)


    # No scikit-learn, os métodos de bagging são oferecidos como um metaestimador BaggingClassifier unificado (resp. BaggingRegressor), tomando como entrada um estimador de base especificado pelo usuário junto com parâmetros que especificam a estratégia para desenhar subconjuntos aleatórios. Em particular, max_samples e max_features controlam o tamanho dos subconjuntos (em termos de amostras e recursos), enquanto bootstrap e bootstrap_features controlam se as amostras e recursos são desenhados com ou sem substituição. Ao usar um subconjunto das amostras disponíveis, a precisão da generalização pode ser estimada com as amostras originais, definindo oob_score = True. Como exemplo, o fragmento abaixo ilustra como instanciar um ensemble de bagging de estimadores de base KNeighborsClassifier, cada um construído em subconjuntos aleatórios de 50% das amostras e 50% dos recursos. 

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html#sphx-glr-auto-examples-ensemble-plot-bias-variance-py


    ## Referência:

    ## B1999L. Breiman, “Pasting small votes for classification in large databases and on-line”, Machine Learning, 36(1), 85-103, 1999.

    ## B1996 L. Breiman, “Bagging predictors”, Machine Learning, 24(2), 123-140, 1996.

    ## H1998 T. Ho, “The random subspace method for constructing decision forests”, Pattern Analysis and Machine Intelligence, 20(8), 832-844, 1998.

    ## LG2012 G. Louppe and P. Geurts, “Ensembles on Random Patches”, Machine Learning and Knowledge Discovery in Databases, 346-361, 2012.