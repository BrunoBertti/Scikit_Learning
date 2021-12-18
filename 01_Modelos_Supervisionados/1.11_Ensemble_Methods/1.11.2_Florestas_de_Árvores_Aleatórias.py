########## 1.11.2 Florestas de árvores aleatórias ##########

    # O módulo sklearn.ensemble inclui dois algoritmos de média baseados em árvores de decisão aleatórias: o algoritmo RandomForest e o método Extra-Trees. Ambos os algoritmos são técnicas de perturbação e combinação [B1998] projetadas especificamente para árvores. Isso significa que um conjunto diversificado de classificadores é criado pela introdução de aleatoriedade na construção do classificador. A previsão do conjunto é dada como a previsão média dos classificadores individuais.

    # Como outros classificadores, os classificadores de floresta devem ser equipados com duas matrizes: uma matriz X esparsa ou densa de forma (n_samples, n_features) contendo as amostras de treinamento e uma matriz Y de forma (n_samples) contendo os valores alvo (rótulos de classe) para os exemplos de treinamento: 

from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)

    # Como as árvores de decisão, as florestas de árvores também se estendem a problemas de múltiplas saídas (se Y for uma matriz de forma (n_samples, n_outputs)). 


##### 1.11.2.1. Florestas Aleatórias 

    # Em florestas aleatórias (consulte as classes RandomForestClassifier e RandomForestRegressor), cada árvore no conjunto é construída a partir de uma amostra retirada com substituição (ou seja, uma amostra de bootstrap) do conjunto de treinamento.

    # Além disso, ao dividir cada nó durante a construção de uma árvore, a melhor divisão é encontrada em todos os recursos de entrada ou em um subconjunto aleatório de tamanho max_features. (Consulte as diretrizes de ajuste de parâmetro para obter mais detalhes).

    # O objetivo dessas duas fontes de aleatoriedade é diminuir a variância do estimador florestal. Na verdade, as árvores de decisão individuais geralmente exibem alta variação e tendem a se ajustar demais. A aleatoriedade injetada nas florestas produz árvores de decisão com erros de previsão um tanto dissociados. Tirando uma média dessas previsões, alguns erros podem ser cancelados. As florestas aleatórias alcançam uma variação reduzida combinando diversas árvores, às vezes ao custo de um ligeiro aumento no viés. Na prática, a redução da variância é freqüentemente significativa, resultando em um modelo geral melhor.

    # Em contraste com a publicação original [B2001], a implementação do scikit-learn combina classificadores fazendo a média de sua previsão probabilística, em vez de permitir que cada classificador vote em uma única classe. 


##### 1.11.2.2. Árvores Extremamente Randomizadas 

    # Em árvores extremamente aleatórias (consulte as classes ExtraTreesClassifier e ExtraTreesRegressor), a aleatoriedade vai um passo adiante na forma como as divisões são calculadas. Como em florestas aleatórias, um subconjunto aleatório de recursos candidatos é usado, mas em vez de procurar os limites mais discriminativos, os limites são desenhados aleatoriamente para cada recurso candidato e o melhor desses limites gerados aleatoriamente é escolhido como a regra de divisão. Isso geralmente permite reduzir a variância do modelo um pouco mais, às custas de um aumento ligeiramente maior no viés: 

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean())


clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean())


clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean())



        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html



##### 1.11.2.3. Parâmetros 

    # Os principais parâmetros a serem ajustados ao usar esses métodos são n_estimators e max_features. O primeiro é o número de árvores na floresta. Quanto maior, melhor, mas também mais tempo levará para calcular. Além disso, observe que os resultados deixarão de ser significativamente melhores além de um número crítico de árvores. O último é o tamanho dos subconjuntos aleatórios de recursos a serem considerados ao dividir um nó. Quanto menor, maior será a redução da variância, mas também maior será o aumento do viés. Os bons valores padrão empíricos são max_features = None (sempre considerando todos os recursos em vez de um subconjunto aleatório) para problemas de regressão e max_features = "sqrt" (usando um subconjunto aleatório de tamanho sqrt (n_features)) para tarefas de classificação (onde n_features é o número de recursos nos dados). Bons resultados são frequentemente alcançados ao definir max_depth = None em combinação com min_samples_split = 2 (ou seja, ao desenvolver totalmente as árvores). Porém, lembre-se de que esses valores geralmente não são os ideais e podem resultar em modelos que consomem muita RAM. Os melhores valores de parâmetro devem ser sempre validados. Além disso, observe que em florestas aleatórias, as amostras de bootstrap são usadas por padrão (bootstrap = True), enquanto a estratégia padrão para extra-trees é usar todo o conjunto de dados (bootstrap = False). Ao usar a amostragem bootstrap, a precisão da generalização pode ser estimada nas amostras deixadas de fora ou fora do saco. Isso pode ser habilitado definindo oob_score = True. 


    # OBS: O tamanho do modelo com os parâmetros padrão é O (M * N * log (N)), onde M é o número de árvores e N é o número de amostras. Para reduzir o tamanho do modelo, você pode alterar estes parâmetros: min_samples_split, max_leaf_nodes, max_depth e min_samples_leaf. 



##### 1.11.2.4. Paralelização  

    # Por fim, este módulo também apresenta a construção paralela das árvores e o cálculo paralelo das predições por meio do parâmetro n_jobs. Se n_jobs = k, os cálculos são particionados em k jobs e executados em k núcleos da máquina. Se n_jobs = -1, todos os núcleos disponíveis na máquina são usados. Observe que, devido à sobrecarga de comunicação entre processos, a aceleração pode não ser linear (ou seja, o uso de k jobs infelizmente não será k vezes mais rápido). Uma aceleração significativa ainda pode ser alcançada, embora ao construir um grande número de árvores, ou quando construir uma única árvore requer uma boa quantidade de tempo (por exemplo, em grandes conjuntos de dados). 

    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html#sphx-glr-auto-examples-ensemble-plot-forest-iris-py

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances_faces.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-faces-py

    ## https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_multioutput_face_completion.html#sphx-glr-auto-examples-miscellaneous-plot-multioutput-face-completion-py



    ## Referências:

    ## Breiman, “Random Forests”, Machine Learning, 45(1), 5-32, 2001. (https://scikit-learn.org/stable/modules/ensemble.html#id6)

    ## Breiman, “Arcing Classifiers”, Annals of Statistics 1998.(https://scikit-learn.org/stable/modules/ensemble.html#id5)

    ## P. Geurts, D. Ernst., and L. Wehenkel, “Extremely randomized trees”, Machine Learning, 63(1), 3-42, 2006.




##### 1.11.2.5. Avaliação da importância do recurso 

    # A classificação relativa (ou seja, profundidade) de um recurso usado como um nó de decisão em uma árvore pode ser usada para avaliar a importância relativa desse recurso em relação à previsibilidade da variável alvo. Os recursos usados no topo da árvore contribuem para a decisão de previsão final de uma fração maior das amostras de entrada. A fração esperada das amostras para as quais eles contribuem pode, portanto, ser usada como uma estimativa da importância relativa dos recursos. No scikit-learn, a fração de amostras para a qual um recurso contribui é combinada com a diminuição da impureza de dividi-los para criar uma estimativa normalizada do poder preditivo desse recurso.

    # Fazendo a média das estimativas da capacidade preditiva de várias árvores aleatórias, pode-se reduzir a variância de tal estimativa e usá-la para a seleção de características. Isso é conhecido como a diminuição média de impurezas, ou MDI. Consulte [L2014] para obter mais informações sobre MDI e avaliação de importância de recursos com Florestas Aleatórias. 


    ## Aviso: As importâncias do recurso baseado em impurezas calculadas em modelos baseados em árvore sofrem de duas falhas que podem levar a conclusões enganosas. Primeiro, eles são calculados em estatísticas derivadas do conjunto de dados de treinamento e, portanto, não necessariamente nos informam sobre quais recursos são mais importantes para fazer boas previsões no conjunto de dados retido. Em segundo lugar, eles favorecem recursos de alta cardinalidade, ou seja, recursos com muitos valores exclusivos. A importância do recurso de permutação é uma alternativa à importância do recurso com base em impurezas que não sofre com essas falhas. Esses dois métodos de obtenção da importância do recurso são explorados em: Importância da permutação vs Importância aleatória do recurso florestal (MDI). 

    # O exemplo a seguir mostra uma representação codificada por cores das importâncias relativas de cada pixel individual para uma tarefa de reconhecimento de rosto usando um modelo ExtraTreesClassifier. 

        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances_faces.html

    # Na prática, essas estimativas são armazenadas como um atributo denominado feature_importances_ no modelo ajustado. Esta é uma matriz com forma (n_features,) cujos valores são positivos e somam 1,0. Quanto mais alto o valor, mais importante é a contribuição do recurso de correspondência para a função de previsão. 



    ## Exemples:

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances_faces.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-faces-py

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py

    


    ## Referências:

    ## G. Louppe, “Understanding Random Forests: From Theory to Practice”, PhD Thesis, U. of Liege, 2014. (https://scikit-learn.org/stable/modules/ensemble.html#id7)





##### 1.11.2.6. Incorporação de árvores totalmente aleatórias 


    # RandomTreesEmbedding implementa uma transformação não supervisionada dos dados. Usando uma floresta de árvores completamente aleatórias, RandomTreesEmbedding codifica os dados pelos índices das folhas em que um ponto de dados termina. Esse índice é então codificado de maneira um-de-K, levando a uma codificação binária esparsa e de alta dimensão. Essa codificação pode ser calculada com muita eficiência e pode ser usada como base para outras tarefas de aprendizagem. O tamanho e a dispersão do código podem ser influenciados pela escolha do número de árvores e da profundidade máxima por árvore. Para cada árvore do conjunto, a codificação contém uma entrada de um. O tamanho da codificação é no máximo n_estimators * 2 ** max_depth, o número máximo de folhas na floresta.

    # Como os pontos de dados vizinhos têm mais probabilidade de estar na mesma folha de uma árvore, a transformação executa uma estimativa de densidade implícita e não paramétrica. 


    ## Exemplos:

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_embedding.html#sphx-glr-auto-examples-ensemble-plot-random-forest-embedding-py

    ## https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py

    ## https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py



    # Consulte também: As técnicas de aprendizado múltiplo também podem ser úteis para derivar representações não lineares do espaço de recursos; essas abordagens também focam na redução de dimensionalidade. 